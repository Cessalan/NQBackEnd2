"""
stripe_billing.py
Stripe webhook handling for the Pro subscription (monetization).

Flips `users/{uid}.usage.tier` in Firestore:
  - 'pro'  on a successful checkout (keyed by the Firebase uid the frontend
           passes as `client_reference_id` on the Payment Link)
  - 'free' on cancellation / non-paying subscription states

This is the AUTHORITATIVE entitlement source — the only place tier is granted.
The frontend may READ tier but must never write it (lock this down in Firestore
security rules so a user can't self-grant Pro).

It also records WHEN entitlement changed, in two places:
  users/{uid}.billing            — current state (proSince / proEndedAt / status)
  users/{uid}/billingEvents/{id} — immutable log, one doc per Stripe event

Both are stamped with the STRIPE EVENT TIME, never server write time: a webhook
retried an hour later must not report the upgrade as an hour late. The event id
is the log doc id, so retries overwrite rather than duplicate.

Env vars:
  STRIPE_WEBHOOK_SECRET  (whsec_...)  — required, for signature verification
  STRIPE_SECRET_KEY      (sk_...)     — optional here; set so future API calls work
"""

import datetime
import json
import os
import stripe
from firebase_admin import firestore

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def verify_and_parse(payload: bytes, sig_header: str):
    """
    Verify the Stripe signature and parse the event.
    Returns (event, None) on success or (None, reason) on failure.
    """
    if not STRIPE_WEBHOOK_SECRET:
        return None, "webhook_secret_not_configured"
    try:
        # construct_event verifies the signature. We then return the PLAIN
        # parsed JSON, not the stripe Event wrapper: in stripe-python 15.x
        # event["data"]["object"] is a StripeObject that doesn't support
        # dict-style .get(), which handle_event relies on.
        stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        return json.loads(payload.decode("utf-8")), None
    except ValueError:
        # Malformed JSON body
        return None, "invalid_payload"
    except stripe.error.SignatureVerificationError:
        # Bad/spoofed signature — reject
        return None, "invalid_signature"
    except Exception as e:
        # Signed but structurally unexpected payload — reject as malformed
        # rather than letting it bubble up as a 500.
        return None, f"unparseable_event: {type(e).__name__}"


def _event_time(event) -> datetime.datetime:
    """
    Stripe's own timestamp for the event, as a tz-aware datetime (Firestore
    stores it as a real Timestamp, so it sorts and range-queries).

    Deliberately NOT server time: Stripe retries a failed webhook for up to
    3 days, and a retry must record when the customer actually paid, not when
    we finally processed it. Falls back to now() only for a malformed event.
    """
    ts = event.get("created") if hasattr(event, "get") else None
    if not isinstance(ts, (int, float)):
        return datetime.datetime.now(datetime.timezone.utc)
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def _log_billing_event(user_ref, event, fields: dict):
    """
    Append one immutable record to users/{uid}/billingEvents.

    Keyed by the STRIPE EVENT ID, which makes this idempotent for free: Stripe
    retries deliver the same id, so a retry rewrites the same doc instead of
    logging a second upgrade. Read it with a collection-group query on
    'billingEvents' ordered by 'at' to get the whole revenue timeline.
    """
    doc = {"type": event.get("type"), "at": _event_time(event)}
    doc.update(fields)
    user_ref.collection("billingEvents").document(str(event.get("id"))).set(doc)


def _set_user_tier(uid: str, tier: str, event, extra: dict = None, billing: dict = None):
    """
    Merge-write usage.tier (plus optional top-level and billing fields) onto
    users/{uid}, and log the event. Returns the user's DocumentReference.
    """
    db = firestore.client()
    user_ref = db.collection("users").document(uid)
    data = {"usage": {"tier": tier}}
    if extra:
        data.update(extra)
    if billing:
        # merge=True deep-merges, so this only touches the keys named here.
        data["billing"] = dict(billing)
    # merge=True deep-merges the usage map, preserving windowStart/count.
    user_ref.set(data, merge=True)
    return user_ref


def _find_user_by_customer(customer_id: str):
    """Resolve a Stripe customer id to the user's DocumentReference, or None."""
    if not customer_id:
        return None
    db = firestore.client()
    docs = db.collection("users").where("stripeCustomerId", "==", customer_id).limit(1).stream()
    for doc in docs:
        return doc.reference
    return None


def _downgrade_by_customer(customer_id: str, event, sub_status: str = None):
    """
    Find the user linked to a Stripe customer, set tier back to free, stamp
    billing.proEndedAt with the event time, and log it. Returns the uid.
    """
    user_ref = _find_user_by_customer(customer_id)
    if user_ref is None:
        return None
    at = _event_time(event)
    user_ref.set({
        "usage": {"tier": "free"},
        "billing": {
            "status": sub_status or "canceled",
            "proEndedAt": at,
            # Cleared so a returning subscriber's next proSince is unambiguous.
            "cancelAtPeriodEnd": False,
        },
    }, merge=True)
    _log_billing_event(user_ref, event, {
        "tier": "free",
        "subStatus": sub_status,
        "customerId": customer_id,
    })
    return user_ref.id


def create_portal_session(uid: str, return_url: str):
    """
    Create a Stripe Billing Portal session for the user's saved customer.
    The portal is where the user cancels/updates the subscription; the
    resulting customer.subscription.deleted/updated webhook downgrades them.
    Returns (url, None) on success or (None, reason) on failure.
    """
    if not STRIPE_SECRET_KEY:
        return None, "stripe_not_configured"
    db = firestore.client()
    snap = db.collection("users").document(uid).get()
    customer_id = (snap.to_dict() or {}).get("stripeCustomerId") if snap.exists else None
    if not customer_id:
        return None, "no_stripe_customer"
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return session.url, None
    except stripe.error.StripeError as e:
        return None, f"stripe_error: {getattr(e, 'user_message', None) or str(e)}"


def handle_event(event) -> dict:
    """Dispatch a verified Stripe event. Returns a small status dict for logging."""
    etype = event["type"]
    obj = event["data"]["object"]

    # ── Payment succeeded → grant Pro ────────────────────────────────────────
    if etype == "checkout.session.completed":
        uid = obj.get("client_reference_id")
        if not uid:
            return {"status": "ignored", "reason": "no client_reference_id"}
        at = _event_time(event)
        db = firestore.client()
        snap = db.collection("users").document(uid).get()
        prior = ((snap.to_dict() or {}).get("billing") or {}) if snap.exists else {}

        billing = {
            "status": "pro",
            # Start of the CURRENT Pro run. Reset on every upgrade, so a
            # returning subscriber's tenure is measured from their comeback.
            "proSince": at,
            "cancelAtPeriodEnd": False,
        }
        # Never overwritten: the day this person first paid, which is what
        # cohort and lifetime-value questions are actually asking for.
        if not prior.get("firstProAt"):
            billing["firstProAt"] = at

        user_ref = _set_user_tier(uid, "pro", event, extra={
            "stripeCustomerId": obj.get("customer"),
            "stripeSubscriptionId": obj.get("subscription"),
        }, billing=billing)
        _log_billing_event(user_ref, event, {
            "tier": "pro",
            "customerId": obj.get("customer"),
            "subscriptionId": obj.get("subscription"),
            "amountTotal": obj.get("amount_total"),
            "currency": obj.get("currency"),
        })
        return {"status": "upgraded", "uid": uid, "at": at.isoformat()}

    # ── Subscription ended → revoke Pro ──────────────────────────────────────
    if etype == "customer.subscription.deleted":
        uid = _downgrade_by_customer(obj.get("customer"), event, obj.get("status"))
        return {"status": "downgraded", "uid": uid}

    # ── Subscription changed → revoke if it's no longer paying ───────────────
    if etype == "customer.subscription.updated":
        status = obj.get("status")
        if status in ("canceled", "unpaid"):
            uid = _downgrade_by_customer(obj.get("customer"), event, status)
            return {"status": "downgraded", "uid": uid, "subStatus": status}

        # Still paying, but they clicked cancel in the portal: Stripe keeps the
        # subscription alive until the period ends, so entitlement MUST NOT
        # change here. Record it anyway — this is the churn signal, and it
        # arrives up to a month before the deletion event does.
        if obj.get("cancel_at_period_end"):
            user_ref = _find_user_by_customer(obj.get("customer"))
            if user_ref is None:
                return {"status": "ignored", "reason": "unknown customer"}
            ends_at = obj.get("current_period_end")
            user_ref.set({"billing": {
                "cancelAtPeriodEnd": True,
                "canceledAt": _event_time(event),
                "accessEndsAt": (
                    datetime.datetime.fromtimestamp(ends_at, datetime.timezone.utc)
                    if isinstance(ends_at, (int, float)) else None
                ),
            }}, merge=True)
            _log_billing_event(user_ref, event, {
                "tier": "pro",
                "subStatus": status,
                "cancelAtPeriodEnd": True,
                "customerId": obj.get("customer"),
            })
            return {"status": "cancel_scheduled", "uid": user_ref.id}

        return {"status": "ignored", "reason": f"sub status {status}"}

    return {"status": "ignored", "reason": etype}
