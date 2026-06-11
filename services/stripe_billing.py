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

Env vars:
  STRIPE_WEBHOOK_SECRET  (whsec_...)  — required, for signature verification
  STRIPE_SECRET_KEY      (sk_...)     — optional here; set so future API calls work
"""

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
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        return event, None
    except ValueError:
        # Malformed JSON body
        return None, "invalid_payload"
    except stripe.error.SignatureVerificationError:
        # Bad/spoofed signature — reject
        return None, "invalid_signature"


def _set_user_tier(uid: str, tier: str, extra: dict = None):
    """Merge-write usage.tier (and optional extra fields) onto users/{uid}."""
    db = firestore.client()
    data = {"usage": {"tier": tier}}
    if extra:
        data.update(extra)
    # merge=True deep-merges the usage map, preserving windowStart/count.
    db.collection("users").document(uid).set(data, merge=True)


def _downgrade_by_customer(customer_id: str):
    """Find the user linked to a Stripe customer and set tier back to free."""
    if not customer_id:
        return None
    db = firestore.client()
    docs = db.collection("users").where("stripeCustomerId", "==", customer_id).limit(1).stream()
    for doc in docs:
        doc.reference.set({"usage": {"tier": "free"}}, merge=True)
        return doc.id
    return None


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
        _set_user_tier(uid, "pro", {
            "stripeCustomerId": obj.get("customer"),
            "stripeSubscriptionId": obj.get("subscription"),
        })
        return {"status": "upgraded", "uid": uid}

    # ── Subscription ended → revoke Pro ──────────────────────────────────────
    if etype == "customer.subscription.deleted":
        uid = _downgrade_by_customer(obj.get("customer"))
        return {"status": "downgraded", "uid": uid}

    # ── Subscription changed → revoke if it's no longer paying ───────────────
    if etype == "customer.subscription.updated":
        status = obj.get("status")
        if status in ("canceled", "unpaid"):
            uid = _downgrade_by_customer(obj.get("customer"))
            return {"status": "downgraded", "uid": uid, "subStatus": status}
        return {"status": "ignored", "reason": f"sub status {status}"}

    return {"status": "ignored", "reason": etype}
