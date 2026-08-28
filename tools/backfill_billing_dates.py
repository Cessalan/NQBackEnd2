"""
One-off: backfill users/{uid}.billing for subscribers who converted BEFORE the
webhook started recording timestamps (everyone through 2026-08-27).

Source of truth is the Stripe subscription's `created` — the same instant the
webhook would now stamp as proSince. Writes are additive and idempotent:
existing `billing` maps are left alone unless --force is passed.

    python tools/backfill_billing_dates.py           # dry run (default)
    python tools/backfill_billing_dates.py --apply
"""
import datetime
import os
import sys

import firebase_admin
import stripe
from dotenv import load_dotenv
from firebase_admin import credentials, firestore

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(HERE, ".env"))
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

APPLY = "--apply" in sys.argv
FORCE = "--force" in sys.argv

if not firebase_admin._apps:
    firebase_admin.initialize_app(
        credentials.Certificate(os.path.join(HERE, "FireBaseAccess.json")))
db = firestore.client()

print(f"{'APPLYING' if APPLY else 'DRY RUN'} — pro users with a stripeSubscriptionId\n")
written = skipped = missing = 0

for doc in db.collection("users").where("usage.tier", "==", "pro").stream():
    data = doc.to_dict()
    email = data.get("email") or "(no email)"
    sub_id = data.get("stripeSubscriptionId")
    if not sub_id:
        print(f"  SKIP  {email:34} no stripeSubscriptionId")
        missing += 1
        continue
    if data.get("billing") and not FORCE:
        print(f"  SKIP  {email:34} billing already set (use --force)")
        skipped += 1
        continue
    try:
        sub = stripe.Subscription.retrieve(sub_id)
    except Exception as e:
        # Test-mode subs (the owner's comped account) are not in live mode.
        print(f"  SKIP  {email:34} {str(e)[:60]}")
        missing += 1
        continue

    at = datetime.datetime.fromtimestamp(sub.created, datetime.timezone.utc)
    billing = {
        "status": "pro" if sub.status == "active" else sub.status,
        "proSince": at,
        "firstProAt": at,
        "cancelAtPeriodEnd": bool(sub.cancel_at_period_end),
        # Marks the value as reconstructed, not observed by the webhook.
        "backfilledFrom": "stripe.subscription.created",
    }
    print(f"  {'WRITE' if APPLY else 'would'}  {email:34} proSince={at:%Y-%m-%d %H:%M} UTC  status={billing['status']}")
    if APPLY:
        doc.reference.set({"billing": billing}, merge=True)
        doc.reference.collection("billingEvents").document(f"backfill_{sub_id}").set({
            "type": "backfill.subscription.created",
            "at": at,
            "tier": "pro",
            "subscriptionId": sub_id,
            "customerId": data.get("stripeCustomerId"),
        })
    written += 1

print(f"\n{written} to write, {skipped} already set, {missing} unresolved")
if not APPLY:
    print("Dry run — nothing written. Re-run with --apply.")
