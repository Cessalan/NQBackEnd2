"""
One-off: mark subscriptions that were already past_due before the webhook
learned about that state (2026-10-30). The webhook only sees the transition,
so these never get an event of their own.

Dry run by default; `--yes` writes users/{uid}.billing.paymentIssue.

    venv/Scripts/python.exe tools/backfill_payment_issues.py [--yes]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import firebase_admin
from firebase_admin import credentials
import stripe
from services import stripe_billing


def main(write: bool):
    firebase_admin.initialize_app(credentials.Certificate("service-account-key.json"))
    for sub in stripe.Subscription.list(status="past_due").auto_paging_iter():
        user_ref = stripe_billing._find_user_by_customer(sub.customer)
        if user_ref is None:
            print(f"  skip {sub.id}: no user for {sub.customer}")
            continue
        email = (user_ref.get().to_dict() or {}).get("email")
        if not write:
            print(f"  would mark {email} ({user_ref.id}) {sub.id}")
            continue
        # The failed renewal invoice's own date is the best "since" we have.
        inv = stripe.Invoice.retrieve(sub.latest_invoice)
        issue = stripe_billing.record_payment_issue(
            user_ref, {"id": sub.id, "latest_invoice": sub.latest_invoice}, {"created": inv.created})
        print(f"  marked {email} since {issue['since']:%Y-%m-%d} url={'yes' if issue['invoiceUrl'] else 'NO'}")


if __name__ == "__main__":
    main("--yes" in sys.argv)
