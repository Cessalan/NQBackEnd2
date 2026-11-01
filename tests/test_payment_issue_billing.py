import datetime
import unittest
from unittest.mock import MagicMock, patch
from firebase_admin import firestore
from services.stripe_billing import handle_event


def _updated(status, was=None, cancel=False, created=500):
    data = {"object": {"id": "sub_1", "customer": "cus_1", "status": status,
                       "latest_invoice": "in_1", "cancel_at_period_end": cancel}}
    if was:
        data["previous_attributes"] = {"status": was}
    return {"id": f"evt_{status}_{created}", "type": "customer.subscription.updated", "created": created, "data": data}


class PaymentIssueBillingTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.user = MagicMock(id="alice")
        self.user.get.return_value.exists = True
        self.user.get.return_value.to_dict.return_value = {"billing": {"status": "pro"}}
        self.db.collection.return_value.where.return_value.limit.return_value.stream.return_value = [
            MagicMock(reference=self.user)]
        self.invoice = MagicMock(hosted_invoice_url="https://invoice.stripe.com/i/abc", next_payment_attempt=900)
        patches = [
            patch("services.stripe_billing.firestore.client", return_value=self.db),
            patch("services.stripe_billing.STRIPE_SECRET_KEY", "sk_test"),
            patch("services.stripe_billing.stripe.Invoice.retrieve", return_value=self.invoice),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _billing_writes(self):
        return [c.args[0]["billing"] for c in self.user.set.call_args_list if "billing" in c.args[0]]

    def test_past_due_keeps_pro_and_records_the_invoice_link(self):
        result = handle_event(_updated("past_due", was="active"))
        self.assertEqual(result["status"], "payment_issue")
        issue = self._billing_writes()[0]["paymentIssue"]
        self.assertEqual(issue["invoiceUrl"], "https://invoice.stripe.com/i/abc")
        self.assertEqual(issue["since"].timestamp(), 500)
        self.assertEqual(issue["nextAttemptAt"].timestamp(), 900)
        for call in self.user.set.call_args_list:
            self.assertNotIn("usage", call.args[0])

    def test_a_replayed_failure_keeps_the_first_date(self):
        first = datetime.datetime.fromtimestamp(100, datetime.timezone.utc)
        self.user.get.return_value.to_dict.return_value = {"billing": {"paymentIssue": {"since": first}}}
        handle_event(_updated("past_due", created=800))
        self.assertEqual(self._billing_writes()[0]["paymentIssue"]["since"], first)

    def test_recovery_clears_the_issue(self):
        result = handle_event(_updated("active", was="past_due"))
        self.assertEqual(result["status"], "payment_recovered")
        self.assertIs(self._billing_writes()[0]["paymentIssue"], firestore.DELETE_FIELD)

    def test_an_ordinary_active_update_writes_nothing(self):
        self.assertEqual(handle_event(_updated("active"))["status"], "ignored")
        self.user.set.assert_not_called()

    def test_past_due_with_a_scheduled_cancel_records_both(self):
        result = handle_event(_updated("past_due", was="active", cancel=True))
        self.assertEqual(result["status"], "cancel_scheduled")
        writes = self._billing_writes()
        self.assertIn("paymentIssue", writes[0])
        self.assertTrue(writes[1]["cancelAtPeriodEnd"])

    def test_deletion_clears_the_issue(self):
        handle_event({"id": "evt_d", "type": "customer.subscription.deleted", "created": 5,
                      "data": {"object": {"customer": "cus_1", "status": "canceled"}}})
        self.assertIs(self._billing_writes()[0]["paymentIssue"], firestore.DELETE_FIELD)

    def test_a_stripe_outage_still_records_the_failure(self):
        import stripe
        with patch("services.stripe_billing.stripe.Invoice.retrieve", side_effect=stripe.error.APIConnectionError("down")):
            handle_event(_updated("past_due", was="active"))
        issue = self._billing_writes()[0]["paymentIssue"]
        self.assertIsNone(issue["invoiceUrl"])
        self.assertEqual(issue["since"].timestamp(), 500)


if __name__ == "__main__":
    unittest.main()
