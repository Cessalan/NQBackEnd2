import unittest
from unittest.mock import MagicMock
from services.stripe_billing import _log_billing_event


class SeoBillingTests(unittest.TestCase):
    def test_paid_checkout_preserves_amount_and_attribution_with_stripe_id(self):
        user = MagicMock()
        snap = user.collection.return_value.document.return_value.get.return_value
        snap.exists = True
        snap.to_dict.return_value = {"landingPage": "nclex-study-plan", "keywordCluster": "NCLEX", "funnelId": "seo_test", "source": "organic_search"}
        event = {"id": "evt_test", "type": "checkout.session.completed", "created": 100, "data": {"object": {"payment_status": "paid"}}}
        _log_billing_event(user, event, {"amountTotal": 2900, "currency": "usd"})
        saved = user.collection.return_value.document.return_value.set.call_args.args[0]
        self.assertEqual(saved["amountTotal"], 2900)
        self.assertEqual(saved["seoConversion"]["landingPage"], "nclex-study-plan")
        user.collection.return_value.document.assert_any_call("evt_test")

    def test_unpaid_or_zero_value_checkout_is_not_a_paid_conversion(self):
        for status, amount in [("unpaid", 2900), ("paid", 0)]:
            user = MagicMock()
            event = {"id": "evt_test", "type": "checkout.session.completed", "data": {"object": {"payment_status": status}}}
            _log_billing_event(user, event, {"amountTotal": amount})
            saved = user.collection.return_value.document.return_value.set.call_args.args[0]
            self.assertNotIn("seoConversion", saved)

    def test_missing_attribution_does_not_block_billing_log(self):
        user = MagicMock()
        user.collection.return_value.document.return_value.get.side_effect = RuntimeError("offline")
        event = {"id": "evt_test", "type": "checkout.session.completed", "data": {"object": {"payment_status": "paid"}}}
        _log_billing_event(user, event, {"amountTotal": 2900})
        self.assertTrue(user.collection.return_value.document.return_value.set.called)


if __name__ == '__main__':
    unittest.main()
