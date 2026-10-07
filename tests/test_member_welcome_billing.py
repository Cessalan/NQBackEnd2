import datetime
import unittest
from unittest.mock import MagicMock, patch
from services.stripe_billing import handle_event


class MemberWelcomeBillingTests(unittest.TestCase):
    def test_checkout_marks_new_activation_and_retries_keep_the_same_welcome_key(self):
        db = MagicMock()
        user = db.collection.return_value.document.return_value
        user.get.return_value.exists = True
        first_paid = datetime.datetime.fromtimestamp(10, datetime.timezone.utc)
        user.get.return_value.to_dict.return_value = {
            "billing": {"firstProAt": first_paid}, "memberWelcomeSeen": "v1:100000"
        }
        event = {"id": "evt_test", "type": "checkout.session.completed", "created": 200,
                 "data": {"object": {"client_reference_id": "alice", "customer": "cus_test",
                                     "subscription": "sub_test", "payment_status": "paid"}}}
        with patch('services.stripe_billing.firestore.client', return_value=db):
            for _ in range(2):
                self.assertEqual(handle_event(event)["status"], "upgraded")
        writes = [call.args[0] for call in user.set.call_args_list]
        self.assertEqual(len(writes), 2)
        self.assertEqual(writes[0], writes[1])
        self.assertEqual(writes[0]["billing"]["welcomeVersion"], 1)
        self.assertEqual(writes[0]["billing"]["proSince"].timestamp(), 200)
        self.assertNotIn("firstProAt", writes[0]["billing"])
        self.assertNotIn("memberWelcomeSeen", writes[0])
        self.assertEqual(writes[0]["usage"]["tier"], "pro")

    def test_renewals_do_not_create_new_welcome_activations(self):
        with patch('services.stripe_billing.firestore.client') as client:
            handle_event({"type": "invoice.paid", "data": {"object": {}}})
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
