"""Later batches of a planned request: the blueprint reference and her rebuilt requirements."""
import unittest

from services import practice_api, practice_profile as pp, request_brief as rb

PLAN = "req-" + "a" * 28


class PlanReference(unittest.TestCase):
    def test_a_valid_reference_is_kept(self):
        self.assertEqual(practice_api.request_plan_ref({"request_plan": {"id": PLAN, "offset": 50}}),
                         {"id": PLAN, "offset": 50})

    def test_a_reference_names_a_document_so_it_is_validated(self):
        for bad in ({"id": "../users/x"}, {"id": "req-zz"}, {"id": 7}, "req-x", None):
            with self.subTest(bad=bad):
                self.assertIsNone(practice_api.request_plan_ref({"request_plan": bad}))

    def test_a_bad_offset_is_zero(self):
        for offset in (-1, True, "10", 10 ** 9):
            with self.subTest(offset=offset):
                self.assertEqual(practice_api.request_plan_ref({"request_plan": {"id": PLAN, "offset": offset}})["offset"], 0)


class RebuiltRequirements(unittest.TestCase):
    def setUp(self):
        brief, _ = rb.merge([], rb.analyzer_items({"requirements": [
            {"kind": "rule", "label": "Medication class, not drug names", "said": "class not names",
             "instruction": "Never name a specific drug."}]}), floor_style=rb.floor("nclex style")["style"])
        self.profile = pp.normalize({"brief": brief, "requestedTotal": 50, "difficulty": "hard",
                                     "source": {"kind": "uploads", "files": ["a.pdf", "b.pdf"]}})

    def test_a_later_batch_is_held_to_the_chats_saved_rules(self):
        view = practice_api.request_view_for_batch(self.profile, {"requested_total": 50, "difficulty": "hard"},
                                                   {"exam": True})
        kinds = [i["kind"] for i in view["items"]]
        self.assertIn("rule", kinds)
        self.assertIn("style", kinds)
        self.assertIn("conditions", kinds)
        # Her rule and style, plus the hard standard: the chat saved "hard".
        self.assertEqual({c["id"] for c in rb.checkable(view)},
                         {i["id"] for i in view["items"] if i["kind"] in ("rule", "style")} | {"difficulty"})

    def test_clinical_style_keeps_every_batch_a_scenario(self):
        view = practice_api.request_view_for_batch(self.profile, {}, {"quiz_mode": "knowledge"})
        self.assertEqual(practice_api.planned_quiz_mode({"quiz_mode": "knowledge"}, view), "nclex")
        self.assertEqual(practice_api.planned_quiz_mode({"quiz_mode": "knowledge"}, {"items": []}), "knowledge")
        self.assertEqual(practice_api.planned_quiz_mode({"quiz_mode": "made-up"}, {"items": []}), "nclex")



class StemsStartWithTheQuestion(unittest.TestCase):
    def test_an_echoed_slot_number_is_removed(self):
        from services.planned_practice import strip_number
        for raw in ("Question 5. Une femme de 32 ans", "question 12: A client", "Q3) A client", "  Question 2 - A client"):
            with self.subTest(raw=raw):
                self.assertFalse(strip_number({"question": raw})["question"].lower().startswith(("question", "q3")))
        self.assertEqual(strip_number({"question": "Question the order if the client"})["question"],
                         "Question the order if the client")
        self.assertIsNone(strip_number(None))


if __name__ == "__main__":
    unittest.main()
