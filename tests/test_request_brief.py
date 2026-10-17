"""Replays real student requests (2026-10-11 read) through services/request_brief."""
import unittest

from services import practice_profile as pp
from services import request_brief as rb


def floor(text):
    return rb.floor(text, pp.parse_changes(text).get("requested_total"))


class FloorReadsRealRequests(unittest.TestCase):
    def test_the_paying_students_four_messages(self):
        first = floor("i need you to generate a super tough practice exam nclex styple with so much "
                      "clinicla reasoning on the material i gave you")
        self.assertTrue(first["exam"])
        self.assertIsNone(first["total"])
        self.assertEqual(first["style"]["id"], "style:clinical")
        self.assertIn("nclex", first["style"]["said"])

        second = floor("i need a full exam on both these chapter with medications questions also. the doctor "
                       "expects us to know medication class not medications name and also everything related to "
                       "this class. she gives us very ahrd exams")
        self.assertTrue(second["exam"] and second["full"])
        self.assertEqual(second["said"]["exam"], "i need a full exam on both these chapter with medications questions also")

        self.assertTrue(floor("i said i need a full exam")["full"])
        self.assertFalse(floor("i need to practice for med surg exam on these chapters")["exam"])
        self.assertFalse(floor("i need harder questions on medications")["exam"])

        hundred = floor("i need a 100  questions practice exam these are very easy ")
        self.assertTrue(hundred["exam"])
        self.assertEqual(hundred["total"], 100)

    def test_requests_that_got_fewer_questions_than_asked(self):
        cases = {
            "Give me 60 items quiz": (60, False),
            "I would like you to provide a 50 question practice test covering ideally all likely/main topics": (50, True),
            "make a 20 question NCLEX style quiz with real patient and nursing scenarios": (20, False),
            "I want a 50 questions nclex style nursing exam on the presentations I just posted": (50, True),
            "make me a 100 question mock exam": (100, True),
            "give me a 15 question quiz": (15, False),
            "give me a 15 question hard exam ": (15, True),
            "can you make me a nclex style test using the material i gave you, minimum 20 questions": (20, True),
        }
        for text, (total, exam) in cases.items():
            with self.subTest(text=text):
                result = floor(text)
                self.assertEqual(result["total"], total)
                self.assertEqual(result["exam"], exam)

    def test_mentioning_an_exam_is_not_asking_for_one(self):
        for text in ("help me prepare for my exam",
                     "make me 20 questions nclex style and ATI style i have an exam tommorow",
                     "i have an exam tomorrow so can u help me prepare",
                     "Generate a knowledge test quiz about Maslow's Hierarchy of Needs",
                     "make sure the questions test application not recall",
                     "Create a short targeted practice on: Heart Failure with Preserved Ejection Fraction",
                     "test me on the cardiac meds"):
            with self.subTest(text=text):
                self.assertFalse(floor(text)["exam"])

    def test_style_is_read_from_how_students_write_it(self):
        for text in ("nxlex style quiz", "NGN questions please", "ATI style", "clinical judgment questions",
                     "situational questions on sepsis", "critical thinking questions"):
            with self.subTest(text=text):
                self.assertIsNotNone(floor(text)["style"])
        self.assertIsNone(floor("quiz me on diabetes")["style"])

    def test_a_refused_style_is_not_a_request_for_it(self):
        # The app's own Knowledge Quiz button (found 2026-10-13).
        self.assertIsNone(floor("Generate a knowledge test quiz about Maslow. Use direct factual questions, not clinical scenarios")["style"])
        for text in ("no nclex style please, just definitions", "questions without case-based scenarios",
                     "factual recall rather than clinical reasoning", "pas de mises en situation"):
            with self.subTest(text=text):
                self.assertIsNone(floor(text)["style"])

    def test_refusing_the_style_withdraws_a_saved_one(self):
        result = floor("Use direct factual questions, not clinical scenarios")
        self.assertTrue(result["style_refused"])
        saved, _ = rb.merge([], [], floor_style=floor("nclex style")["style"])
        brief, _ = rb.merge(saved, [], drop=["style"])
        self.assertEqual(brief, [])
        self.assertFalse(floor("make them nclex style")["style_refused"])
        self.assertFalse(floor("quiz me on diabetes")["style_refused"])

    def test_a_negation_elsewhere_does_not_cancel_her_request(self):
        style = floor("no ordering questions, but make them nclex style")["style"]
        self.assertIsNotNone(style)
        self.assertIsNotNone(floor("I don't know much yet. Give me NCLEX style questions")["style"])


class TotalIsWhatSheAskedFor(unittest.TestCase):
    def test_her_number_wins_over_everything(self):
        self.assertEqual(rb.resolve_total(explicit=100, analyzer_total=15, router_total=5), (100, False))

    def test_the_analyzer_reads_numbers_code_cannot(self):
        self.assertEqual(rb.resolve_total(analyzer_total=30, router_total=5), (30, False))

    def test_an_exam_with_no_number_is_a_full_exam_and_says_so(self):
        self.assertEqual(rb.resolve_total(floor_result={"exam": True}, router_total=15), (rb.FULL_EXAM_TOTAL, True))

    def test_more_keeps_the_saved_size(self):
        self.assertEqual(rb.resolve_total(is_continuation=True, saved_total=40, router_total=5), (40, False))

    def test_a_targeted_drill_is_not_a_full_exam(self):
        self.assertEqual(rb.resolve_total(floor_result={"exam": True}, router_total=5, focus=True), (5, True))

    def test_clamped(self):
        self.assertEqual(rb.resolve_total(explicit=900), (rb.MAX_TOTAL, False))


MEDS = {"kind": "focus", "label": "Medication classes", "said": "with medications questions also",
        "instruction": "Include questions on the medication classes in her material.", "share": 0.3}
CLASS_RULE = {"kind": "rule", "label": "Medication class, not drug names",
              "said": "the doctor expects us to know medication class not medications name",
              "instruction": "Ask about the medication class and everything about it, never a specific drug name."}
SCOPE = {"kind": "scope", "label": "Both chapters", "said": "on both these chapter",
         "instruction": "Cover both uploaded chapters."}


class BriefSurvivesTheConversation(unittest.TestCase):
    def test_a_later_message_adds_and_never_erases(self):
        brief, _ = rb.merge([], rb.analyzer_items({"requirements": [SCOPE, MEDS, CLASS_RULE]}), scope_action="change")
        brief, _ = rb.merge(brief, rb.analyzer_items({"requirements": [
            {"kind": "focus", "label": "Harder medication questions", "said": "i need harder questions on medications",
             "instruction": "Make the medication questions hard.", "share": 0.5}]}), scope_action="keep")
        labels = [i["label"] for i in brief]
        self.assertIn("Medication class, not drug names", labels)
        self.assertIn("Both chapters", labels)
        self.assertIn("Harder medication questions", labels)

    def test_a_new_subject_keeps_her_rules_and_style(self):
        brief, _ = rb.merge([], rb.analyzer_items({"requirements": [SCOPE, MEDS, CLASS_RULE]}),
                            floor_style=floor("nclex style")["style"])
        brief, _ = rb.merge(brief, rb.analyzer_items({"requirements": [
            {"kind": "scope", "label": "Wound care", "said": "now quiz me on wound care", "instruction": "Wound care."}]}),
            scope_action="change")
        kinds = {i["kind"]: i["label"] for i in brief}
        self.assertEqual(kinds["scope"], "Wound care")
        self.assertNotIn("focus", kinds)
        self.assertEqual(kinds["rule"], "Medication class, not drug names")
        self.assertEqual(kinds["style"], "Clinical reasoning, NCLEX style")

    def test_a_targeted_drill_is_a_one_off(self):
        brief, _ = rb.merge([], rb.analyzer_items({"requirements": [SCOPE]}))
        brief, request_only = rb.merge(brief, rb.analyzer_items({"requirements": [
            {"kind": "scope", "label": "HFpEF", "said": "Heart Failure with Preserved Ejection Fraction",
             "instruction": "Only HFpEF."}]}), scope_action="focus")
        self.assertEqual([i["label"] for i in brief], ["Both chapters"])
        self.assertEqual([i["label"] for i in request_only], ["HFpEF"])

    def test_she_can_withdraw_one(self):
        brief, _ = rb.merge([], rb.analyzer_items({"requirements": [MEDS, CLASS_RULE]}))
        rule_id = next(i["id"] for i in brief if i["kind"] == "rule")
        brief, _ = rb.merge(brief, [], drop=[rule_id])
        self.assertEqual([i["kind"] for i in brief], ["focus"])

    def test_junk_from_the_model_is_dropped(self):
        self.assertEqual(rb.analyzer_items({"requirements": [{"kind": "vibes", "label": "x"}, {"kind": "rule"}, "x"]}), [])
        focus = rb.analyzer_items({"requirements": [{"kind": "focus", "label": "Meds", "share": 7}]})[0]
        self.assertEqual(focus["share"], 1.0)


class WhatTheWriterAndTheCheckerSee(unittest.TestCase):
    def setUp(self):
        brief, _ = rb.merge([], rb.analyzer_items({"requirements": [SCOPE, MEDS, CLASS_RULE]}),
                            floor_style=floor("nclex style")["style"])
        self.view = rb.request_view(brief, total=50, assumed=True, exam=True, difficulty="hard",
                                    formats=["mcq", "sata"], files=["chf.pdf", "cad.pdf"],
                                    said={"exam": "i need a full exam"})

    def test_order_is_count_scope_focus_style_difficulty_formats_rules_conditions(self):
        self.assertEqual([i["kind"] for i in self.view["items"]],
                         ["count", "scope", "focus", "style", "difficulty", "format", "rule", "conditions"])
        self.assertTrue(self.view["items"][0]["assumed"])
        self.assertEqual(self.view["items"][1]["files"], ["chf.pdf", "cad.pdf"])

    def test_the_writer_gets_her_words_with_precedence(self):
        text = rb.writer_requirements(self.view)
        self.assertIn("outrank", text)
        self.assertIn("never a specific drug name", text)
        self.assertIn('she said: "the doctor expects us to know medication class not medications name"', text)

    def test_a_focus_is_checked_only_on_questions_planned_for_it(self):
        checks = {c["id"]: c["applies"] for c in rb.checkable(self.view)}
        self.assertEqual(checks["style:clinical"], "all")
        self.assertEqual(checks[next(i["id"] for i in self.view["items"] if i["kind"] == "focus")], "serves")
        self.assertEqual(checks[next(i["id"] for i in self.view["items"] if i["kind"] == "rule")], "all")


class HardIsChecked(unittest.TestCase):
    def view(self, difficulty):
        return rb.request_view([], total=10, assumed=False, exam=False, difficulty=difficulty)

    def test_a_hard_request_is_held_to_the_standard_and_told_it(self):
        from services.requirement_check import HARD_ID
        self.assertIn(HARD_ID, {c["id"] for c in rb.checkable(self.view("hard"))})
        self.assertIn("SHE ASKED FOR HARD QUESTIONS", rb.writer_requirements(self.view("hard")))
        self.assertNotIn(HARD_ID, {c["id"] for c in rb.checkable(self.view(None))})
        self.assertEqual(rb.writer_requirements(self.view(None)), "")

    def test_harder_only_when_she_is_getting_most_of_them_right(self):
        self.assertIsNone(rb.escalation({"answered": 4, "correct": 4}))
        self.assertIsNone(rb.escalation({"answered": 10, "correct": 7}))
        self.assertIn("9 of 10", rb.escalation({"answered": 10, "correct": 9}))
        self.assertIn("8 of 10", rb.escalation({"answered": 10, "correct": 8}))
        for junk in (None, {}, {"answered": "10", "correct": 9}, {"answered": 5, "correct": 6}, {"answered": True, "correct": 1}):
            with self.subTest(junk=junk):
                self.assertIsNone(rb.escalation(junk))


if __name__ == "__main__":
    unittest.main()
