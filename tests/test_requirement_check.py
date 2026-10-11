"""services/requirement_check: the pure half of the checking pass."""
import unittest

from services import requirement_check as rc

RULE = {"id": "rule:class-not-names", "test": "Ask about the medication class, never a specific drug name.", "applies": "all"}
STYLE = {"id": "style:clinical", "test": "A patient scenario requiring clinical judgment.", "applies": "all"}
MEDS = {"id": "focus:medication-classes", "test": "About a medication class in her material.", "applies": "serves"}


class Checks(unittest.TestCase):
    def test_a_focus_applies_only_to_questions_planned_for_it(self):
        self.assertEqual([c["id"] for c in rc.applicable([RULE, STYLE, MEDS])], [RULE["id"], STYLE["id"]])
        self.assertEqual(len(rc.applicable([RULE, STYLE, MEDS], serves=[MEDS["id"]])), 3)

    def test_render_reads_every_format_without_markup(self):
        sata = {"questionType": "sata", "question": "<b>Which</b> apply?", "options": ["A) One", "B) Two"],
                "answer": ["A) One"]}
        text = rc.render(sata)
        self.assertIn("Question: Which apply?", text)
        self.assertIn("Keyed answer: A) One", text)
        case = {"questionType": "casestudy", "question": "Order these", "caseStudy": {"nursesNotes": "<p>BP 80/50</p>"},
                "options": [{"id": "i1", "text": "Assess airway"}], "correctOrder": [{"id": "i1", "text": "Assess airway"}]}
        self.assertIn("nursesNotes: BP 80/50", rc.render(case))
        self.assertIn("- Assess airway", rc.render(case))

    def test_a_skipped_check_is_not_counted_as_met(self):
        results = rc.validate({"results": [{"id": RULE["id"], "meets": True, "why": "meets"}]}, [RULE, STYLE])
        self.assertEqual(rc.verified_map(results), {RULE["id"]: True, STYLE["id"]: None})

    def test_unusable_output_is_none_never_verified(self):
        self.assertIsNone(rc.validate({"results": "yes"}, [RULE]))
        self.assertIsNone(rc.validate(None, [RULE]))

    def test_failures_become_a_rewrite_note(self):
        results = rc.validate({"results": [
            {"id": RULE["id"], "meets": False, "why": "Option B names digoxin."},
            {"id": STYLE["id"], "meets": True, "why": "meets"}]}, [RULE, STYLE])
        failed = rc.failures(results)
        self.assertEqual(failed, [{"id": RULE["id"], "why": "Option B names digoxin."}])
        self.assertIn("Option B names digoxin.", rc.repair_note(failed))

    def test_prompt_lists_each_requirement_with_its_id(self):
        prompt = rc.build_prompt({"question": "Q?", "options": ["A) x"]}, [RULE])
        self.assertIn(f"[{RULE['id']}] {RULE['test']}", prompt)


HARD = {"id": rc.HARD_ID, "test": rc.HARD_STANDARD, "applies": "all"}

# Q3 of the "very hard" exam of 2026-10-11: the key is by far the longest.
BETA_BLOCKER = {"questionType": "mcq",
    "question": "A 72-year-old woman with coronary artery disease is due to receive a beta-blocker class medication. "
                "She reports dizziness; her heart rate is 38/min and blood pressure is 82/46 mm Hg. Which action should the nurse take FIRST?",
    "options": ["A) Recheck her blood pressure and apical pulse in 15 minutes, then decide whether to administer the dose.",
                "B) Document the findings, keep her on bed rest, and reassess her vital signs at the next routine observation.",
                "C) Administer the scheduled dose with food and monitor for improvement in her dizziness.",
                "D) Withhold the dose and promptly notify the provider, initiating the facility’s urgent escalation response for her symptomatic bradycardia and hypotension."],
    "answer": "D) Withhold the dose and promptly notify the provider, initiating the facility’s urgent escalation response for her symptomatic bradycardia and hypotension."}
BALANCED = {"questionType": "mcq", "question": "Which first?",
    "options": ["A) Hold the dose and recheck the apical pulse", "B) Give the dose and recheck the apical pulse",
                "C) Hold the dose and obtain a 12-lead ECG", "D) Give the dose and report the heart rate"],
    "metadata": {"correctAnswerIndex": 2}}


class HardGates(unittest.TestCase):
    def test_the_key_is_found_in_every_writer_shape(self):
        self.assertEqual(rc.key_indices(BETA_BLOCKER), [3])
        self.assertEqual(rc.key_indices(BALANCED), [2])
        self.assertEqual(rc.key_indices({"options": ["A) x", "B) y", "C) z"], "answer": "B"}), [1])
        sata = {"questionType": "sata", "options": ["A) One", "B) Two", "C) Three"], "answer": ["A) One", "C) Three"]}
        self.assertEqual(rc.key_indices(sata), [0, 2])

    def test_a_stand_out_longest_key_is_a_cue(self):
        self.assertIn("longest", rc.length_cue(BETA_BLOCKER))
        self.assertIsNone(rc.length_cue(BALANCED))

    def test_select_all_keys_longer_than_the_rest_are_a_cue(self):
        sata = {"questionType": "sata", "options": ["A) Assess lung sounds and oxygen saturation every hour", "B) Rest",
                "C) Weigh daily at the same time on the same scale", "D) Fluids"],
                "answer": ["A) Assess lung sounds and oxygen saturation every hour", "C) Weigh daily at the same time on the same scale"]}
        self.assertIsNotNone(rc.length_cue(sata))

    def test_a_key_picked_from_the_options_alone_fails_hard_whatever_the_judge_says(self):
        meets, why = rc.hard_verdict(GOOD, None, guessed=True)
        self.assertFalse(meets)
        self.assertIn("options alone", why)

    def test_length_fails_hard_even_when_the_judge_was_down(self):
        meets, why = rc.hard_verdict(None, rc.length_cue(BETA_BLOCKER), False)
        self.assertFalse(meets)
        self.assertIn("longest", why)

    def test_without_the_judge_and_without_a_cue_hard_is_unknown_not_passed(self):
        self.assertEqual(rc.hard_verdict(None, None, False), (None, ""))

    def test_the_judges_findings_are_held_to_fixed_thresholds(self):
        self.assertEqual(rc.hard_verdict(GOOD), (True, "meets"))
        for change, words in (({"decision": False}, "recall"), ({"findings_to_combine": 1}, "two findings"),
                              ({"extreme_finding": True}, "extreme"), ({"giveaway_options": ["B", "D"]}, "B, D"),
                              ({"lone_active_key": True}, "only option"), ({"key_names_conclusion": True}, "names the conclusion")):
            with self.subTest(change=change):
                meets, why = rc.hard_verdict({**GOOD, **change})
                self.assertFalse(meets)
                self.assertIn(words, why)

    def test_the_beta_blocker_question_of_2026_10_11_fails(self):
        judged = {**GOOD, "extreme_finding": True, "giveaway_options": ["C"], "lone_active_key": True}
        meets, why = rc.hard_verdict(judged, rc.length_cue(BETA_BLOCKER), guessed=True)
        self.assertFalse(meets)


GOOD = {"decision": True, "findings_to_combine": 3, "extreme_finding": False, "giveaway_options": [],
        "lone_active_key": False, "key_names_conclusion": False, "note": "fine"}


if __name__ == "__main__":
    unittest.main()
