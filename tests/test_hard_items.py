"""services/hard_items: the writer's JSON becomes a question without leaking the key."""
import random
import unittest

from services.hard_items import build_question
from services import requirement_check as rc

PLAN = {"decision": "which action first", "findings": ["K 5.3 up from 4.6", "creatinine 1.5 up from 1.1"],
        "near_misses": ["if K were normal", "if BP were low", "if no crackles"],
        "question": "A client with HFrEF ... Which action should the nurse take first?",
        "options": ["Hold the MRA and report the potassium trend", "Hold the loop diuretic and report the weight loss",
                    "Hold the beta blocker and report the blood pressure", "Encourage oral fluids and recheck orthostatics"],
        "correct": 0, "rationale": "Rising potassium on an MRA with falling kidney function risks dysrhythmia.", "topic": "MRA safety"}


class BuildQuestion(unittest.TestCase):
    def test_the_key_lands_wherever_the_draw_puts_it_and_stays_keyed(self):
        positions = set()
        for seed in range(12):
            q = build_question(PLAN, "mcq", random.Random(seed))
            index = q["metadata"]["correctAnswerIndex"]
            positions.add(index)
            self.assertTrue(q["options"][index].endswith("Hold the MRA and report the potassium trend"))
            self.assertEqual(q["answer"], q["options"][index])
            self.assertEqual(rc.key_indices(q), [index])
        self.assertGreater(len(positions), 1)

    def test_select_all_keeps_every_key(self):
        data = {**PLAN, "options": [f"Action {i}" for i in range(6)], "correct": [1, 4]}
        q = build_question(data, "sata", random.Random(3))
        self.assertEqual(q["questionType"], "sata")
        self.assertEqual(sorted(o.split(") ", 1)[1] for o in q["answer"]), ["Action 1", "Action 4"])
        self.assertEqual(len(rc.key_indices(q)), 2)

    def test_malformed_output_is_refused(self):
        for bad in ({**PLAN, "correct": 9}, {**PLAN, "options": PLAN["options"][:3]}, {**PLAN, "question": " "},
                    {**PLAN, "options": ["Same", "same", "Other", "Another"]}, {**PLAN, "correct": "A"}):
            with self.subTest(bad=bad):
                self.assertIsNone(build_question(bad, "mcq", random.Random(0)))
        self.assertIsNone(build_question({**PLAN, "options": [f"x{i}" for i in range(6)], "correct": [0]}, "sata"))

    def test_the_plan_is_kept_for_inspection(self):
        q = build_question(PLAN, "mcq", random.Random(1))
        self.assertEqual(q["hardPlan"]["findings"], PLAN["findings"])
        self.assertEqual(q["correct_blurb"], PLAN["rationale"])


if __name__ == "__main__":
    unittest.main()
