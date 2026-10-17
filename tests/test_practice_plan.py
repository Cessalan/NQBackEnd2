"""services/practice_plan: selection safety and prefix-stable scheduling.

The catalogue below is synthetic but shaped like the 2026-10-11 student's two
uploads: a larger coronary artery disease file and a smaller heart failure
file, each with several headings, with medication goals in both.
"""
import unittest
from collections import Counter

from services import practice_plan as plan

CAD = "coronary artery disease.pdf"
CHF = "congestive heart failure .pdf"


def catalogue():
    analyses = []
    layout = {CAD: [("Angina Pectoris", 4, False), ("Medications for angina", 5, True),
                    ("Acute Coronary Syndrome", 5, False), ("Percutaneous Coronary Intervention", 4, False),
                    ("Patient Teaching", 3, False), ("Risk Factors", 3, False)],
              CHF: [("Classification of HF", 3, False), ("Four Foundational Medication Classes for HFrEF", 4, True),
                    ("Heart failure with preserved ejection fraction (HFpEF)", 3, False),
                    ("Pulmonary Edema", 3, False)]}
    for filename, headings in layout.items():
        goals = []
        for heading, count, meds in headings:
            for n in range(count):
                goals.append({"id": f"{filename[:3]}-{heading[:12]}-{n}".replace(" ", "_"), "mainTopic": heading,
                              "topic": f"{heading} item {n}", "outcome": f"Explain {heading.lower()} point {n}",
                              "reasoning": "application", "meds": meds,
                              "context": {"quote": f"passage about {heading} {n}"}})
        analyses.append({"filename": filename, "practiceReady": True, "goals": goals})
    return analyses


MEDS = {"id": "focus:medication-classes", "kind": "focus", "label": "Medication classes", "share": 0.3}


class Selection(unittest.TestCase):
    def setUp(self):
        self.analyses = catalogue()
        self.goals = plan.compact_goals(self.analyses)
        self.med_ids = [g["id"] for a in self.analyses for g in a["goals"] if g["meds"]]

    def test_catalogue_is_every_goal_with_its_passage(self):
        self.assertEqual(len(self.goals), 37)
        self.assertEqual(self.goals[0]["context"], "passage about Angina Pectoris 0")
        self.assertEqual(len(plan.compact_goals(self.analyses, files=[CHF])), 13)

    def test_unreadable_files_contribute_nothing(self):
        self.analyses[1]["practiceReady"] = False
        self.analyses[1]["warnings"] = ["congestive heart failure .pdf: [PDF page 3: UNREADABLE]"]
        self.assertEqual({g["file"] for g in plan.compact_goals(self.analyses)}, {CAD})

    def test_a_deck_with_an_uncertain_image_slide_still_plans_from_its_text(self):
        # Seen 2026-10-11: one cartoon title slide marked UNCERTAIN blocked a
        # whole heart failure deck, and the exam fell back to a 12k excerpt.
        self.analyses[1]["practiceReady"] = False
        self.analyses[1]["warnings"] = ["congestive heart failure .pdf: [PDF page 13: visual description; UNCERTAIN]"]
        self.assertTrue(plan.partly_readable(self.analyses[1]))
        self.assertEqual({g["file"] for g in plan.compact_goals(self.analyses)}, {CAD, CHF})

    def test_a_targeted_topic_matches_the_materials_own_heading(self):
        matched = plan.match_topic(self.goals, "Heart Failure with Preserved Ejection Fraction")
        self.assertEqual(len(matched), 3)
        self.assertTrue(all("HFpEF" in next(g["chapter"] for g in self.goals if g["id"] == m) for m in matched))
        self.assertEqual(plan.match_topic(self.goals, "wound care"), [])

    def test_model_selection_is_validated(self):
        index = {g["id"]: i for i, g in enumerate(self.goals)}
        raw = {"all_in_scope": True,
               "serves": [{"focus": MEDS["id"], "goals": [f"g{index[m]}" for m in self.med_ids] + ["g999", "invented"]}],
               "gaps": []}
        selection = plan.validate_selection(raw, self.goals, [MEDS])
        self.assertEqual(len(selection["inScope"]), 37)
        self.assertEqual(sorted(selection["serves"][MEDS["id"]]), sorted(self.med_ids))

    def test_a_loosely_echoed_focus_id_still_maps_to_her_focus(self):
        # Seen live 2026-10-11: the model answered "medication-classes".
        index = {g["id"]: i for i, g in enumerate(self.goals)}
        for echoed in ("medication-classes", "Medication classes", MEDS["id"]):
            raw = {"all_in_scope": True, "serves": [{"focus": echoed, "goals": [f"g{index[self.med_ids[0]]}"]}]}
            with self.subTest(echoed=echoed):
                self.assertEqual(plan.validate_selection(raw, self.goals, [MEDS])["serves"][MEDS["id"]], [self.med_ids[0]])

    def test_a_focus_nothing_serves_is_a_gap_not_a_substitute(self):
        other = {"id": "focus:dysrhythmias", "kind": "focus", "label": "Dysrhythmias", "share": 0.2}
        selection = plan.validate_selection({"all_in_scope": True, "serves": []}, self.goals, [other])
        self.assertEqual(selection["serves"][other["id"]], [])
        self.assertEqual(selection["gaps"], [{"id": other["id"], "kind": "none"}])

    def test_a_file_with_none_of_her_focus_is_named(self):
        index = {g["id"]: i for i, g in enumerate(self.goals)}
        cad_meds = [m for m in self.med_ids if m.startswith("cor")]
        raw = {"all_in_scope": True, "serves": [{"focus": MEDS["id"], "goals": [f"g{index[m]}" for m in cad_meds]}],
               "gaps": [{"id": MEDS["id"], "note": "A model-written caveat is never shown."}]}
        selection = plan.validate_selection(raw, self.goals, [MEDS])
        self.assertEqual(selection["gaps"], [{"id": MEDS["id"], "kind": "file", "file": CHF}])

    def test_her_full_coverage_has_no_gaps(self):
        index = {g["id"]: i for i, g in enumerate(self.goals)}
        raw = {"all_in_scope": True, "serves": [{"focus": MEDS["id"], "goals": [f"g{index[m]}" for m in self.med_ids]}]}
        self.assertEqual(plan.validate_selection(raw, self.goals, [MEDS])["gaps"], [])

    def test_an_empty_scope_falls_back_to_everything_and_says_so(self):
        selection = plan.validate_selection({"in_scope": ["nope"]}, self.goals, [])
        self.assertEqual(len(selection["inScope"]), 37)
        self.assertEqual(selection["gaps"], [{"id": "scope", "kind": "scope"}])

    def test_when_a_model_is_needed(self):
        self.assertTrue(plan.needs_model_selection([MEDS], False))
        self.assertFalse(plan.needs_model_selection([{"kind": "scope", "derived": True}], False))
        self.assertTrue(plan.needs_model_selection([{"kind": "scope", "label": "only angina"}], False))
        self.assertFalse(plan.needs_model_selection([{"kind": "scope", "label": "HFpEF"}], True))

    def test_the_topic_she_named_is_a_scope(self):
        # The knowledge-quiz button and "a quiz about X" arrive as the topic, not a brief item.
        self.assertTrue(plan.needs_model_selection([], False, scope_text="kidney disorders and filtration"))
        self.assertTrue(plan.needs_model_selection(
            [], False, scope_text="Prokaryotic and eukaryotic cells, Gram-positive and gram-negative cell walls, Biofilms"))
        for whole in (None, "", "the uploaded material", "my notes", "all of my files", "everything"):
            with self.subTest(whole=whole):
                self.assertFalse(plan.needs_model_selection([], False, scope_text=whole))
        self.assertFalse(plan.needs_model_selection([], True, scope_text="HFpEF"))


class Scheduling(unittest.TestCase):
    def setUp(self):
        analyses = catalogue()
        self.goals = plan.compact_goals(analyses)
        med_ids = [g["id"] for a in analyses for g in a["goals"] if g["meds"]]
        self.selection = {"inScope": [g["id"] for g in self.goals], "serves": {MEDS["id"]: med_ids}}
        self.full = plan.schedule(self.goals, self.selection, [MEDS], ["mcq", "sata"], 0, 50)

    def test_prefix_stable_so_batches_and_size_changes_never_replan(self):
        self.assertEqual(plan.schedule(self.goals, self.selection, [MEDS], ["mcq", "sata"], 0, 10), self.full[:10])
        self.assertEqual(plan.schedule(self.goals, self.selection, [MEDS], ["mcq", "sata"], 40, 50), self.full[40:50])
        hundred = plan.schedule(self.goals, self.selection, [MEDS], ["mcq", "sata"], 0, 100)
        self.assertEqual(hundred[:50], self.full)

    def test_every_batch_of_ten_has_both_chapters_and_her_medication_share(self):
        for start in range(0, 50, 10):
            window = self.full[start:start + 10]
            with self.subTest(batch=start // 10):
                self.assertEqual({s["file"] for s in window}, {CAD, CHF})
                self.assertEqual(sum(1 for s in window if s["bucket"] == MEDS["id"]), 3)

    def test_medication_questions_come_from_both_files(self):
        meds = [s for s in self.full if s["bucket"] == MEDS["id"]]
        self.assertEqual({s["file"] for s in meds}, {CAD, CHF})
        self.assertTrue(all(MEDS["id"] in s["serves"] for s in meds))

    def test_questions_spread_over_every_heading_not_the_first_few(self):
        headings = {g["chapter"] for g in self.goals}
        self.assertEqual({s["chapter"] for s in self.full}, headings)

    def test_formats_alternate(self):
        self.assertEqual(Counter(s["format"] for s in self.full), {"mcq": 25, "sata": 25})

    def test_a_goal_used_again_asks_something_new(self):
        reused = [s for s in self.full if s["angle"] > 0]
        self.assertTrue(reused)
        self.assertIn("different aspect", plan.slot_concept(reused[0]))
        self.assertNotIn("different aspect", plan.slot_concept(self.full[0]))

    def test_question_shapes_vary_by_the_kind_of_goal(self):
        priority = {"index": 0, "topic": "Nitrates", "outcome": "Prioritize", "format": "mcq", "reasoning": "priority"}
        self.assertIn("first for this client", plan.slot_concept(priority))
        self.assertIn("sees first", plan.slot_concept({**priority, "index": 1}))
        evaluation = {**priority, "reasoning": "evaluation"}
        self.assertIn("working", plan.slot_concept(evaluation))

    def test_a_hard_request_asks_recall_goals_as_decisions(self):
        recall = {"index": 0, "topic": "Statins", "outcome": "Describe", "format": "mcq", "reasoning": "recall"}
        self.assertIn("accurate", plan.slot_concept(recall))
        self.assertIn("requires follow-up", plan.slot_concept(recall, hard=True))
        sata = {**recall, "format": "sata"}
        self.assertNotIn("Ask", plan.slot_concept(sata, hard=True))

    def test_without_a_focus_the_whole_material_is_the_core(self):
        slots = plan.schedule(self.goals, {"inScope": [g["id"] for g in self.goals]}, [], ["mcq"], 0, 37)
        self.assertEqual(len({s["goalId"] for s in slots}), 37)
        self.assertTrue(all(s["bucket"] is None for s in slots))

    def test_coverage_is_what_the_total_delivers(self):
        summary = plan.coverage(self.full, [MEDS])
        self.assertEqual(summary["focus"][MEDS["id"]], 15)
        self.assertEqual(sum(summary["files"].values()), 50)


if __name__ == "__main__":
    unittest.main()
