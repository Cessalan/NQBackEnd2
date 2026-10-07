import unittest

from services.material_context import example_formats, excluded_subjects, material_brief


def analysis(**overrides):
    base = {'filename': 'Trauma week 3.pptx', 'goals': [
                {'topic': 'Pre-alert', 'outcome': 'List the information gathered during a prehospital pre-alert',
                 'evidence': {'quote': 'Pre-alert: mechanism, GCS, vitals, ETA'}},
                {'topic': 'Pre-alert', 'outcome': 'List the information gathered during a prehospital pre-alert',
                 'evidence': {'quote': 'duplicate outcome'}},
                {'topic': 'Triage', 'outcome': 'Apply the ABCDE sequence to a trauma arrival',
                 'evidence': {'quote': 'ABCDE on arrival'}}],
            'signals': [
                {'kind': 'emphasis', 'subject': 'Glasgow Coma Scale scoring',
                 'evidence': {'quote': 'MUST KNOW: GCS scoring will be on the exam'}},
                {'kind': 'exclusion', 'subject': 'Burn fluid formulas',
                 'evidence': {'quote': 'Parkland formula is NOT on this exam'}}],
            'examples': [
                {'format': 'sata', 'stem': 'A client arrives after a fall. Which findings require immediate action? Select all that apply.',
                 'options': ['GCS 8', 'HR 72', 'SBP 80', 'Temp 37.1'], 'correctIndices': [0, 2],
                 'wording': 'Short clinical stem, four to five short findings, "Select all that apply"'},
                {'format': 'mcq', 'stem': 'Which action does the nurse take first?', 'options': ['A', 'B', 'C', 'D'],
                 'correctIndices': [], 'wording': 'priority stem'}]}
    base.update(overrides)
    return base


class MaterialBriefTests(unittest.TestCase):
    def test_brief_carries_instructions_markers_objectives_and_examples(self):
        brief = material_brief([analysis()], {'generationInstructions': 'Focus on the first 48 hours', 'emphasis': 'shock'},
                               'make it like my examples')
        self.assertIn("STUDENT'S OWN INSTRUCTIONS", brief)
        self.assertIn('- Focus on the first 48 hours', brief)
        self.assertIn('- make it like my examples', brief)
        self.assertIn('- shock', brief)
        self.assertIn('Marked as important: Glasgow Coma Scale scoring', brief)
        self.assertIn('MUST KNOW: GCS scoring will be on the exam', brief)
        self.assertIn('NOT ON THE EXAM', brief)
        self.assertIn('- Burn fluid formulas', brief)
        self.assertIn('Pre-alert: List the information gathered', brief)
        self.assertEqual(brief.count('List the information gathered'), 1, 'duplicate objectives collapse')
        self.assertIn('Select all that apply', brief)
        self.assertIn('  Answer: A, C', brief)
        self.assertIn('Style: Short clinical stem', brief)
        self.assertNotIn('Answer:\n', brief, 'an example without a known answer states none')

    def test_examples_are_omitted_when_student_opted_out(self):
        brief = material_brief([analysis()], {}, None, match_examples=False)
        self.assertNotIn('EXAMPLE QUESTIONS', brief)
        self.assertIn('Marked as important', brief)

    def test_empty_inputs_produce_no_brief(self):
        self.assertEqual(material_brief([], {}, None), '')
        self.assertEqual(material_brief([{'goals': [], 'signals': [], 'examples': []}], {}, ''), '')

    def test_instruction_only_brief_needs_no_analysis(self):
        brief = material_brief([], {'generationInstructions': 'Only ask about medications'}, None)
        self.assertIn('- Only ask about medications', brief)
        self.assertNotIn('LEARNING OBJECTIVES', brief)

    def test_example_formats_map_to_generator_types_most_common_first(self):
        two_sata = analysis(examples=[{'format': 'sata'}, {'format': 'sata'}, {'format': 'case'}, {'format': 'true_false'}, {'format': 'unknown'}])
        self.assertEqual(example_formats([two_sata]), ['sata', 'casestudy', 'mcq'])
        self.assertEqual(example_formats([]), [])

    def test_excluded_subjects_only_come_from_explicit_exclusion_markers(self):
        self.assertEqual(excluded_subjects([analysis()]), ['Burn fluid formulas'])
        self.assertEqual(excluded_subjects([analysis(signals=[{'kind': 'emphasis', 'subject': 'GCS'}])]), [])

    def test_quiz_path_no_longer_diverts_to_material_practice(self):
        import inspect
        from services import quiz_with_bank
        source = inspect.getsource(quiz_with_bank.stream_quiz_questions)
        self.assertNotIn('stream_material_practice(', source)
        self.assertIn('material_brief(', source)

    def test_objectives_are_grouped_under_main_topics(self):
        a = analysis(goals=[
            {'mainTopic': 'Primary survey', 'topic': 'Airway', 'outcome': 'Assess the airway', 'evidence': {'quote': 'x'}},
            {'mainTopic': 'Primary survey', 'topic': 'Breathing', 'outcome': 'Count the rate', 'evidence': {'quote': 'y'}},
            {'mainTopic': 'Secondary survey', 'topic': 'MIST', 'outcome': 'Take a MIST history', 'evidence': {'quote': 'z'}}])
        brief = material_brief([a], {}, None)
        self.assertIn('BY MAIN TOPIC', brief)
        self.assertLess(brief.index('- Primary survey'), brief.index('  - Airway: Assess the airway'))
        self.assertLess(brief.index('  - Breathing: Count the rate'), brief.index('- Secondary survey'))
        self.assertIn('  - MIST: Take a MIST history', brief)



if __name__ == '__main__':
    unittest.main()
