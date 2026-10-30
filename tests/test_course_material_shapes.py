"""Regression for framework objects emitted by the real upload pipeline."""
import ast
from pathlib import Path
import unittest


def summarize_materials():
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath(
        'services/course_intelligence.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'summarize_materials')
    namespace = {'CONFIDENCE_VERIFIED': 'verified'}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'materials', 'exec'), namespace)
    return namespace['summarize_materials']


def group(title, count):
    return {'title': title, 'outcomes': [f'{title} outcome {i}' for i in range(count)]}


class MaterialShapeTests(unittest.TestCase):
    def test_upload_framework_objects_and_legacy_labels_are_supported(self):
        result = summarize_materials()({'lecture.pdf': {
            'topics': ['ABCDE Assessment'], 'concepts': ['Primary survey'],
            'frameworks': [{'id': 'adpie', 'name': 'Nursing Process', 'confidence': 'high'},
                           'Nursing Process', {'id': 'sbar'}, {}, None],
        }})
        self.assertEqual(result['frameworks'], ['Nursing Process', 'sbar'])
        self.assertEqual(result['topic_count'], 1)
        self.assertEqual(result['file_count'], 1)

    def test_a_topic_is_weighed_by_its_own_goals_not_its_files(self):
        # 2026-10-30: every topic of a long deck shared the file's concepts,
        # tied at the same emphasis, and the alphabet picked "Ambulation" first.
        topics = [group('Oxygen therapy', 9), group('Ambulation', 2), group('Atelectasis', 5)]
        result = summarize_materials()({'oxygenation.pptx': {
            'topics': [g['title'] for g in topics], 'mainTopics': topics,
            'concepts': [o for g in topics for o in g['outcomes']]}})
        self.assertEqual([t['topic'] for t in result['topics']], ['Oxygen therapy', 'Atelectasis', 'Ambulation'])
        self.assertEqual(result['topics'][0]['key_points'][:2], ['Oxygen therapy outcome 0', 'Oxygen therapy outcome 1'])
        self.assertEqual(result['topics'][0]['emphasis'], 100)

    def test_ties_keep_her_files_order_not_the_alphabet(self):
        result = summarize_materials()({
            'tissue.pptx': {'topics': ['TISSUE INTEGRITY', 'BATH ADMINISTRATION'], 'concepts': []},
            'oxygenation.pptx': {'topics': ['Oxygenation', 'Atelectasis'], 'concepts': []}})
        self.assertEqual([t['topic'] for t in result['topics']],
                         ['TISSUE INTEGRITY', 'BATH ADMINISTRATION', 'Oxygenation', 'Atelectasis'])

    def test_a_topic_in_more_of_her_files_still_ranks_first(self):
        result = summarize_materials()({
            'a.pptx': {'topics': ['Deep topic', 'Assessment'], 'mainTopics': [group('Deep topic', 9), group('Assessment', 1)]},
            'b.pptx': {'topics': ['Assessment'], 'mainTopics': [group('Assessment', 1)]}})
        self.assertEqual(result['topics'][0]['topic'], 'Assessment')
        self.assertEqual(result['topics'][0]['file_count'], 2)


if __name__ == '__main__':
    unittest.main()
