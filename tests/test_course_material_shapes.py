"""Regression for framework objects emitted by the real upload pipeline."""
import ast
from pathlib import Path
import unittest


class MaterialShapeTests(unittest.TestCase):
    def test_upload_framework_objects_and_legacy_labels_are_supported(self):
        tree = ast.parse(Path(__file__).resolve().parents[1].joinpath(
            'services/course_intelligence.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'summarize_materials')
        namespace = {'CONFIDENCE_VERIFIED': 'verified'}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'materials', 'exec'), namespace)
        result = namespace['summarize_materials']({'lecture.pdf': {
            'topics': ['ABCDE Assessment'], 'concepts': ['Primary survey'],
            'frameworks': [{'id': 'adpie', 'name': 'Nursing Process', 'confidence': 'high'},
                           'Nursing Process', {'id': 'sbar'}, {}, None],
        }})
        self.assertEqual(result['frameworks'], ['Nursing Process', 'sbar'])
        self.assertEqual(result['topic_count'], 1)
        self.assertEqual(result['file_count'], 1)


if __name__ == '__main__':
    unittest.main()
