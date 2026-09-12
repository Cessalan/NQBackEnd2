"""Evidence validation tests that do not require Firebase or model credentials."""
import ast
import pathlib
import re
import unittest

class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code

source = pathlib.Path(__file__).parents[1] / 'services' / 'practice_debrief.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
namespace = {'HTTPException': HTTPException, 're': re}
functions = {'session_evidence', '_short', 'format_reflection'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions], type_ignores=[]), str(source), 'exec'), namespace)
evidence = namespace['session_evidence']
format_reflection = namespace['format_reflection']

class EvidenceTests(unittest.TestCase):
    def quiz(self):
        return {'quizData': [{'question': 'First?', 'options': ['A', 'B']}], 'practice': {'answers': {'0': {'isCorrect': True}}}}

    def test_unanswered_is_not_complete(self):
        q = self.quiz(); q['practice']['answers'] = {}
        with self.assertRaises(HTTPException): evidence(q)

    def test_pending_batch_is_not_complete(self):
        q = self.quiz(); q['practice']['pendingBatch'] = {'count': 1}
        with self.assertRaises(HTTPException): evidence(q)

    def test_retry_does_not_erase_first_mistake(self):
        q = self.quiz(); q['practice']['firstAnswers'] = {'0': {'isCorrect': False, 'selectedIndex': 1}}
        row = evidence(q)[0]
        self.assertFalse(row['first_correct'])
        self.assertTrue(row['latest_selection']['isCorrect'])

    def test_legacy_scores_are_not_called_first_attempts(self):
        self.assertFalse(evidence(self.quiz())[0]['first_known'])

    def test_duplicate_questions_do_not_inflate_score(self):
        q = self.quiz(); q['practice']['questions'] = [{'question': ' first? '}]
        self.assertEqual(len(evidence(q)), 1)

    def test_debrief_is_three_short_scan_lines(self):
        text = format_reflection({'good': 'You noticed the airway priority correctly and stayed focused throughout the entire long question.',
                                  'review': 'Look again at circulation assessments and the complete set of actions required.',
                                  'next': 'First say what the question asks you to decide before reading every option.'}, 'fallback')
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(len(re.sub(r'[-*:]','', line).split()) <= 16 for line in lines))

    def test_invalid_model_shape_uses_short_fallback(self):
        self.assertEqual(format_reflection({'good': 'Only one field'}, 'fallback'), 'fallback')

    def test_zero_score_drops_model_good_claim(self):
        text = format_reflection({'good': 'You understood neurological monitoring.',
                                  'review': 'Look again at question one.',
                                  'next': 'Name the priority before choosing.'}, 'fallback', include_good=False)
        self.assertNotIn('Good', text)
        self.assertNotIn('neurological', text)
        self.assertEqual(len(text.splitlines()), 2)

if __name__ == '__main__': unittest.main()
