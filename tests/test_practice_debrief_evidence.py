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
functions = {'session_evidence', '_sentences', 'format_note'}
namespace['NOTE_MAX_WORDS'] = 60
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions], type_ignores=[]), str(source), 'exec'), namespace)
evidence = namespace['session_evidence']
format_note = namespace['format_note']

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

    def test_note_is_two_plain_paragraphs_with_no_labels_or_score(self):
        text = format_note({'strength': 'You have the cardiac arrest call down: no central pulse means CPR straight away.',
                            'focus': 'The one to work on is the order of the primary survey. Catastrophic bleeding comes before the airway, and two of your misses started at A.'},
                           'fallback')
        paragraphs = text.split('\n\n')
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(paragraphs[0].startswith('You have the cardiac arrest call down'))
        self.assertTrue(paragraphs[1].endswith('started at A.'))
        self.assertNotIn('**', text)
        self.assertNotIn('Good:', text)

    def test_sentences_are_never_cut_mid_thought(self):
        long_second = 'word ' * 70
        text = format_note({'strength': '', 'focus': 'Bleeding comes before the airway. ' + long_second + 'end.'}, 'fallback')
        self.assertEqual(text, 'Bleeding comes before the airway.')
        self.assertEqual(format_note({'strength': '', 'focus': long_second}, 'fallback'), 'fallback')

    def test_invalid_model_shape_uses_fallback(self):
        self.assertEqual(format_note({'strength': 'Only a strength'}, 'fallback'), 'fallback')
        self.assertEqual(format_note('not json', 'fallback'), 'fallback')

    def test_zero_score_drops_model_strength_claim(self):
        text = format_note({'strength': 'You understood neurological monitoring.',
                            'focus': 'Name the priority before choosing.'}, 'fallback', include_strength=False)
        self.assertNotIn('neurological', text)
        self.assertEqual(text, 'Name the priority before choosing.')

    def test_french_closing_quotes_count_as_sentence_end(self):
        text = format_note({'strength': '', 'focus': 'Revois l’ordre « C, puis A ». Le saignement passe avant les voies aériennes.'}, 'fallback')
        self.assertEqual(text, 'Revois l’ordre « C, puis A ». Le saignement passe avant les voies aériennes.')

if __name__ == '__main__': unittest.main()
