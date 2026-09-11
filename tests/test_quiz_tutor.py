import unittest
from services.quiz_tutor import explicit_action, sanitize_settings, requested_question_total, is_hint_request, tutor_question_context, protect_hint
from types import SimpleNamespace


class TutorRoutingTests(unittest.TestCase):
    def test_unanswered_question_excludes_solution_context(self):
        question = {'question': 'What does this ask?', 'options': ['Secret correct option'], 'correctIndex': 0, 'rationale': 'Secret rationale', 'metadata': {'correctAnswerIndex': 0}, 'correctOrder': [1]}
        request = SimpleNamespace(question=question, selection={'selectedIndex': None})
        self.assertTrue(is_hint_request(request))
        self.assertEqual(tutor_question_context(request), {'question': 'What does this ask?'})
        request.selection = {'selectedIndex': 1, 'isCorrect': False}
        self.assertFalse(is_hint_request(request))
        self.assertEqual(tutor_question_context(request), question)

    def test_hint_rejects_explicit_solution(self):
        question = {'options': ['Facilitate family presence with staff accompaniment']}
        for reply in ['The correct answer (D) is...', 'Select D.', 'Facilitate family presence with staff accompaniment']:
            self.assertIn('Which word', protect_hint(reply, question, 'en'))
        self.assertEqual(protect_hint('What is the question asking you to decide?', question, 'en'), 'What is the question asking you to decide?')

    def test_help_never_starts_quiz(self):
        for text in ['explain this quiz question', 'idk this one', 'yes break it down for me',
                     'I need a review like tutoring before questions', 'why is B wrong?', 'why should we stop the infusion?']:
            self.assertEqual(explicit_action(text), 'explain')

    def test_stop_and_continue(self):
        self.assertEqual(explicit_action('quit quiz, I am an instructor'), 'stop')
        self.assertEqual(explicit_action('continue please'), 'continue')
        self.assertIsNone(explicit_action('give me more SATA questions'))

    def test_model_settings_are_bounded(self):
        self.assertEqual(sanitize_settings({'requested_total': 999999, 'question_types': ['sata', 'hack', 'sata'], 'difficulty': 'impossible'}),
                         {'requested_total': 200, 'question_types': ['sata']})
        self.assertEqual(sanitize_settings({'requested_total': True}), {})

    def test_requested_counts_are_bounded(self):
        self.assertEqual(requested_question_total('give me 50 NCLEX questions'), 50)
        self.assertEqual(requested_question_total('make 99999 questions'), 200)
        self.assertEqual(requested_question_total('explain page 50'), 5)


if __name__ == '__main__':
    unittest.main()
