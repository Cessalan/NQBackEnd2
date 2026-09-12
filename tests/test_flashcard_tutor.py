import unittest
from types import SimpleNamespace
from services.flashcard_tutor import tutor_payload, protect_card_hint


class FlashcardTutorTests(unittest.TestCase):
    def test_unrevealed_context_omits_answer_hint_and_previous_discussion(self):
        card = {'front': 'What is this term?', 'back': 'The secret answer', 'hint': 'A revealing hint'}
        body = SimpleNamespace(revealed=False, language='en', message='The secret answer? Give me a hint',
                               history=[{'role': 'assistant', 'content': 'The secret answer'}])
        payload = tutor_payload(body, card)
        self.assertEqual(payload, {'question': 'What is this term?', 'language': 'en',
                                   'task': 'offer a small process hint', 'revealed': False})

    def test_revealed_tutor_uses_saved_answer_and_bounded_discussion(self):
        body = SimpleNamespace(revealed=True, language='fr', message='Explique',
                               history=[{'role': 'user', 'content': 'Pourquoi ?'}] * 20)
        payload = tutor_payload(body, {'front': 'Q', 'back': 'A'})
        self.assertEqual(payload['answer'], 'A')
        self.assertEqual(len(payload['history']), 8)

    def test_answer_leaks_are_replaced_with_process_hint(self):
        for text in ['The answer is 42.', 'It stands for Focused Assessment with Sonography for Trauma.',
                     'Think of Focused Assessment with Sonography.']:
            self.assertIn('before turning', protect_card_hint(text, {'back': 'Focused Assessment with Sonography for Trauma'}, 'en'))
        self.assertEqual(protect_card_hint('What kind of information is being requested?', {'back': 'A definition'}, 'en'),
                         'What kind of information is being requested?')


if __name__ == '__main__':
    unittest.main()
