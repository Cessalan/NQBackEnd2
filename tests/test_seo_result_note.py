import unittest

from services.seo_result_note import _brief, _clean, sanitize


class SeoResultNoteTests(unittest.TestCase):
    def test_sanitize_recomputes_score_and_clips(self):
        data = sanitize({
            'title': 'x' * 200,
            'correct': 99,
            'total': 99,
            'items': [
                {'concept': 'Valve anatomy', 'section': 'Heart', 'correct': True, 'why': 'A' * 500},
                {'concept': 'Perfusion', 'correct': False, 'why': 'Missed the priority.'},
                {'concept': '', 'correct': True},
            ],
        })
        self.assertEqual(data['correct'], 1)
        self.assertEqual(data['total'], 2)
        self.assertEqual(len(data['title']), 80)
        self.assertEqual(len(data['items'][0]['why']), 280)

    def test_empty_items_are_rejected(self):
        with self.assertRaises(ValueError):
            sanitize({'items': []})

    def test_clean_keeps_two_sentences(self):
        note = _clean(
            'You got the algebra move. Ratios are the next thing to review. '
            'This sample is small. Do not keep going into endocrine and respiration after that.'
        )
        self.assertEqual(note.count('.'), 2)
        self.assertNotIn('endocrine', note)
        self.assertNotIn('\n', note)

    def test_brief_names_one_win_and_one_miss(self):
        text = _brief({
            'title': 'HESI A2 practice',
            'correct': 1,
            'total': 2,
            'items': [
                {'concept': 'Algebra', 'correct': True, 'why': 'Divided both sides by 3.'},
                {'concept': 'Ratios', 'correct': False, 'why': 'Missed the unit change.'},
                {'concept': 'Endocrine system', 'correct': False, 'why': 'Mixed glands.'},
            ],
        })
        self.assertIn('Algebra', text)
        self.assertIn('Ratios', text)
        self.assertNotIn('Endocrine', text)


if __name__ == '__main__':
    unittest.main()
