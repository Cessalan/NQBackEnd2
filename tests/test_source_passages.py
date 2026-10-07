"""Source addressing avoids relying on model transcription of PDF quotations."""
import asyncio
import unittest
from services.document_understanding import (MaterialError, analyse_document,
    merge_sections, source_passages, validate_section)


class PassageTests(unittest.TestCase):
    def setUp(self):
        self.text = ('E\u0301valuation\u00a0(C)ABCDE\r\n\r\n'
                     'ABCDE\nReconnaître la première et la dernière étape.\n'
                     'Quelle étape commence la séquence ?\n'
                     'A. Alpha\nB. Écho\nRéponse : A\n'
                     'Justification : Alpha commence la séquence.\n')
        self.part = {'start': 10500, 'end': 10500 + len(self.text), 'text': self.text}
        self.passages = source_passages(self.part)

    def selection(self, first, last=None):
        return {'start': self.passages[first]['id'], 'end': self.passages[first if last is None else last]['id']}

    def goal(self, outcome='Reconnaître la séquence'):
        return {'topic': 'ABCDE', 'outcome': outcome, 'reasoning': 'recognition', 'source': self.selection(2, 3)}

    def example(self):
        return {'source': self.selection(4, 8), 'stemSource': self.selection(4),
                'optionSources': [self.selection(5), self.selection(6)],
                'answerSource': self.selection(7), 'rationaleSource': self.selection(8),
                'correctIndices': [0], 'format': 'mcq', 'reasoning': 'recognition'}

    def validate(self, **payload):
        return validate_section(payload, self.part, 'evaluation.pdf')

    def test_every_character_and_absolute_offset_survives_long_lines_and_blank_lines(self):
        raw = self.text + ('Mot très long\u00a0' * 400) + '\n' + 'x' * 1805
        part = {**self.part, 'text': raw, 'end': 10500 + len(raw)}
        passages = source_passages(part)
        self.assertEqual(''.join(p['text'] for p in passages), raw)
        self.assertEqual(len({p['id'] for p in passages}), len(passages))
        for p in passages:
            self.assertEqual(p['text'], raw[p['start'] - part['start']:p['end'] - part['start']])
            self.assertLessEqual(len(p['text']), 900)

    def test_model_does_not_need_to_copy_quotes_or_accent_encoding(self):
        goal = {**self.goal(), 'quote': 'A rewritten quote cannot replace source text.'}
        result = self.validate(goals=[goal], signals=[{'kind':'emphasis', 'subject':'ABCDE', 'source':self.selection(0)}])
        self.assertEqual(result['goals'][0]['evidence']['quote'], ''.join(p['text'] for p in self.passages[2:4]))
        self.assertEqual(result['signals'][0]['evidence']['quote'], self.passages[0]['text'])
        self.assertNotIn('source', result['goals'][0])
        self.assertNotIn('quote', result['goals'][0])

    def test_short_heading_is_addressable_with_supporting_context(self):
        result = self.validate(goals=[{**self.goal(), 'source': self.selection(2)}])
        self.assertEqual(result['goals'][0]['evidence']['quote'], 'ABCDE\n')
        self.assertIn('Reconnaître', result['goals'][0]['context']['quote'])

    def test_missing_reversed_other_section_or_blank_references_never_fall_back_to_quote(self):
        for source in [None, {'start':'p0_3','end':'p0_4'}, self.selection(3,2), self.selection(1),
                       {'start':10500,'end':10600}, {'start':self.passages[2]['id'],'end':'invented'}]:
            with self.subTest(source=source), self.assertRaises(MaterialError):
                self.validate(goals=[{**self.goal(), 'source':source, 'quote':self.text}])

    def test_all_example_parts_are_original_source_slices(self):
        example = self.validate(examples=[self.example()])['examples'][0]
        self.assertEqual(example['stem'], self.passages[4]['text'])
        self.assertEqual(example['options'], [self.passages[5]['text'], self.passages[6]['text']])
        self.assertEqual(example['rationale'], self.passages[8]['text'])
        for ref in [example['evidence'],example['stemEvidence'],example['answerEvidence'],
                    example['rationaleEvidence'],*example['optionEvidence']]:
            self.assertEqual(ref['quote'], self.text[ref['start']-10500:ref['end']-10500])

    def test_example_answers_require_evidence_and_valid_option_indices(self):
        for changes in [{'correctIndices':[True]}, {'correctIndices':[2]}, {'correctIndices':[-1]},
                        {'correctIndices':'A'}, {'answerSource':None, 'rationaleSource':None},
                        {'answerSource':None, 'rationaleSource':None,
                         'answerEvidence':{'quote':'Invented answer'}, 'rationaleEvidence':{'quote':'Invented rationale'}},
                        {'stemSource':{'start':'unknown','end':'unknown'}}, {'optionSources':{}}]:
            with self.subTest(changes=changes), self.assertRaises(MaterialError):
                self.validate(examples=[{**self.example(), **changes}])
        example = self.validate(examples=[{**self.example(), 'correctIndices':[],
            'answerSource':None, 'rationaleSource':None}])['examples'][0]
        self.assertEqual(example['correctIndices'], [])

    def test_distinct_outcomes_sharing_a_passage_survive_overlap_deduplication(self):
        result = self.validate(goals=[self.goal('Nommer la première étape'), self.goal('Nommer la dernière étape')])
        merged = merge_sections('evaluation.pdf', self.text, [result, result])
        self.assertEqual(len(merged['goals']), 2)
        self.assertEqual(len({goal['id'] for goal in merged['goals']}), 2)

    def test_both_sections_use_their_own_source_registry_and_preserve_all_text(self):
        raw = self.text * 85
        observed = []
        async def fake(system, payload):
            observed.append(payload)
            passage = next(p for p in payload['sourcePassages'] if 'ABCDE' in p['text'])
            return {'goals':[{'topic':'ABCDE', 'outcome':'Reconnaître la séquence',
                'source':{'start':passage['id'],'end':passage['id']}}]}
        result = asyncio.run(analyse_document(raw, 'evaluation.pdf', call=fake))
        self.assertTrue(result['complete'])
        self.assertGreater(result['sections'], 1)
        self.assertEqual(len(observed), result['sections'])
        for request in observed:
            self.assertEqual(''.join(p['text'] for p in request['sourcePassages']), request['material'])
        for goal in result['goals']:
            ref = goal['evidence']
            self.assertEqual(raw[ref['start']:ref['end']], ref['quote'])

    def test_invalid_source_id_is_repaired_with_the_failed_record_and_same_registry(self):
        observed = []
        async def fake(system, payload):
            observed.append(payload)
            identifier = payload['sourcePassages'][2]['id'] if 'repair' in payload else 'wrong_section_id'
            return {'goals':[{'topic':'ABCDE','outcome':'Reconnaître la séquence',
                'source':{'start':identifier,'end':identifier}}]}
        result = asyncio.run(analyse_document(self.text,'evaluation.pdf',call=fake))
        self.assertTrue(result['complete'])
        self.assertEqual(len(observed),2)
        self.assertIn('goals[0].source',observed[1]['repair']['validationError'])
        self.assertEqual(observed[0]['sourcePassages'],observed[1]['sourcePassages'])


if __name__ == '__main__':
    unittest.main()
