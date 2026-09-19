import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.quick_check import QuickCheckRecord
from models.requests import StudyPlanRequest
from services.quick_check import load_quick_check, attach_review_evidence, lesson_focus_terms, lesson_focus_instruction, load_lesson_review_reason


def baseline():
    return dict(schemaVersion=1, checkId='check-1', chatId='chat', funnelId=None,
                completedAt='2026-09-18T14:01:00Z', offered=8, answered=1, completion='ended_early',
                topics=[dict(topic='fabricated', correct=100)], answers=[dict(
                    schemaVersion=1, checkId='check-1', questionId='check-1:0', questionIndex=0,
                    source='quick_check', attempt=1, recordedAt='2026-09-18T14:00:00Z',
                    topic='ABCDE', topicKey='abcde', concept='Airway', question='Choose actions',
                    options=['A', 'B', 'C', 'D'], scenario=None, format='sata', kind='prioritization',
                    difficulty=None, correctIndices=[0, 1], rationale=None, selection=[0],
                    correct=False, partial=True, unsure=False, questionFingerprint='client-fingerprint')])


class QuickCheckTests(unittest.TestCase):
    def test_lesson_focus_matches_preview_and_respects_document_grounding(self):
        reason = attach_review_evidence([dict(type='lesson', topicKey='abcde')], QuickCheckRecord.model_validate(baseline()))[0]['reviewReason']
        self.assertEqual(lesson_focus_terms(reason), ['Airway'])
        prompt = lesson_focus_instruction(reason)
        self.assertIn('Airway', prompt)
        self.assertIn('uploaded document', prompt)
        self.assertIn('rather than inventing content', prompt)
        self.assertEqual(lesson_focus_instruction(None), '')
        self.assertEqual(lesson_focus_terms({**reason, 'priorityMisses': 2}), ['Choosing which action to take first'])

    def test_saved_node_focus_is_rebuilt_and_not_taken_from_stale_metadata(self):
        db = MagicMock()
        chat = db.collection.return_value.document.return_value.get.return_value
        chat.exists = True
        chat.to_dict.return_value = {'userId': 'owner', 'study': {'path': {'quickCheckId': 'check-1', 'nodes': [
            dict(id='lesson1', type='lesson', label='ABCDE', topicKey='abcde', reviewReason={'missedConcepts': ['Stale']})]}}}
        check = db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value
        check.exists = True
        check.to_dict.return_value = baseline()
        self.assertEqual(load_lesson_review_reason('chat', 'lesson1', 'ABCDE', db)['missedConcepts'], ['Airway'])
        self.assertIsNone(load_lesson_review_reason('chat', 'other', 'ABCDE', db))
        self.assertIsNone(load_lesson_review_reason('chat', 'lesson1', 'Different topic', db))

    def test_aggregates_are_rebuilt_from_answers(self):
        record = QuickCheckRecord.model_validate(baseline())
        self.assertEqual(record.diagnostic(), {'ABCDE': 0})
        self.assertEqual(record.topic_results()['abcde']['partial'], 1)

    def test_rejects_forged_grade_and_invalid_selections(self):
        for change in [dict(correct=True), dict(partial=False), dict(selection=[9]),
                       dict(selection=[0, 0]), dict(selection=True), dict(correctIndices=[0, 9])]:
            with self.subTest(change=change):
                data = baseline()
                data['answers'][0].update(change)
                with self.assertRaises(ValueError):
                    QuickCheckRecord.model_validate(data)

    def test_unsure_and_single_answer_grade(self):
        data = baseline()
        data['answers'][0].update(selection='unsure', unsure=True, partial=False)
        self.assertEqual(QuickCheckRecord.model_validate(data).topic_results()['abcde']['unsure'], 1)
        data['answers'][0].update(format='mcq', correctIndices=[1], selection=1, correct=True, unsure=False)
        self.assertEqual(QuickCheckRecord.model_validate(data).diagnostic(), {'ABCDE': 100})

    def test_rejects_foreign_duplicate_and_inconsistent_records(self):
        data = baseline()
        data['answers'].append(copy.deepcopy(data['answers'][0]))
        data['answered'] = 2
        with self.assertRaises(ValueError):
            QuickCheckRecord.model_validate(data)
        for field, value in [('checkId', 'other'), ('topicKey', 'other'), ('questionId', 'other')]:
            data = baseline()
            data['answers'][0][field] = value
            with self.assertRaises(ValueError):
                QuickCheckRecord.model_validate(data)

    def test_review_evidence_uses_exact_topic_and_real_misses(self):
        record = QuickCheckRecord.model_validate(baseline())
        nodes = [dict(type='lesson', topicKey='abcde'), dict(type='lesson', topicKey='abcde advanced'),
                 dict(type='quiz', topicKey='abcde')]
        result = attach_review_evidence(nodes, record)
        self.assertEqual(result[0]['reviewReason']['answered'], 1)
        self.assertEqual(result[0]['reviewReason']['topic'], 'ABCDE')
        self.assertEqual(result[0]['reviewReason']['missedConcepts'], ['Airway'])
        self.assertEqual(result[0]['reviewReason']['priorityMisses'], 1)
        self.assertNotIn('reviewReason', result[1])
        self.assertNotIn('reviewReason', result[2])
        data = baseline()
        data['answers'][0].update(selection=[0, 1], correct=True, partial=False)
        self.assertNotIn('reviewReason', attach_review_evidence([dict(type='lesson', topicKey='abcde')], QuickCheckRecord.model_validate(data))[0])

    def test_load_resolves_owner_and_checks_identity(self):
        db = MagicMock()
        chat = db.collection.return_value.document.return_value.get.return_value
        chat.exists = True
        chat.to_dict.return_value = {'userId': 'owner'}
        check_ref = (db.collection.return_value.document.return_value.collection.return_value
                     .document.return_value.collection.return_value.document.return_value)
        snapshot = check_ref.get.return_value
        snapshot.exists = True
        snapshot.to_dict.return_value = baseline()
        self.assertIsNotNone(load_quick_check('chat', 'check-1', db))
        db.collection.return_value.document.assert_any_call('owner')
        self.assertIsNone(load_quick_check('other-chat', 'check-1', db))
        snapshot.exists = False
        self.assertIsNone(load_quick_check('chat', 'check-1', db))

    def test_legacy_request_and_invalid_reference(self):
        self.assertIsNone(StudyPlanRequest(chat_id='chat').quickCheckId)
        with self.assertRaises(ValueError):
            StudyPlanRequest(chat_id='chat', quickCheckId='../another/check')


if __name__ == '__main__':
    unittest.main()
