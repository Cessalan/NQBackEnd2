import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException
from services import admin_api, admin_study


class Store:
    def __init__(self, records):
        self.records = records

    def collection(self, name):
        return Collection(self, name)


class Document:
    def __init__(self, db, path):
        self.db, self.path, self.id = db, path, path.split('/')[-1]

    def get(self):
        return SimpleNamespace(id=self.id, exists=self.path in self.db.records,
                               to_dict=lambda: self.db.records.get(self.path))

    def collection(self, name):
        return Collection(self.db, self.path + '/' + name)


class Collection:
    def __init__(self, db, path):
        self.db, self.path, self.count, self.cursor, self.filters = db, path, 10000, '', []

    def document(self, key):
        return Document(self.db, self.path + '/' + key)

    def where(self, key, op, value):
        self.filters.append((key, value))
        return self

    def order_by(self, field):
        assert field == '__name__'
        return self

    def limit(self, count):
        self.count = count
        return self

    def select(self, fields):
        return self

    def start_after(self, cursor):
        self.cursor = cursor['__name__'].id
        return self

    def stream(self):
        docs = []
        for key, data in sorted(self.db.records.items()):
            if key.rsplit('/', 1)[0] != self.path or key.split('/')[-1] <= self.cursor:
                continue
            if any(data.get(field) != value for field, value in self.filters):
                continue
            docs.append(self.document(key.split('/')[-1]).get())
        return iter(docs[:self.count])


class AdminStudyTests(unittest.TestCase):
    def setUp(self):
        self.db = Store({
            'chats/old': {'title': 'Existing plan', 'isStudySession': True, 'userId': 'student',
                          'study': {'path': {'nodes': [{'id': 'n1', 'status': 'done'}]}}},
            'users/student': {'email': 'student@example.com', 'stripeCustomerId': 'private-billing'},
            'users/student/studyPerformance/old': {'history': [{'nodeId': 'n1', 'correct': 1, 'total': 3}]},
            'chats/old/messages/a': {'id': 'logical-a', 'nodeId': 'n1', 'type': 'study_quiz',
                'studyContent': {'questions': [{'question': 'Stored question'}]},
                'quizProgress': {'firstAttemptAnswers': {'0': {'correct': False}}, 'firstAttemptStatuses': {'0': 'incorrect'}},
                'content': 'Original feedback', 'privateExtra': 'excluded'},
            'chats/old/messages/a/reasoningDiscussions/0': {'history': [{'role': 'user', 'content': 'My reasoning'}, {'role': 'assistant', 'content': 'Saved coaching'}]},
            'chats/old/messages/a/reasoningSummaries/latest': {'focus': {'skill': 'Check the first action'}, 'summaries': []},
            'chats/other/messages/secret': {'content': 'Another session'},
        })

    def test_existing_plan_reads_owner_performance_without_billing(self):
        result = admin_study.plan_detail(self.db, 'old')
        self.assertTrue(result['readOnly'])
        self.assertEqual(result['performance']['history'][0]['correct'], 1)
        self.assertEqual(result['chat']['title'], 'Existing plan')
        self.assertNotIn('stripeCustomerId', result['owner'])

    def test_detail_includes_saved_questions_and_nested_discussions_only(self):
        result = admin_study.message_detail(self.db, 'old', 'a')
        self.assertEqual(result['message']['id'], 'a')
        self.assertEqual(result['message']['storedId'], 'logical-a')
        self.assertEqual(result['message']['studyContent']['questions'][0]['question'], 'Stored question')
        self.assertEqual(result['discussions'][0]['history'][1]['content'], 'Saved coaching')
        self.assertEqual(result['summaries'][0]['focus']['skill'], 'Check the first action')
        self.assertNotIn('privateExtra', result['message'])
        self.assertFalse(result['reasoningTruncated'])

    def test_pagination_keeps_undated_records_and_does_not_duplicate(self):
        for i in range(101):
            self.db.records[f'chats/old/messages/m{i:03}'] = {'type': 'study_quiz'}
        first = admin_study.message_page(self.db, 'old')
        second = admin_study.message_page(self.db, 'old', first['cursor'])
        ids = [m['id'] for m in first['items'] + second['items']]
        self.assertEqual(len(ids), 102)
        self.assertEqual(len(set(ids)), 102)
        self.assertIsNone(second['cursor'])
        self.assertEqual(first['items'][0]['answerRecords']['statuses'], {'0': 'incorrect'})

    def test_directory_has_no_date_cutoff(self):
        self.db.records['chats/old']['createdAt'] = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.db.records['chats/plain'] = {'isStudySession': False}
        self.assertEqual([p['id'] for p in admin_study.list_plans(self.db)['items']], ['old'])

    def test_missing_and_cross_session_records_are_not_generated(self):
        for call in [lambda: admin_study.plan_detail(self.db, 'missing'), lambda: admin_study.message_detail(self.db, 'old', 'secret')]:
            with self.assertRaises(HTTPException) as cm:
                call()
            self.assertEqual(cm.exception.status_code, 404)
        with self.assertRaises(HTTPException):
            admin_study.plan_detail(self.db, '../other')

    def test_routes_reject_anonymous_requests_before_reading(self):
        app = FastAPI()
        app.include_router(admin_api.router)

        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                for suffix in ['', '/old', '/old/messages', '/old/messages/a']:
                    response = await client.get('/admin/workspace/study-plans' + suffix)
                    self.assertEqual(response.status_code, 401)
        # require_admin imports Firebase before examining the header.
        with patch('services.admin_api.database', side_effect=AssertionError('Anonymous read')):
            asyncio.run(check())

    def test_authorized_route_serializes_historical_timestamp(self):
        app = FastAPI()
        app.include_router(admin_api.router)
        app.dependency_overrides[admin_api.require_admin] = lambda: {'uid': 'admin', 'role': 'admin'}
        self.db.records['chats/old']['createdAt'] = datetime(2025, 1, 1, tzinfo=timezone.utc)

        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.get('/admin/workspace/study-plans/old')
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()['chat']['createdAt'].startswith('2025-01-01'))
        with patch.object(admin_api, 'database', return_value=self.db):
            asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
