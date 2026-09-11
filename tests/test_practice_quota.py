"""Exercise the production transaction functions with an in-memory Firestore adapter.

Uses only the standard library so quota tests don't require cloud credentials.
"""
import ast
import copy
import pathlib
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


class HttpError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class Ref:
    def __init__(self, db, path):
        self.db, self.path = db, path
    def collection(self, name):
        return Ref(self.db, self.path + '/' + name)
    def document(self, name):
        return Ref(self.db, self.path + '/' + name)
    def get(self, transaction=None):
        data = copy.deepcopy(self.db.rows.get(self.path))
        return types.SimpleNamespace(exists=data is not None, to_dict=lambda: data)


class DB:
    def __init__(self):
        self.rows = {}
        self.lock = threading.Lock()
    def collection(self, name):
        return Ref(self, name)
    def transaction(self):
        return self
    def set(self, ref, data, merge=False):
        self.rows[ref.path] = {**(self.rows.get(ref.path, {}) if merge else {}), **copy.deepcopy(data)}


class PracticeQuotaTests(unittest.TestCase):
    def setUp(self):
        self.db = DB()
        def transactional(fn):
            def wrapped(tx):
                with self.db.lock:
                    return fn(tx)
            return wrapped
        firestore = types.SimpleNamespace(client=lambda: self.db, transactional=transactional)
        auth = types.SimpleNamespace(verify_id_token=lambda _: {'uid': 'u'})
        self.modules = patch.dict(sys.modules, {'firebase_admin': types.SimpleNamespace(firestore=firestore, auth=auth)})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        source = pathlib.Path(__file__).parents[1] / 'services' / 'practice_api.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('reserve', 'finish', 'owner')]
        self.ns = {'time': time, 'HTTPException': HttpError,
                   'usage_guard': types.SimpleNamespace(FREE_LIMIT=70, WINDOW_MS=604800000, QUOTA_MESSAGE='limit')}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), self.ns)
        self.db.rows['users/u'] = {'usage': {'tier': 'free', 'count': 69, 'windowStart': int(time.time() * 1000)}}
    def request(self, ident='request-1'):
        return types.SimpleNamespace(request_id=ident, chat_id='chat', count=5, existing_questions=[])
    def test_last_unit_and_concurrent_requests(self):
        def attempt(ident):
            try:
                return self.ns['reserve']('u', self.request(ident))['count']
            except HttpError as exc:
                return exc.status_code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, ['request-1', 'request-2']))
        self.assertCountEqual(results, [1, 429])
        self.assertEqual(self.db.rows['users/u']['usage']['count'], 70)
    def test_replay_does_not_charge_again(self):
        req = self.request()
        self.ns['reserve']('u', req)
        self.ns['finish']('u', req.request_id, [{'question': 'Q'}])
        replay = self.ns['reserve']('u', req)
        self.assertEqual(replay['questions'], [{'question': 'Q'}])
        self.assertEqual(self.db.rows['users/u']['usage']['count'], 70)
    def test_failed_generation_refunds_unused_units(self):
        self.ns['reserve']('u', self.request())
        self.ns['finish']('u', 'request-1', [])
        self.ns['finish']('u', 'request-1', [])
        self.assertEqual(self.db.rows['users/u']['usage']['count'], 69)
    def test_pro_does_not_consume_free_allowance(self):
        self.db.rows['users/u']['usage']['tier'] = 'pro'
        self.assertEqual(self.ns['reserve']('u', self.request())['count'], 5)
        self.assertEqual(self.db.rows['users/u']['usage']['count'], 69)
    def test_expired_window_resets(self):
        self.db.rows['users/u']['usage']['windowStart'] = 1
        self.assertEqual(self.ns['reserve']('u', self.request())['count'], 5)
        self.assertEqual(self.db.rows['users/u']['usage']['count'], 5)
    def test_other_users_chat_is_forbidden(self):
        self.db.rows['chats/chat'] = {'userId': 'someone-else'}
        with self.assertRaises(HttpError) as raised:
            self.ns['owner'](types.SimpleNamespace(headers={'authorization': 'Bearer token'}), 'chat')
        self.assertEqual(raised.exception.status_code, 403)


if __name__ == '__main__':
    unittest.main()
