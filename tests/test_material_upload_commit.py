"""Exercise upload commit ordering without booting the server or cloud clients."""
import ast
import asyncio
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.document_understanding import MaterialError


class MaterialUploadCommitTests(unittest.TestCase):
    def build_upload(self, insights, store=None, embeddings=None):
        tree = ast.parse(Path(__file__).resolve().parents[1].joinpath('main.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == 'embed_document_task')
        faiss = MagicMock()
        session = SimpleNamespace(vectorstore=store)
        namespace = {
            'asyncio': asyncio, 'os': SimpleNamespace(path=SimpleNamespace(exists=lambda _: True)),
            'get_loader_for_file': lambda *a, **kw: SimpleNamespace(load=lambda: [SimpleNamespace(page_content='Synthetic teaching material.')]),
            'extract_file_insights_from_text': insights,
            'Document': lambda **fields: SimpleNamespace(**fields),
            'OpenAIEmbeddings': lambda: SimpleNamespace(aembed_documents=embeddings or AsyncMock(return_value=[[0.1]])),
            'ACTIVE_SESSIONS': {'chat': SimpleNamespace(session=session)}, 'FAISS': faiss,
            '_get_vectorstore_write_lock': lambda _: asyncio.Lock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'upload', 'exec'), namespace)
        return namespace['embed_document_task'], faiss, session

    def run_upload(self, upload, updates):
        with patch('core.material_model.service_tier_for_chat', return_value='default'), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return asyncio.run(upload('synthetic.pdf', 'synthetic.pdf', 'chat', 'file', updates))

    def test_rejected_analysis_never_changes_existing_index_or_reports_upload_ready(self):
        for store in (None, MagicMock()):
            async def fail(*args, **kwargs):
                raise MaterialError('Unsupported source evidence')
            upload, faiss, session = self.build_upload(fail, store)
            updates = []
            with self.assertRaises(MaterialError):
                self.run_upload(upload, updates)
            self.assertIs(session.vectorstore, store)
            faiss.from_embeddings.assert_not_called()
            if store is not None:
                store.add_embeddings.assert_not_called()
            self.assertFalse(any(update['type'] == 'embedding_complete' for update in updates))

    def test_success_commits_and_reports_ready_only_after_analysis_has_passed(self):
        completed = []
        async def insights(*args, **kwargs):
            await asyncio.sleep(0)
            completed.append(True)
            return {'topics': ['Synthetic topic']}
        upload, faiss, session = self.build_upload(insights)
        def commit(*args, **kwargs):
            self.assertTrue(completed)
            return 'new-index'
        faiss.from_embeddings.side_effect = commit
        updates = []
        result = self.run_upload(upload, updates)
        self.assertEqual(session.vectorstore, 'new-index')
        self.assertEqual(result['insights'], {'topics': ['Synthetic topic']})
        self.assertEqual(len(result['documents']), 1)
        self.assertTrue(any(update['type'] == 'embedding_complete' for update in updates))

    def test_embedding_failure_cancels_and_drains_pending_analysis(self):
        stopped = []
        async def insights(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopped.append(True)
                raise
        async def fail(*args):
            await asyncio.sleep(0)
            raise RuntimeError('Synthetic embedding failure')
        upload, faiss, _ = self.build_upload(insights, embeddings=fail)
        with self.assertRaises(RuntimeError):
            self.run_upload(upload, [])
        self.assertEqual(stopped, [True])
        faiss.from_embeddings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
