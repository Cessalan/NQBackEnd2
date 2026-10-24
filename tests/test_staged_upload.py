import asyncio
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.staged_upload import MAX_STAGED_FILE_BYTES, read_staged_files, staged_names


class FakeBucket:
    def __init__(self, blobs):
        self.blobs = blobs
        self.asked = []

    def get_blob(self, path):
        self.asked.append(path)
        data = self.blobs.get(path)
        if data is None:
            return None
        size, payload = data
        return types.SimpleNamespace(size=size, content_type='application/pdf',
                                     download_as_bytes=lambda: payload)


class StagedUploadTests(unittest.TestCase):
    def test_names_cannot_leave_the_chat_uploads_folder(self):
        self.assertEqual(staged_names(json.dumps(['Renal deck.pptx', 'Renal deck.pptx', 'notes.pdf'])),
                         ['Renal deck.pptx', 'notes.pdf'])
        for bad in [['../other/x.pdf'], ['a/b.pdf'], ['a\\b.pdf'], ['..'], [''], [3], [],
                    ['x' * 256], ['ok.pdf'] * 0 + [f'{i}.pdf' for i in range(31)]]:
            self.assertIsNone(staged_names(json.dumps(bad)), bad)
        self.assertIsNone(staged_names('not json'))
        self.assertIsNone(staged_names(None))

    def test_reads_from_the_chats_own_folder_and_flags_staged(self):
        bucket = FakeBucket({'chats/c1/uploads/a.pdf': (3, b'abc')})
        out = asyncio.run(read_staged_files(bucket, 'c1', ['a.pdf', 'missing.pdf']))
        self.assertEqual(bucket.asked, ['chats/c1/uploads/a.pdf', 'chats/c1/uploads/missing.pdf'])
        self.assertEqual(out[0], {'filename': 'a.pdf', 'content_type': 'application/pdf',
                                  'bytes': b'abc', 'size': 3, 'staged': True})
        self.assertIn('error', out[1])
        self.assertNotIn('bytes', out[1])

    def test_an_oversized_file_is_refused_with_a_code_before_download(self):
        def boom():
            raise AssertionError('must not download')
        bucket = FakeBucket({})
        bucket.get_blob = lambda path: types.SimpleNamespace(size=MAX_STAGED_FILE_BYTES + 1, content_type=None,
                                                             download_as_bytes=boom)
        out = asyncio.run(read_staged_files(bucket, 'c1', ['huge.pptx']))
        self.assertEqual(out[0]['code'], 'file_too_large')

    def test_a_storage_error_becomes_a_file_error(self):
        bucket = FakeBucket({})
        def fail(path):
            raise RuntimeError('storage down')
        bucket.get_blob = fail
        out = asyncio.run(read_staged_files(bucket, 'c1', ['a.pdf']))
        self.assertEqual(out[0], {'filename': 'a.pdf', 'error': 'storage down'})


if __name__ == '__main__':
    unittest.main()
