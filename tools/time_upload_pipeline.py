"""Time each step of a document upload on a real uploaded file.

Steps, as embed_document_task runs them: read the file (TeachingMaterialLoader,
which sends visual pages to Luna one at a time), embed the chunks, and the full
Luna analysis the upload waits for before reporting ready. Nothing is saved:
the analysis runs without a chat id, so no cache is read or written.
Usage: venv/Scripts/python tools/time_upload_pipeline.py CHAT_ID [default|fast]
"""
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import firebase_admin
from firebase_admin import credentials, firestore, storage

KEY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'service-account-key.json')
firebase_admin.initialize_app(credentials.Certificate(KEY), {'storageBucket': 'docai-efb03.firebasestorage.app'})


def download(chat_id):
    rows = [u.to_dict() for u in firestore.client().collection('chats').document(chat_id).collection('uploads').stream()]
    name = rows[0]['name']
    blob = storage.bucket().blob(f'chats/{chat_id}/uploads/{name}')
    path = os.path.join(tempfile.mkdtemp(), name)
    blob.download_to_filename(path)
    return name, path


async def main(chat_id, tier):
    from core import material_loader
    from core.material_loader import TeachingMaterialLoader
    from services.document_understanding import analyse_document, model_json, sections
    from functools import partial
    from core.material_model import MODEL_POLICY
    from langchain_openai import OpenAIEmbeddings

    name, path = download(chat_id)
    print(f'file: {name} ({os.path.getsize(path) // 1024} KB), tier={tier}')

    visual_times = []
    original = material_loader.describe_visual

    def timed_visual(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            visual_times.append(time.perf_counter() - start)

    material_loader.describe_visual = timed_visual
    start = time.perf_counter()
    pages = await asyncio.to_thread(TeachingMaterialLoader(path, service_tier=tier).load)
    read = time.perf_counter() - start
    material_loader.describe_visual = original
    text = '\n\n'.join(p.page_content for p in pages)
    print(f'1. read file:   {read:6.1f}s  pages={len(pages)}  pages sent to Luna as images={len(visual_times)}'
          + (f'  ({min(visual_times):.1f}-{max(visual_times):.1f}s each)' if visual_times else ''))

    from services.document_understanding import quick_overview
    chunks = [text[s:s + 1000] for s in range(0, len(text), 800)]

    async def embed():
        started = time.perf_counter()
        await OpenAIEmbeddings().aembed_documents(chunks)
        return time.perf_counter() - started

    async def preview():
        started = time.perf_counter()
        result = await quick_overview(text, name)
        return time.perf_counter() - started, result

    start = time.perf_counter()
    embed_seconds, (preview_seconds, shown) = await asyncio.gather(embed(), preview())
    waited = time.perf_counter() - start
    print(f'2. embeddings:  {embed_seconds:6.1f}s  chunks={len(chunks)}  words={len(text.split())}')
    print(f'   quick topics:{preview_seconds:6.1f}s  {shown["topics"]}')
    print(f'   UPLOAD READY after {read + waited:.1f}s (read + the slower of embeddings / quick topics)')

    call = partial(model_json, service_tier=tier, reasoning_effort=MODEL_POLICY['analysisReasoning'])
    start = time.perf_counter()
    analysis = await analyse_document(text, name, chat_id=None, call=call)
    print(f'3. background analysis (no longer waited for): {time.perf_counter() - start:4.1f}s  sections={len(list(sections(text)))}  goals={len(analysis["goals"])}')


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'default'))
