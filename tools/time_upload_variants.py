"""Try faster settings for the two slow upload steps on a real uploaded PDF.

Reading: describe visual pages in parallel, skip pages with plenty of native
text and no images, and try low reasoning for the image descriptions.
Analysis: smaller sections run in parallel, low vs medium reasoning, default
vs fast tier. Reports time and how many learning goals come out, so speed is
never bought by silently analysing less. Nothing is saved (no chat id).
Usage: venv/Scripts/python tools/time_upload_variants.py CHAT_ID
"""
import asyncio
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import firebase_admin
from firebase_admin import credentials, firestore, storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
firebase_admin.initialize_app(credentials.Certificate(os.path.join(ROOT, 'service-account-key.json')),
                              {'storageBucket': 'docai-efb03.firebasestorage.app'})

from core import material_loader
from core.material_model import MODEL_POLICY
from services import document_understanding as du


def download(chat_id):
    name = [u.to_dict() for u in firestore.client().collection('chats').document(chat_id).collection('uploads').stream()][0]['name']
    path = os.path.join(tempfile.mkdtemp(), name)
    storage.bucket().blob(f'chats/{chat_id}/uploads/{name}').download_to_filename(path)
    return name, path


def read_pdf(path, *, tier, effort, skip_text_pages, workers):
    import fitz
    MODEL_POLICY['analysisReasoning'] = effort
    jobs, texts = [], []
    with fitz.open(path) as document:
        for index, page in enumerate(document):
            text = page.get_text(sort=True)
            images, drawings = page.get_images(), page.get_drawings()
            words = len(text.split())
            visual = images or drawings or words < 10
            if skip_text_pages and not images and words >= 80:
                visual = False
            texts.append(text)
            if visual:
                png = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes('png')
                jobs.append((index, png))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: (job[0], material_loader.describe_visual(
            job[1], 'image/png', f'PDF page {job[0] + 1}', service_tier=tier)), jobs))
    for index, description in results:
        texts[index] += '\n' + description
    MODEL_POLICY['analysisReasoning'] = 'medium'
    return '\n\n'.join(f'[Page {i + 1}]\n{t}' for i, t in enumerate(texts)), len(jobs)


async def analyse(text, name, *, tier, effort, section_size, concurrency):
    from functools import partial
    du.SECTION_SIZE, du.OVERLAP = section_size, min(1500, section_size // 6)
    original = asyncio.Semaphore
    asyncio.Semaphore = lambda _n: original(concurrency)
    try:
        call = partial(du.model_json, service_tier=tier, reasoning_effort=effort)
        start = time.perf_counter()
        result = await du.analyse_document(text, name, chat_id=None, call=call)
        return time.perf_counter() - start, len(list(du.sections(text))), len(result['goals']), len(result['signals'])
    except Exception as error:  # noqa: BLE001
        return time.perf_counter() - start, len(list(du.sections(text))), f'FAILED {type(error).__name__}: {str(error)[:100]}', 0
    finally:
        asyncio.Semaphore = original
        du.SECTION_SIZE, du.OVERLAP = 12000, 1500


async def main(chat_id):
    name, path = download(chat_id)
    print(f'file: {name}')
    print('\nREADING')
    text = None
    for label, kw in (
        ('current (one at a time, every page)', dict(tier='default', effort='medium', skip_text_pages=False, workers=1)),
        ('parallel, every page', dict(tier='default', effort='medium', skip_text_pages=False, workers=8)),
        ('parallel + skip text pages', dict(tier='default', effort='medium', skip_text_pages=True, workers=8)),
        ('parallel + skip + low reasoning', dict(tier='default', effort='low', skip_text_pages=True, workers=8)),
        ('parallel + skip + low + fast tier', dict(tier='fast', effort='low', skip_text_pages=True, workers=8)),
    ):
        start = time.perf_counter()
        text, sent = await asyncio.to_thread(read_pdf, path, **kw)
        print(f'  {time.perf_counter() - start:5.1f}s  {label:38s} pages sent to Luna={sent}')

    print('\nANALYSIS (same text for every row)')
    for label, kw in (
        ('current: 12k sections, medium, default', dict(tier='default', effort='medium', section_size=12000, concurrency=3)),
        ('4k sections x8, medium, default', dict(tier='default', effort='medium', section_size=4000, concurrency=8)),
        ('4k sections x8, low, default', dict(tier='default', effort='low', section_size=4000, concurrency=8)),
        ('12k sections, low, default', dict(tier='default', effort='low', section_size=12000, concurrency=3)),
        ('4k sections x8, low, fast', dict(tier='fast', effort='low', section_size=4000, concurrency=8)),
    ):
        seconds, n_sections, goals, signals = await analyse(text, name, **kw)
        print(f'  {seconds:5.1f}s  {label:40s} sections={n_sections} goals={goals} signals={signals}')


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1]))
