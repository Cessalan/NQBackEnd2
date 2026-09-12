"""A short, human note after a public SEO landing-page practice set."""
import os
import re
import time


OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
DEFAULT_MODEL = 'inclusionai/ling-3.0-flash'
MAX_ITEMS = 15
MAX_NOTE = 420
_HITS = {}


def client_ip(request):
    forwarded = (request.headers.get('x-forwarded-for') or '').split(',')[0].strip()
    return forwarded[:64] or (request.client.host if request.client else 'unknown')


def allow(ip, limit=20, window=3600):
    now = time.time()
    recent = [stamp for stamp in _HITS.get(ip, []) if now - stamp < window]
    if len(recent) >= limit:
        _HITS[ip] = recent
        return False
    recent.append(now)
    _HITS[ip] = recent
    if len(_HITS) > 4000:
        stale = [key for key, stamps in _HITS.items() if not stamps or now - stamps[-1] > window]
        for key in stale:
            _HITS.pop(key, None)
    return True


def _clip(value, limit):
    return ' '.join(str(value or '').replace('\n', ' ').split())[:limit]


def sanitize(body):
    if not isinstance(body, dict):
        raise ValueError('invalid body')
    items = []
    for raw in (body.get('items') or [])[:MAX_ITEMS]:
        if not isinstance(raw, dict):
            continue
        concept = _clip(raw.get('concept'), 80)
        if not concept:
            continue
        items.append({
            'concept': concept,
            'section': _clip(raw.get('section'), 60),
            'correct': bool(raw.get('correct')),
            'why': _clip(raw.get('why'), 280),
        })
    if not items:
        raise ValueError('no items')
    correct = sum(1 for item in items if item['correct'])
    return {
        'title': _clip(body.get('title'), 80),
        'cluster': _clip(body.get('cluster'), 40),
        'compact': bool(body.get('compact')),
        'retry': bool(body.get('retry')),
        'correct': correct,
        'total': len(items),
        'items': items,
    }


def _clean(text):
    raw = re.sub(r'^```(?:\w+)?\s*|\s*```$', '', (text or '').strip())
    raw = re.sub(r'\*+([^*]+)\*+', r'\1', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    if not raw or raw.startswith('{') or re.search(r'\bas an ai\b', raw, re.I):
        return ''
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', raw) if part.strip()]
    return ' '.join(sentences[:2])[:MAX_NOTE]


def _brief(data):
    held = next((item for item in data['items'] if item['correct']), None)
    missed = next((item for item in data['items'] if not item['correct']), None)
    parts = [f"{data['correct']}/{data['total']} on {data['title']}."]
    if held:
        parts.append(f"Right: {held['concept']}. {held['why'][:80]}")
    if missed:
        parts.append(f"Review: {missed['concept']}. {missed['why'][:80]}")
    return ' '.join(parts)


SYSTEM = (
    'Write exactly 2 short sentences to a nursing student. Warm and specific. '
    'Name one thing she got right, then one thing to review. '
    'If she got nothing right, say this small sample is not a verdict, then name one review item. '
    'No markdown. No score. No extra facts.'
)


async def _complete(client, key, extra):
    body = {
        'model': os.getenv('OPENROUTER_MODEL', DEFAULT_MODEL),
        'temperature': 0.5,
        'max_tokens': 280,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': extra.pop('user')},
        ],
    }
    body.update(extra)
    response = await client.post(
        OPENROUTER_URL,
        headers={
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://nursequizai.com',
            'X-Title': 'NurseQuiz',
        },
        json=body,
    )
    response.raise_for_status()
    return (((response.json().get('choices') or [{}])[0].get('message') or {}).get('content'))


async def generate_note(data):
    key = os.getenv('OPENROUTER_API_KEY')
    if not key:
        return None
    user = _brief(data)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=20) as client:
            content = await _complete(client, key, {
                'user': user,
                'reasoning': {'max_tokens': 60, 'exclude': True},
            })
            if not content:
                content = await _complete(client, key, {'user': user})
        return _clean(content)
    except Exception:
        return None


async def write_note(body, request):
    from fastapi import HTTPException
    ip = client_ip(request)
    if not allow(ip):
        raise HTTPException(429, 'Try again in a little while.')
    try:
        data = sanitize(body)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {'note': await generate_note(data)}
