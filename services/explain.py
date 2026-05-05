"""
Explain service: returns a short, NCLEX-tailored explanation for an
arbitrary phrase/sentence the user selected in the app (chat, quiz,
rationale, flashcard).

Sibling of services/glossary.py. The glossary endpoint targets single
medical terms; this one targets free-form selections — sentences, clauses,
or multi-word phrases.

Uses Claude Haiku 4.5 with a plain in-process dict cache (30 DAU scale).
"""

import hashlib
import json
import os
import re
from typing import Optional

from anthropic import AsyncAnthropic

_CACHE: dict[str, dict] = {}
_client: Optional[AsyncAnthropic] = None

_CONTEXT_HINTS = {
    "chat": "the AI tutor's chat reply",
    "rationale": "the explanation under a quiz answer",
    "quiz": "the question or answer choices in a quiz",
    "flashcard": "a flashcard front or back",
}

# ISO 639-1 → human-readable language name used in the prompt.
_LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
}


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _cache_key(text: str, context: str, language: str) -> str:
    payload = f"{language}|{context}|{_normalize(text)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


_PROMPT_TEMPLATE = """You are an NCLEX tutor. A nursing student highlighted the snippet below inside {context_hint} and asked you to explain it. They want clarity, not a textbook dump.

Snippet:
\"\"\"
{text}
\"\"\"

Write your entire response in {language_name}. Even if the snippet itself is in another language, you must respond in {language_name}.

Return ONLY a JSON object — no prose, no markdown fences — with EXACTLY these keys:
{{
  "explanation": "2-4 short sentences (<= 70 words total) explaining the snippet in plain clinical language, focused on what a nursing student needs to take away",
  "key_points": ["up to 3 short bullets (<= 12 words each), each one a discrete fact or NCLEX-relevant nuance; empty array if none add value"]
}}

Rules:
- Tailor every word to NCLEX-style nursing practice — assessment, prioritization, safety, pharmacology mechanism, etc.
- Plain text only. No HTML, no markdown, no emojis, no bullet characters in the explanation field.
- If the snippet is gibberish, empty, or has no clinical meaning at all, return:
  {{"explanation": "", "key_points": [], "not_explainable": true}}
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


def _empty_result(text: str) -> dict:
    return {
        "text": text,
        "explanation": "",
        "key_points": [],
        "not_explainable": True,
    }


async def explain_selection(
    text: str,
    context: str = "chat",
    language: str = "en",
) -> dict:
    """
    Returns a dict: { text, explanation, key_points: list[str],
    optionally not_explainable=True }.
    """
    if not text or not text.strip():
        return _empty_result(text or "")

    context = context if context in _CONTEXT_HINTS else "chat"
    language = (language or "en").split("-")[0].lower()
    language_name = _LANGUAGE_NAMES.get(language, "English")

    key = _cache_key(text, context, language)
    if key in _CACHE:
        return _CACHE[key]

    client = _get_client()
    prompt = _PROMPT_TEMPLATE.format(
        text=text.strip(),
        context_hint=_CONTEXT_HINTS[context],
        language_name=language_name,
    )

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text if response.content else ""
    cleaned = _strip_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return _empty_result(text)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return _empty_result(text)

    key_points = parsed.get("key_points") or []
    if not isinstance(key_points, list):
        key_points = []
    key_points = [str(p).strip() for p in key_points if str(p).strip()][:3]

    result = {
        "text": text,
        "explanation": (parsed.get("explanation") or "").strip(),
        "key_points": key_points,
    }
    if parsed.get("not_explainable") or not result["explanation"]:
        result["not_explainable"] = True

    _CACHE[key] = result
    return result
