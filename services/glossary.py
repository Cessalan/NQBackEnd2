"""
Glossary service: returns NCLEX-tailored definitions for medical terms
clicked inside quiz rationales.

Uses Claude Haiku 4.5 with a plain in-process dict cache (30 DAU scale).
"""

import json
import os
import re
from typing import Optional

from anthropic import AsyncAnthropic

_CACHE: dict[str, dict] = {}
_client: Optional[AsyncAnthropic] = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def _normalize(term: str) -> str:
    # Collapse whitespace, lower-case, strip trailing punctuation/possessives.
    t = re.sub(r"\s+", " ", term).strip().lower()
    t = t.rstrip(".,;:!?’'\"")
    return t


_PROMPT_TEMPLATE = """You are an NCLEX tutor producing a tiny popover definition for a nursing student who tapped a bolded medical term inside a quiz rationale.

Term: {term}

Return ONLY a JSON object — no prose, no markdown fences — with EXACTLY these keys:
{{
  "term": "the canonical capitalized form of the term",
  "definition": "ONE sentence (<= 25 words) explaining what it is in plain clinical language",
  "nclex_relevance": "1-2 sentences (<= 40 words) on why the NCLEX tests this — what concept/judgment it points to",
  "nursing_consideration": "ONE concrete bedside nursing action, assessment, or red flag (<= 25 words)"
}}

Rules:
- If the term is NOT a real medical/clinical/pharmacology/anatomy term (e.g. "Option A", a generic English word, gibberish), return:
  {{"term": "{term}", "definition": "", "nclex_relevance": "", "nursing_consideration": "", "not_a_term": true}}
- Tailor every field to NCLEX-style nursing practice — not a textbook dump.
- Use plain text. No HTML, no markdown, no bullets, no emojis.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove leading ```json or ``` and trailing ```
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


async def get_term_definition(term: str) -> dict:
    """
    Returns a dict with keys: term, definition, nclex_relevance,
    nursing_consideration, and optionally not_a_term=True.

    Caches by normalized term for the lifetime of the process.
    """
    if not term or not term.strip():
        return {
            "term": term or "",
            "definition": "",
            "nclex_relevance": "",
            "nursing_consideration": "",
            "not_a_term": True,
        }

    key = _normalize(term)
    if key in _CACHE:
        return _CACHE[key]

    client = _get_client()
    prompt = _PROMPT_TEMPLATE.format(term=term.strip())

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text if response.content else ""
    cleaned = _strip_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort: try to pull the first {...} block out of the response.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {
                "term": term,
                "definition": "",
                "nclex_relevance": "",
                "nursing_consideration": "",
                "not_a_term": True,
            }
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "term": term,
                "definition": "",
                "nclex_relevance": "",
                "nursing_consideration": "",
                "not_a_term": True,
            }

    result = {
        "term": parsed.get("term") or term,
        "definition": parsed.get("definition", "") or "",
        "nclex_relevance": parsed.get("nclex_relevance", "") or "",
        "nursing_consideration": parsed.get("nursing_consideration", "") or "",
    }
    if parsed.get("not_a_term"):
        result["not_a_term"] = True

    _CACHE[key] = result
    return result
