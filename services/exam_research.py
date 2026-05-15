"""
Exam Research Service
=====================

When a student names a specific school, exam board, or country-specific
standard (e.g. "Korle Bu nursing finals", "UK NMC test of competence",
"Ghana Health Service licensure exam"), this service:

1. Web-searches for that school's exam format AND the recurring question
   topics in the requested subject area.
2. Returns a structured research brief + a list of citations (url, title,
   snippet) the orchestrator forwards to the frontend.
3. Caches the result by (exam_board, school, topic) in Firestore for 24
   hours so the same school+topic combination doesn't get re-searched on
   every request.

The downstream quiz generator receives the research brief as
`additional_context` and grounds its question generation in that material.
The frontend renders the citations as a sources panel above the quiz, so
the student sees exactly what the quiz is built from.

Model & cost
------------
Claude Sonnet 4.6 with `web_search_20260209`. Adaptive thinking off (we want
fast, focused gathering, not deep reasoning). Capped at 3 search calls per
request via `max_uses` so cost stays predictable: roughly $0.05-0.10 per
uncached request, ~free on cache hit.
"""

import asyncio
import hashlib
import json
import os
import re
from typing import Optional

from anthropic import AsyncAnthropic

_client: Optional[AsyncAnthropic] = None
_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


RESEARCH_SYSTEM_PROMPT = """You are an exam research assistant for a nursing
study app. The student wants quiz practice that matches a SPECIFIC exam -
a school's finals, a country's licensure exam, a board exam like NMC or
NCLEX. Your job is to web-search for that exam's format and the topics it
covers, then emit ONE structured brief by calling the emit_research_brief
tool.

What to search for:
- The exam's question style (MCQ, SATA, case-based scenario, short-answer)
- Difficulty bar (recall vs application vs clinical judgment)
- Topic coverage in the subject area the student named
- Any recurring question themes mentioned in syllabi, past-paper PDFs,
  exam-prep blogs, or university course pages

What to AVOID:
- Generic nursing-quiz websites (Quizlet, Brainscape, generic NCLEX prep)
  unless they are the only source. Prefer official sources: university
  websites, ministry/council pages, published study guides, professor
  slides.
- Inventing sources. If a real source isn't found, say so honestly in the
  brief - do not make up URLs.

Output rules:
- Call emit_research_brief exactly once at the end.
- Keep the brief tight: 4-6 short paragraphs at most. The downstream quiz
  generator will use it as grounding context, not literal text.
- Each citation must be a URL you actually retrieved during web_search.
  Do NOT include URLs from memory.
"""


EMIT_RESEARCH_BRIEF_TOOL = {
    "name": "emit_research_brief",
    "description": (
        "Emit the final research brief for the requested exam + topic. "
        "Call this exactly once at the end, after web_search results have "
        "been gathered."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "exam_summary": {
                "type": "string",
                "description": (
                    "One paragraph (60-120 words) describing the exam's "
                    "format and difficulty: question types used, typical "
                    "structure, and what the exam tests for."
                ),
            },
            "topic_coverage": {
                "type": "string",
                "description": (
                    "Two to three paragraphs covering the most relevant "
                    "subtopics within the requested subject area, drawn "
                    "from the search results. Include specific clinical "
                    "scenarios, drug classes, or care priorities that "
                    "this exam is known to test on this topic."
                ),
            },
            "key_question_themes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "5-10 short strings describing recurring question "
                    "themes (e.g. 'priority action for hyperkalemia', "
                    "'antibiotic timing post-surgery'). These become "
                    "seeds for quiz question generation."
                ),
                "minItems": 3,
                "maxItems": 12,
            },
            "found_real_papers": {
                "type": "boolean",
                "description": (
                    "True only if a search result actually contained "
                    "verbatim past-paper questions. False if we only "
                    "found syllabi, blog posts, or general format info."
                ),
            },
            "honesty_note": {
                "type": ["string", "null"],
                "description": (
                    "If the search did NOT find specific past papers or "
                    "official material for this exam, a 1-sentence "
                    "honest message to show the user "
                    "(e.g. 'I couldn't find official Korle Bu past "
                    "papers online, but I gathered the exam style and "
                    "typical topics from the school's nursing program "
                    "page'). Null if real papers were found."
                ),
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "snippet": {
                            "type": "string",
                            "description": "Short quote or summary, <40 words.",
                        },
                    },
                    "required": ["url", "title", "snippet"],
                },
                "minItems": 1,
                "maxItems": 8,
                "description": (
                    "Sources cited by the brief. Use actual URLs from "
                    "the web_search results - not hallucinated. Order "
                    "by relevance to the request."
                ),
            },
        },
        "required": [
            "exam_summary",
            "topic_coverage",
            "key_question_themes",
            "found_real_papers",
            "citations",
        ],
    },
}


WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    # Cap searches to keep cost predictable. Two queries (one for exam
    # format, one for topic content) is usually enough; three handles
    # the harder cases where the first query was off.
    "max_uses": 3,
}


# ──────────────────────────────────────────────────────────────────────
# CACHE LAYER
# ──────────────────────────────────────────────────────────────────────
# Firestore-backed, 24h TTL. Keyed on a stable hash of
# (exam_board, school, topic, language). Two users in the same window
# asking "Korle Bu cardio" share one search.
# ──────────────────────────────────────────────────────────────────────


def _cache_key(
    exam_board: Optional[str],
    school: Optional[str],
    topic: str,
    language: str,
) -> str:
    payload = "|".join(
        [
            (exam_board or "").strip().lower(),
            (school or "").strip().lower(),
            re.sub(r"\s+", " ", topic).strip().lower(),
            (language or "en").strip().lower()[:5],
        ]
    ).encode("utf-8")
    return "exam_research_" + hashlib.sha1(payload).hexdigest()[:24]


async def _cache_get(key: str) -> Optional[dict]:
    """
    Returns the cached brief if fresh, else None. Wrapped in to_thread
    because firebase_admin's Firestore client is synchronous.
    """
    try:
        from firebase_admin import firestore
    except Exception:
        return None

    def _read():
        try:
            db = firestore.client()
            doc = db.collection("exam_research_cache").document(key).get()
            if not doc.exists:
                return None
            data = doc.to_dict() or {}
            saved_at = data.get("saved_at_epoch")
            if not saved_at:
                return None
            import time
            if (time.time() - saved_at) > _CACHE_TTL_SECONDS:
                return None
            return data.get("brief")
        except Exception as e:
            print(f"exam_research cache read failed: {e}")
            return None

    return await asyncio.to_thread(_read)


async def _cache_set(key: str, brief: dict) -> None:
    try:
        from firebase_admin import firestore
    except Exception:
        return

    def _write():
        try:
            import time
            db = firestore.client()
            db.collection("exam_research_cache").document(key).set(
                {
                    "brief": brief,
                    "saved_at_epoch": time.time(),
                }
            )
        except Exception as e:
            print(f"exam_research cache write failed: {e}")

    await asyncio.to_thread(_write)


# ──────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────────


async def gather_exam_research(
    topic: str,
    exam_board: Optional[str] = None,
    school: Optional[str] = None,
    research_query: Optional[str] = None,
    language: str = "english",
) -> Optional[dict]:
    """
    Returns a research brief dict matching the emit_research_brief tool
    schema, or None on total failure. Caller is responsible for streaming
    the citations to the user and passing the brief into the quiz
    generator as additional_context.

    Shape on success:
        {
            "exam_summary": str,
            "topic_coverage": str,
            "key_question_themes": [str, ...],
            "found_real_papers": bool,
            "honesty_note": str | None,
            "citations": [{"url": str, "title": str, "snippet": str}, ...],
            "cached": bool,
        }
    """
    if not topic or not topic.strip():
        return None

    key = _cache_key(exam_board, school, topic, language)
    cached = await _cache_get(key)
    if cached:
        print(f"exam_research CACHE HIT: {key}")
        cached = dict(cached)
        cached["cached"] = True
        return cached

    # Build the user prompt the research model sees
    target_parts = []
    if school:
        target_parts.append(f"school/institution: {school}")
    if exam_board:
        target_parts.append(f"exam board/standard: {exam_board}")
    target = "; ".join(target_parts) if target_parts else "general nursing exam"

    query_hint = (
        f"Search guidance: {research_query}\n" if research_query else ""
    )

    user_prompt = (
        f"Topic: {topic}\n"
        f"Target exam: {target}\n"
        f"Response language: {language}\n"
        f"{query_hint}"
        f"\n"
        f"Search the web for this exam's format and topic coverage, then "
        f"emit one structured research brief via emit_research_brief."
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": RESEARCH_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[WEB_SEARCH_TOOL, EMIT_RESEARCH_BRIEF_TOOL],
            messages=[{"role": "user", "content": user_prompt}],
        )

        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "emit_research_brief"
            ):
                brief = dict(block.input or {})
                # Defensive: cap citations to 8 and strip any without URLs
                brief["citations"] = [
                    c for c in (brief.get("citations") or [])
                    if isinstance(c, dict) and (c.get("url") or "").startswith(("http://", "https://"))
                ][:8]
                if not brief.get("citations"):
                    # No real citations means web_search likely failed.
                    print("exam_research: emitted brief without valid citations - discarding")
                    return None
                brief["cached"] = False
                # Fire-and-forget cache write; don't block the caller
                asyncio.create_task(_cache_set(key, brief))
                return brief

        print(
            f"exam_research: Claude responded without calling "
            f"emit_research_brief. stop_reason={getattr(response, 'stop_reason', '?')}"
        )
        return None

    except Exception as e:
        print(f"exam_research ERROR: {type(e).__name__}: {e}")
        return None


def format_brief_as_context(brief: dict) -> str:
    """
    Flatten the structured brief into a single string the quiz generator
    can paste into its content-context block. The themes are bullet-pointed
    so the downstream concept-extraction step picks them up.
    """
    if not brief:
        return ""

    lines = []
    summary = (brief.get("exam_summary") or "").strip()
    coverage = (brief.get("topic_coverage") or "").strip()
    themes = brief.get("key_question_themes") or []

    if summary:
        lines.append(f"EXAM FORMAT:\n{summary}")
    if coverage:
        lines.append(f"TOPIC COVERAGE:\n{coverage}")
    if themes:
        theme_lines = "\n".join(f"- {t}" for t in themes)
        lines.append(f"RECURRING QUESTION THEMES:\n{theme_lines}")
    return "\n\n".join(lines)
