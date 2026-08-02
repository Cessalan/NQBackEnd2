"""
Quiz rationale service: generates the per-option "Option X is correct/
incorrect because..." HTML on demand when a user clicks Learn more on a
quiz question.

Originally this rationale was produced inline by the question-generation
prompt (tools/quiztools.py) and shipped with every question. That cost
~200 output tokens per question that most users never read. This service
defers generation to the moment the user actually asks for it.

Output format matches what ChatQuizStream / StudyQuizCard already parse,
so the frontend renderer is unchanged:
    "<b>Option X is correct</b> because [reason]. ...
     <br><br><b>Option A is incorrect</b> because [reason].
     <br><b>Option B is incorrect</b> because [reason]. ..."

Uses Claude Haiku 4.5 with a SHA-1-keyed in-process cache. Same question
hit twice (e.g. across users) only costs one LLM call.
"""

import hashlib
import json
import os
import re
from typing import Optional

from anthropic import AsyncAnthropic

_CACHE: dict[str, dict] = {}
_client: Optional[AsyncAnthropic] = None

_LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
}


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def _cache_key(question: str, options: list[str], correct_index: int, language: str) -> str:
    payload = "|".join([
        language,
        str(correct_index),
        re.sub(r"\s+", " ", question).strip().lower(),
        "::".join(re.sub(r"\s+", " ", o).strip().lower() for o in options),
    ]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


def _letter(i: int) -> str:
    return chr(ord("A") + i) if 0 <= i < 26 else "?"


_PROMPT_TEMPLATE = """You are an NCLEX tutor explaining why one option of a multiple-choice question is correct and the others are not. The student already answered and is now asking for the full reasoning.

Question: {question}

Options:
{options_block}

Correct option: {correct_letter}) {correct_option_text}

Write the explanation in {language_name}. Even if the question is in another language, respond in {language_name}.

Return ONLY a JSON object — no prose, no markdown fences — with EXACTLY this shape:
{{
  "rationale_html": "<b>Option {correct_letter} is correct</b> because [factual explanation, wrapping any medical term like <strong>hyperkalemia</strong> in strong tags].<br><br><b>Option [other] is incorrect</b> because [reason].<br><b>Option [other] is incorrect</b> because [reason].<br><b>Option [other] is incorrect</b> because [reason]."
}}

Formatting rules (CRITICAL):
- Use <b>...</b> ONLY for the "Option X is correct/incorrect" headers — visual emphasis, never clickable.
- Use <strong>...</strong> ONLY for individual medical terms (drugs, conditions, signs, labs, anatomy, procedures, electrolytes, vital-sign findings). Wrap the noun phrase only — never a whole sentence and never an "Option X" header.
- Do NOT wrap generic English/French words, numbers, or option letters in <strong>.
- Aim for 2–5 <strong> terms total when clinically relevant; zero is fine if the rationale is non-clinical.
- One <br> between siblings, two <br><br> between the correct block and the incorrect block.
- Do not include the option text inside the rationale itself — refer to options by letter only.
- Keep each "because [reason]" sentence ≤ 30 words. Plain text, no markdown, no emojis, no bullet characters.
"""


def _empty_html(correct_index: int, options: list[str]) -> str:
    return f"<b>Option {_letter(correct_index)} is correct</b>."


async def generate_rationale(
    question: str,
    options: list[str],
    correct_index: int,
    language: str = "en",
) -> dict:
    """
    Returns { rationale_html: str }. On total failure, returns a minimal
    fallback so the frontend never sees an empty popover.
    """
    if not question or not question.strip():
        return {"rationale_html": _empty_html(correct_index, options)}
    if not options or correct_index < 0 or correct_index >= len(options):
        return {"rationale_html": _empty_html(correct_index, options)}

    language = (language or "en").split("-")[0].lower()
    language_name = _LANGUAGE_NAMES.get(language, "English")

    key = _cache_key(question, options, correct_index, language)
    if key in _CACHE:
        return _CACHE[key]

    _LETTER_PREFIX = re.compile(r"^[A-Z]\)\s*")
    correct_letter = _letter(correct_index)
    correct_text = _LETTER_PREFIX.sub("", options[correct_index]).strip()
    options_block = "\n".join(
        f"{_letter(i)}) {_LETTER_PREFIX.sub('', o).strip()}"
        for i, o in enumerate(options)
    )

    client = _get_client()
    prompt = _PROMPT_TEMPLATE.format(
        question=question.strip(),
        options_block=options_block,
        correct_letter=correct_letter,
        correct_option_text=correct_text,
        language_name=language_name,
    )

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text if response.content else ""
        cleaned = _strip_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return {"rationale_html": _empty_html(correct_index, options)}
            parsed = json.loads(match.group(0))

        rationale_html = (parsed.get("rationale_html") or "").strip()
        if not rationale_html:
            return {"rationale_html": _empty_html(correct_index, options)}

        result = {"rationale_html": rationale_html}
        _CACHE[key] = result
        return result
    except Exception as e:
        print(f"⚠️ Rationale LLM call failed: {e}")
        return {"rationale_html": _empty_html(correct_index, options)}
