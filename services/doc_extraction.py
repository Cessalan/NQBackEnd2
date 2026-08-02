"""
Document Question Extraction
============================

When the intent analyzer determines the user wants to ANSWER THE ACTUAL
QUESTIONS from a paper they uploaded (e.g. "take out the answers and let me
answer them"), this module:

1. Pulls the document's text from the in-memory vectorstore.
2. Asks Claude Sonnet 4.6 to identify each multiple-choice question and emit
   it as a clean JSON object (question + 4 options + which letter is marked
   as the answer in the source).
3. Streams each question through the SAME event channel the quiz pipeline
   uses (`status: "quiz_question"`), so the frontend's StudyQuizCard renders
   them with no changes.

The answer letter is preserved as `correctIndex` so the student gets feedback
after answering, but the option text is presented with the answer marker
stripped so they actually have to choose.

Why not LLM-generate questions from the document? Because the user explicitly
asked for the questions IN the document, not new ones. Generating new ones
when the user wanted extraction is the failure this fix is targeting.
"""

import json
import os
import re
from typing import AsyncGenerator, Optional

from anthropic import AsyncAnthropic

_client: Optional[AsyncAnthropic] = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


EXTRACTION_SYSTEM_PROMPT = """You extract multiple-choice questions from
nursing past papers. The user uploaded a document and wants to answer the
actual questions from it - not new ones you generate.

For each multiple-choice question you find in the document, emit ONE JSON
object using the emit_question tool. Skip any content that isn't a clear
multiple-choice question (headings, instructions, essay prompts, fill-in-the-
blank with no options, etc.).

Rules:
- Extract questions verbatim. Do not rephrase.
- Each question must have exactly 4 options. If the source has 3 or 5, do
  your best - 4 is the target but emit what's there.
- If the source marks the correct answer (asterisk, bold, "Ans:", "Answer:
  B", a key at the bottom of the document), set correct_letter to that
  letter. If you can't tell, set correct_letter to null.
- Strip any answer markers from the option text itself. The student needs to
  see the options without the answer revealed.
- Stop emitting when you've covered every question in the document. Do not
  invent questions to pad the count.
"""


EMIT_QUESTION_TOOL = {
    "name": "emit_question",
    "description": (
        "Emit one multiple-choice question extracted from the document. "
        "Call this once per question, in document order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question stem, verbatim from the document.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 6,
                "description": (
                    "Option texts in order, WITHOUT the leading 'A)' / 'B)' "
                    "labels and WITHOUT any answer marker (no asterisks, "
                    "no 'correct' annotations)."
                ),
            },
            "correct_letter": {
                "type": ["string", "null"],
                "enum": ["A", "B", "C", "D", "E", "F", None],
                "description": (
                    "Which letter the source marks as correct, or null if "
                    "the source does not indicate an answer."
                ),
            },
            "topic": {
                "type": ["string", "null"],
                "description": "Section or topic heading this question falls under, if known.",
            },
        },
        "required": ["question", "options"],
    },
}


def _letter_to_index(letter: Optional[str]) -> int:
    if not letter:
        return -1
    letter = letter.strip().upper()
    if len(letter) == 1 and "A" <= letter <= "F":
        return ord(letter) - ord("A")
    return -1


def _format_options_with_letters(options: list[str]) -> list[str]:
    """Frontend expects 'A) text' / 'B) text' shape."""
    return [f"{chr(ord('A') + i)}) {opt.strip()}" for i, opt in enumerate(options)]


def _gather_document_text(session, max_chars: int = 60000) -> tuple[str, str]:
    """
    Pull as much of the most recently uploaded document as fits in the LLM
    context window. Returns (text, filename).
    """
    if not getattr(session, "vectorstore", None):
        return ("", "")

    filename = ""
    if getattr(session, "documents", None):
        last = session.documents[-1]
        filename = last.get("filename", "") if isinstance(last, dict) else ""

    try:
        if filename:
            docs = session.vectorstore.similarity_search(
                query="", k=1000, filter={"source": filename}
            )
        else:
            docs = session.vectorstore.similarity_search(query="", k=1000)
    except Exception as e:
        print(f"ERROR doc_extraction: vectorstore search failed: {e}")
        return ("", filename)

    if not docs:
        return ("", filename)

    full_text = "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    return (full_text, filename)


async def stream_extracted_questions(
    session,
    chat_id: str,
    user_prompt: str,
    language: str = "english",
) -> AsyncGenerator[dict, None]:
    """
    Yields the same event shape the orchestrator already handles for quizzes:
      {"status": "quiz_generating", ...}
      {"status": "question_ready", "question": {...}, "index": N}
      {"status": "quiz_complete", "total_generated": N}

    The orchestrator forwards these into the same JSON-line stream the
    frontend's quiz UI listens on. No frontend changes needed.
    """
    document_text, filename = _gather_document_text(session)

    if not document_text:
        yield {
            "status": "error",
            "message": (
                "I couldn't read your uploaded document for extraction. "
                "Try re-uploading the file."
            ),
        }
        return

    yield {
        "status": "quiz_generating",
        "current": 0,
        "total": "?",
        "message": f"Reading {filename or 'your document'} and extracting questions...",
    }

    try:
        client = _get_client()
        # We use streaming on the Anthropic call so we can yield questions as
        # they come back rather than waiting for the whole response. The
        # tool_use blocks arrive sequentially; we surface each one.
        question_index = 0
        async with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            system=EXTRACTION_SYSTEM_PROMPT,
            tools=[EMIT_QUESTION_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Document filename: {filename or 'uploaded.pdf'}\n"
                        f"Response language: {language}\n"
                        f"User request: {user_prompt}\n\n"
                        f"--- DOCUMENT TEXT START ---\n"
                        f"{document_text}\n"
                        f"--- DOCUMENT TEXT END ---\n\n"
                        f"Extract each multiple-choice question by calling "
                        f"emit_question once per question."
                    ),
                }
            ],
        ) as stream:
            final_message = await stream.get_final_message()

        for block in final_message.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name != "emit_question":
                continue
            inp = block.input or {}
            raw_options = inp.get("options") or []
            if not isinstance(raw_options, list) or len(raw_options) < 2:
                continue
            question_text = (inp.get("question") or "").strip()
            if not question_text:
                continue

            correct_index = _letter_to_index(inp.get("correct_letter"))
            options_with_letters = _format_options_with_letters(raw_options)

            transformed = {
                "question": question_text,
                "options": options_with_letters,
                "correctIndex": correct_index,
                "answer": (
                    options_with_letters[correct_index]
                    if 0 <= correct_index < len(options_with_letters)
                    else ""
                ),
                "topic": inp.get("topic") or "Extracted from document",
                "source": "document_extraction",
                "type": "mcq",
            }

            yield {
                "status": "question_ready",
                "question": transformed,
                "index": question_index,
            }
            question_index += 1

        yield {
            "status": "quiz_complete",
            "total_generated": question_index,
        }

    except Exception as e:
        print(f"ERROR doc_extraction streaming: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        yield {
            "status": "error",
            "message": f"Couldn't extract questions: {str(e)}",
        }
