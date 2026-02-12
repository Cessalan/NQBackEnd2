# Orchestrator Review: Performance, Context Quality, and Intelligence

This review focuses on the chat entrypoint at `@app.websocket("/ws/{chat_id}")` and the orchestrator path in `services/orchestrator.py`.

## What is currently good

- WebSocket flow already parallelizes cold-start work (`insights`, `vectorstore`, `context`) via `asyncio.gather`.
- Session reuse is enabled (`ACTIVE_SESSIONS`) with periodic cleanup.
- `process_message` supports pre-fetched context to avoid duplicate DB queries.
- Tool-based architecture is in place for quiz/flashcard/study-sheet/mindmap/audio specialization.

## High-impact bottlenecks found

1. **Per-message Firebase reload risk for tool context**
   - `set_session_context()` was re-fetching files every message and could re-load vectorstore fallback too often.
   - This creates network overhead and latency spikes under chat bursts.

2. **Noisy / oversized conversation history into LLM**
   - Raw conversation history can include serialized payloads (quiz/flashcard JSON).
   - This degrades model reasoning and wastes prompt tokens.

3. **Shared mutable default in session state**
   - `previously_generate_questions_in_quiz = []` was class-level mutable state.
   - This can leak generated-question memory between sessions and hurt quality.

## Changes applied in this patch

### 1) Session context now prefers in-memory cache
- `set_session_context()` now:
  - loads files only when `session.documents` is empty,
  - only attempts vectorstore fallback load when vectorstore is missing.
- This avoids repeated storage calls on every user turn.

### 2) Context sanitization before LLM call
- Added `_sanitize_message_history()` in orchestrator:
  - keeps only recent turns,
  - keeps only text chat roles (`user`, `assistant`),
  - trims oversized messages,
  - drops serialized UI payload snippets (`"quiz_data"`, `"flashcard_data"`).
- Added `_build_messages()` to centralize and simplify message assembly.

### 3) Session state isolation fix
- `previously_generate_questions_in_quiz` changed to `field(default_factory=list)`.
- Prevents cross-chat contamination.

## Suggested next steps (not yet coded)

1. **Router model + worker model split**
   - Use a small fast model for intent/tool routing.
   - Use stronger model only for generation tasks (e.g. explanations, rationale synthesis).

2. **Context memory tiers**
   - Tier 1: last 6-8 turns (raw)
   - Tier 2: rolling summary (compressed learning state)
   - Tier 3: structured profile (`weak_topics`, `recent_scores`, `preferred_language`, `active_exam_mode`)

3. **Adaptive retrieval for nursing tasks**
   - Increase retrieval depth for long-form tasks (study sheet/plan) and reduce for short Q&A.
   - Add simple reranker pass for quiz-source chunk selection.

4. **Quality guardrails for quiz generation**
   - Add preflight validator before streaming first question:
     - topic coverage,
     - difficulty distribution,
     - clinical safety language check,
     - duplicate similarity threshold.

5. **Observability and metrics**
   - Track p50/p95 for:
     - websocket receive → first status,
     - first status → first content token,
     - tool invoke time by tool name,
     - total response duration by message type.

## Expected impact

- Faster average turn latency for active sessions (fewer remote context fetches).
- Better reasoning consistency from cleaner prompt context.
- Better per-session personalization stability with fixed session-state isolation.
