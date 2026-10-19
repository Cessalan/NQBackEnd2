"""
Writes a requested practice from its blueprint, one batch at a time.

See services/request_brief.py for what she asked for, services/practice_plan.py
for how that becomes a blueprint over her whole material, and
services/requirement_check.py for how each question is checked against her
words before she sees it. This module is the IO around them: the material
catalogue, one model call to select goals when reading is needed, the saved
plan, the question writers and the stream events.

THE CONTRACT WITH THE FRONTEND

- `quiz_plan` is sent before the first question: what the whole request
  delivers (questions per file and per focus), and any requirement her
  material cannot meet, in her language. The loading screen ticks from it.
- Every question carries `request`: the file and section it came from, the
  requirements it was planned for, and `verified`, which is True for a
  requirement only when the checker judged the final question and it met it.
- `plan_id` and `plan_offset` let any later batch be written from the same
  blueprint by index (POST /quiz/practice-stream), so the size she asked for
  is a property of the plan, not of a number the client must remember.

Anything here that fails raises PlanUnavailable before the first question, and
the caller falls back to the concept-first generator with her requirements in
its prompt. A plan never half-delivers.
"""
import asyncio
import hashlib
import json
import random
import re
import time

from services import practice_plan as plan_model
from services import request_brief
from services import requirement_check

SELECTION_MODEL = "claude-haiku-5-5"
MAX_PASSAGE_CHARS = 5000
MAX_AVOID = 12
# Once she has a question, the rest of a batch is not waited on forever. A
# knowledge quiz on 2026-10-16 delivered one question of five and then went
# quiet while the others were judged and rewritten; the browser gave up at 90s
# of silence and cancelled the stream, so she was shown nothing at all. A slot
# still unfinished after this is dropped (its quota refunded by finish()).
BATCH_DEADLINE_S = 120


class PlanUnavailable(Exception):
    """The request cannot be planned from material; use the concept-first path."""


# ── Selection: the only model call in planning ─────────────────────────────

SELECTION_PROMPT = """A nursing student asked for a practice exam from her own uploaded material. Below is the complete catalogue of learning goals in that material, grouped by file and section, each with an id.

Decide two things, using ONLY the ids in the catalogue:
1. in_scope: the goals inside the scope she asked for. If her scope is all of her material (for example "these chapters", "both chapters", "everything", or no narrower subject), set all_in_scope to true and leave in_scope empty.
2. serves: for each FOCUS below, EVERY in-scope goal whose question would genuinely be about that focus. Be inclusive within the focus and strict outside it: a goal that only mentions the focus in passing does not serve it. If no goal serves a focus, return an empty list for it; never substitute something else.

HER REQUEST:
{request}

CATALOGUE:
{catalogue}"""


def selection_schema():
    return {
        "type": "object",
        "properties": {
            "all_in_scope": {"type": "boolean"},
            "in_scope": {"type": "array", "items": {"type": "string"}},
            "serves": {"type": "array", "items": {
                "type": "object",
                "properties": {"focus": {"type": "string"}, "goals": {"type": "array", "items": {"type": "string"}}},
                "required": ["focus", "goals"], "additionalProperties": False}},
        },
        "required": ["all_in_scope", "in_scope", "serves"],
        "additionalProperties": False,
    }


def catalogue_text(goals):
    lines, file_name, chapter = [], None, None
    for index, goal in enumerate(goals):
        if goal["file"] != file_name:
            file_name, chapter = goal["file"], None
            lines.append(f"\nFILE: {file_name}")
        if goal["chapter"] != chapter:
            chapter = goal["chapter"]
            lines.append(f"## {chapter}")
        lines.append(f"g{index}: {goal['topic'][:70]} | {goal['outcome'][:90]}")
    return "\n".join(lines).strip()


def request_text(view, scope_text, focus_topic):
    lines = []
    if focus_topic:
        lines.append(f"Scope: only {focus_topic}")
    elif scope_text:
        lines.append(f"Scope in her words: {scope_text}")
    for item in (view or {}).get("items", []):
        if item.get("kind") in ("scope", "focus", "rule"):
            said = f' (she said: "{item["said"]}")' if item.get("said") else ""
            tag = f'FOCUS (use focus id "{item["id"]}")' if item["kind"] == "focus" else item["kind"].upper()
            lines.append(f"{tag}: {item.get('instruction') or item.get('label')}{said}")
    return "\n".join(lines) or "Scope: all of her material"


async def select_goals(goals, view, scope_text, focus_topic, language):
    from services.requirement_check import _get_client
    prompt = SELECTION_PROMPT.format(request=request_text(view, scope_text, focus_topic),
                                     catalogue=catalogue_text(goals))
    started = time.perf_counter()
    response = await _get_client().messages.create(
        model=SELECTION_MODEL, max_tokens=6000,
        messages=[{"role": "user", "content": prompt}],
        # Matching goals to a focus is reading, not reasoning: with thinking
        # on, 2,200 of the output tokens were thought and the call took 11.6s
        # (2026-10-11, 257 goals) while the student watched a loading screen.
        thinking={"type": "disabled"},
        extra_body={"output_config": {"effort": "low",
                                      "format": {"type": "json_schema", "schema": selection_schema()}}},
    )
    if getattr(response, "stop_reason", None) in ("refusal", "max_tokens"):
        raise PlanUnavailable(f"selection stopped: {response.stop_reason}")
    text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
    usage = getattr(response, "usage", None)
    print(f"🗺️ PLAN selection in {time.perf_counter() - started:.1f}s over {len(goals)} goals "
          f"(in={getattr(usage, 'input_tokens', '?')} out={getattr(usage, 'output_tokens', '?')})")
    return json.loads(text)


# ── Persistence ───────────────────────────────────────────────────────────

def _plan_ref(chat_id, plan_id):
    from firebase_admin import firestore
    return firestore.client().collection("chats").document(chat_id).collection("practicePlans").document(plan_id)


def load_plan(chat_id, plan_id):
    try:
        data = _plan_ref(chat_id, plan_id).get().to_dict()
        return data if data and data.get("kind") == "request" else None
    except Exception as e:  # noqa: BLE001
        print(f"planned_practice: load failed {chat_id}/{plan_id}: {e}")
        return None


def save_plan(chat_id, plan):
    try:
        _plan_ref(chat_id, plan["id"]).set(plan)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"planned_practice: save failed {chat_id}/{plan['id']}: {e}")
        return False


def advance_cursor(chat_id, plan_id, end):
    """The next unused slot of this plan in this chat, so a second request continues it."""
    try:
        ref = _plan_ref(chat_id, plan_id)
        current = (ref.get().to_dict() or {}).get("cursor", 0)
        if end > current:
            ref.update({"cursor": end})
    except Exception as e:  # noqa: BLE001
        print(f"planned_practice: cursor update failed {chat_id}/{plan_id}: {e}")


def plan_identity(goals, selection_key, focus_items, formats):
    data = {"selection": selection_key, "catalogue": hashlib.sha256("|".join(g["id"] for g in goals).encode()).hexdigest(),
            "focus": [[i["id"], i.get("share")] for i in focus_items], "formats": list(formats),
            "version": plan_model.PLAN_VERSION}
    return "req-" + hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:28]


async def resolve_plan(*, chat_id, goals, view, scope_text, focus_topic, formats, language, plan_id=None):
    """The saved plan for this request, built (and saved) when there is none."""
    if plan_id:
        saved = await asyncio.to_thread(load_plan, chat_id, plan_id)
        known = {g["id"] for g in goals}
        if saved and set(saved.get("selection", {}).get("inScope", [])) <= known:
            return saved
    matched = plan_model.match_topic(goals, focus_topic) if focus_topic else []
    # A targeted drill is about its topic; her standing focus shares would
    # pull it off topic, so they wait for the next full practice.
    focus_items = [] if focus_topic else [i for i in (view or {}).get("items", []) if i.get("kind") == "focus"]
    key = plan_model.selection_key(goals, [i for i in view.get("items", []) if i.get("kind") in ("scope", "focus")],
                                   scope_text, focus_topic)
    identity = plan_identity(goals, key, focus_items, formats)
    saved = await asyncio.to_thread(load_plan, chat_id, identity)
    if saved:
        return saved
    if matched:
        selection = {"inScope": matched, "serves": {}, "gaps": []}
    elif plan_model.needs_model_selection(view.get("items", []) if not focus_topic else [{"kind": "scope"}], False,
                                          scope_text=scope_text):
        try:
            raw = await select_goals(goals, view, scope_text, focus_topic, language)
        except PlanUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            raise PlanUnavailable(f"selection failed: {type(e).__name__}: {e}") from e
        selection = plan_model.validate_selection(raw, goals, focus_items)
    else:
        selection = {"inScope": [g["id"] for g in goals], "serves": {}, "gaps": []}
    plan = {"id": identity, "kind": "request", "version": plan_model.PLAN_VERSION, "createdAt": time.time(),
            "selection": selection, "formats": list(formats), "cursor": 0,
            "focus": [{"id": i["id"], "label": i.get("label"), "share": i.get("share")} for i in focus_items],
            "focusTopic": focus_topic or None}
    await asyncio.to_thread(save_plan, chat_id, plan)
    return plan


def plan_summary(plan, goals, total, offset, view, partial=()):
    """What this request delivers, for the loading screen's ticks.

    `partial` names files with image pages the reader could not transcribe
    with confidence; their questions come from the readable rest, and she is
    told so rather than finding a missing slide herself."""
    focus_items = plan.get("focus", [])
    slots = plan_model.schedule(goals, plan["selection"], focus_items, plan["formats"], offset, offset + total)
    cover = plan_model.coverage(slots, focus_items)
    labels = {i["id"]: i.get("label") for i in (view or {}).get("items", []) if i.get("id")}
    gaps = [{**g, "label": labels.get(g["id"])} for g in plan["selection"].get("gaps", [])]
    return {"id": plan["id"], "offset": offset, "total": total, "files": cover["files"], "focus": cover["focus"],
            "chapters": cover["chapters"], "gaps": gaps, "focusTopic": plan.get("focusTopic"),
            "partialFiles": [f for f in partial if f]}


# ── Writing one slot ──────────────────────────────────────────────────────

def slot_content(slot, goal, view, additional_context=None, escalation=None):
    lines = []
    requirements = request_brief.writer_requirements(view)
    if requirements:
        lines.append(requirements)
    if escalation:
        lines.append(escalation)
    focus = next((i for i in (view or {}).get("items", []) if i.get("id") == slot.get("bucket")), None)
    if focus:
        lines.append(f'THIS QUESTION is one of her "{focus.get("label")}" questions: '
                     f'{focus.get("instruction") or focus.get("label")}')
    if additional_context:
        lines.append(f"EXAM RESEARCH BRIEF (this exam's style and topics):\n{additional_context}")
    lines.append(f'SOURCE PASSAGE from her file "{slot["file"]}", section "{slot["chapter"]}". '
                 "Test only what this passage supports; never ask about headings, page layout or the document itself:\n"
                 + (goal.get("context") or "")[:MAX_PASSAGE_CHARS])
    return "\n\n".join(lines)


# Writers are told the slot number and sometimes echo it into the stem
# ("Question 5. Une femme de 32 ans..."), which reads wrong when it arrives
# second, and in exam mode, out of order (2026-10-18).
_NUMBER_PREFIX = re.compile(r"^\s*(?:question|q)\s*\d+\s*[.:)\-–]\s*", re.I)


def strip_number(question):
    if question and isinstance(question.get("question"), str):
        question["question"] = _NUMBER_PREFIX.sub("", question["question"], count=1)
    return question


async def write_question(fmt, concept, content, **kwargs):
    return strip_number(await _write_question(fmt, concept, content, **kwargs))


async def _write_question(fmt, concept, content, *, difficulty, number, language, avoid, quiz_mode, learning_objective,
                          hard=False):
    if hard and fmt in ("mcq", "sata"):
        # Built hard (services/hard_items); the general writer is the fallback.
        from services.hard_items import write_hard
        question = await write_hard(fmt, concept, content, language)
        if question:
            return question
    from tools.casestudy_prompts import generate_casestudy_question
    from tools.matrix_prompts import generate_matrix_question
    from tools.quiztools import _generate_single_question
    from tools.sata_prompts import generate_sata_question
    if fmt == "sata":
        return await generate_sata_question(topic=concept, difficulty=difficulty, question_num=number, language=language,
                                            content_context=content, questions_to_avoid=avoid, quiz_mode=quiz_mode)
    if fmt == "casestudy":
        return await generate_casestudy_question(topic=concept, difficulty=difficulty, question_num=number,
                                                 language=language, content_context=content, questions_to_avoid=avoid)
    if fmt == "matrix":
        return await generate_matrix_question(topic=concept, difficulty=difficulty, question_num=number, language=language,
                                              content_context=content, questions_to_avoid=avoid, quiz_mode=quiz_mode)
    question = await _generate_single_question(content=content, topic=concept, difficulty=difficulty,
                                               question_num=number, language=language, questions_to_avoid=avoid,
                                               target_letter=random.choice("ABCD"), quiz_mode=quiz_mode,
                                               learning_objective=learning_objective)
    if question and "questionType" not in question:
        question["questionType"] = "mcq"
    return question


async def write_checked(slot, goal, view, checks, *, difficulty, language, avoid, quiz_mode, learning_objective,
                        additional_context=None, escalation=None):
    """One slot's question, checked against her requirements, rewritten once on failure."""
    applicable = requirement_check.applicable(checks, slot.get("serves"))
    concept = plan_model.slot_concept(slot, hard=request_brief.is_hard(view))
    content = slot_content(slot, goal, view, additional_context, escalation)
    kwargs = dict(difficulty=difficulty, number=slot["index"] + 1, language=language, avoid=avoid,
                  quiz_mode=quiz_mode, learning_objective=learning_objective, hard=request_brief.is_hard(view))
    question = await write_question(slot["format"], concept, content, **kwargs)
    if not question:
        return None
    results = await requirement_check.judge(question, applicable)
    failed = requirement_check.failures(results)
    if failed:
        print(f"🔁 Slot {slot['index']} broke {[f['id'] for f in failed]}: {[f['why'] for f in failed]}")
        retry = await write_question(slot["format"], concept, content + "\n\n" + requirement_check.repair_note(failed), **kwargs)
        if retry:
            retry_results = await requirement_check.judge(retry, applicable)
            # Keep whichever draft breaks fewer of her requirements; a tie
            # keeps the rewrite, which was told exactly what to fix.
            if len(requirement_check.failures(retry_results)) <= len(failed):
                question, results = retry, retry_results
    verified = requirement_check.verified_map(results) if results is not None else {c["id"]: None for c in applicable}
    question["concept"] = question.get("concept") or slot["topic"][:120]
    question["request"] = {"slot": slot["index"], "file": slot["file"], "chapter": slot["chapter"],
                           "serves": list(slot.get("serves") or []), "bucket": slot.get("bucket"),
                           "verified": verified}
    return question


# ── The stream ────────────────────────────────────────────────────────────

async def stream_planned(*, session, chat_id, view, total, offset=0, start=0, count, difficulty, question_types,
                         quiz_mode, learning_objective, existing_questions=None, source_text=None, focus_topic=None,
                         scope_text=None, additional_context=None, plan_id=None, announce=True, is_cancelled=None,
                         escalation=None):
    """Questions [offset+start, offset+start+count) of a planned request, as stream events.

    Raises PlanUnavailable BEFORE yielding any question when the request cannot
    be planned; never after.
    """
    from services.document_understanding import MaterialError, understand_session
    yield {"status": "material_analyzing", "message": "Reading all of your material…"}
    try:
        analyses = await understand_session(session, source_text, require_ready=False)
    except MaterialError as e:
        raise PlanUnavailable(str(e)) from e
    files = ((getattr(session, "practice_profile", None) or {}).get("source") or {}).get("files") or None
    goals = plan_model.compact_goals(analyses, files)
    partial = [a.get("filename") for a in analyses if plan_model.partly_readable(a)
               and (not files or a.get("filename") in files)]
    if len(goals) < 3:
        raise PlanUnavailable(f"only {len(goals)} usable goals")
    language = getattr(session, "user_language", None) or "en"
    lang = "fr" if str(language).lower().startswith("fr") else "en"
    formats = list(question_types or ["mcq"])
    yield {"status": "quiz_planning", "total": count}
    plan = await resolve_plan(chat_id=chat_id, goals=goals, view=view, scope_text=scope_text, focus_topic=focus_topic,
                              formats=formats, language=lang, plan_id=plan_id)
    if not plan_id and offset == 0 and start == 0:
        # A new request on a plan this chat has already used continues it,
        # so "another 50" does not replay the first 50 slots.
        offset = int(plan.get("cursor") or 0)
    if announce:
        yield {"status": "quiz_plan", "plan": plan_summary(plan, goals, total, offset, view, partial)}

    by_id = {g["id"]: g for g in goals}
    first = offset + start
    slots = plan_model.schedule(goals, plan["selection"], plan.get("focus", []), plan["formats"], first, first + count)
    checks = request_brief.checkable(view)
    avoid = [q for q in (existing_questions or []) if q][-MAX_AVOID:]
    yield {"status": "generating", "current": start + 1, "total": start + count, "plan_id": plan["id"],
           "plan_offset": offset}

    tasks = [asyncio.create_task(write_checked(
        slot, by_id[slot["goalId"]], view, checks, difficulty=difficulty, language=language, avoid=avoid,
        quiz_mode=quiz_mode, learning_objective=learning_objective, additional_context=additional_context,
        escalation=escalation))
        for slot in slots]
    index = start
    delivered = 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BATCH_DEADLINE_S
    pending = set(tasks)
    try:
        while pending:
            if is_cancelled and is_cancelled():
                return
            if delivered and loop.time() >= deadline:
                print(f"⏱️ planned batch: {len(pending)} slot(s) past {BATCH_DEADLINE_S}s, delivering {delivered}")
                break
            wait = max(0.01, min(5, deadline - loop.time())) if delivered else 5
            done, pending = await asyncio.wait(pending, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    question = task.result()
                except Exception as e:  # noqa: BLE001 - one slot never sinks the batch
                    print(f"❌ planned slot failed: {type(e).__name__}: {e}")
                    continue
                if not question:
                    continue
                yield {"status": "question_ready", "question": question, "index": index, "source": "plan"}
                index += 1
                delivered += 1
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
    await asyncio.to_thread(advance_cursor, chat_id, plan["id"], first + count)
    yield {"status": "quiz_complete", "total_generated": delivered, "plan_id": plan["id"], "plan_offset": offset}
