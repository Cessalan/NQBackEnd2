"""Authenticated practice tutoring and metered continuation of existing chat quizzes."""
import asyncio
import json
import os
import re
import time
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from services import usage_guard
from services.quiz_tutor import respond, sanitize_settings

# Questions a chat quiz generates in its first stream. This was 5, with the
# rest left to the focused quiz fetching 5-question batches as she went; that
# relay broke, and a Pro student who asked for a 15-question exam three times
# on 2026-10-11 got 5 questions and a debrief each time. Questions are written
# in parallel (one concept pass, then one call each), so 20 arrive about as
# fast as 5. Larger requests still continue in batches from FocusedQuiz.
CHAT_QUIZ_FIRST_BATCH = 20
# Questions per batch of a planned request (services/planned_practice). The
# owner's call (2026-10-11): an exam is shown at its full size but written ten
# at a time, the next ten starting while she still has three to answer, so a
# 100-question request costs only what she actually reaches.
PLANNED_BATCH = 10
# The chat socket has no server heartbeat of its own, and the browser fails a
# stream after 90s of silence (STALL_MS in the frontend's WebSocketManager.js)
# and sends cancel_stream. Writing, judging and rewriting a batch is often
# silent longer than that, so a chat quiz says it is alive every few seconds.
# Same contract as the upload stream: keep this well under STALL_MS.
QUIZ_HEARTBEAT_S = 10


class TutorRequest(BaseModel):
    chat_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)
    question: dict[str, Any]
    selection: dict[str, Any] = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list, max_length=40)
    settings: dict = Field(default_factory=dict)
    performance: dict = Field(default_factory=dict)
    language: str = Field(default="en", max_length=20)


class PracticeRequest(BaseModel):
    chat_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(pattern=r"^[a-zA-Z0-9-]{8,100}$")
    topic: str = Field(min_length=1, max_length=3000)
    count: int = Field(default=5, ge=1, le=10)
    settings: dict = Field(default_factory=dict)
    existing_questions: list[str] = Field(default_factory=list, max_length=200)
    language: str = Field(default="en", max_length=20)


_PLAN_ID = re.compile(r"^req-[0-9a-f]{28}$")


def request_plan_ref(raw_settings):
    """The blueprint a batch continues, from the quiz's saved settings, or None.

    Read from the raw settings: sanitize_settings keeps only what the tutor may
    change. The id is validated, since it names a Firestore document.
    """
    plan = (raw_settings or {}).get("request_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("id"), str) or not _PLAN_ID.match(plan["id"]):
        return None
    offset = plan.get("offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset <= 10000:
        offset = 0
    return {"id": plan["id"], "offset": offset}


def request_view_for_batch(profile, settings, raw_settings):
    """Her requirements for a later batch, rebuilt from the chat's saved brief.

    Rebuilt on the server rather than accepted from the client, so a batch is
    written to what she asked for even if this quiz predates a rule she added.
    """
    from services import practice_profile as pp, request_brief as rb
    raw_settings = raw_settings or {}
    total = settings.get("requested_total") or profile.get("requestedTotal") or 0
    return rb.request_view(profile.get("brief") or [], total=total, assumed=False,
                           exam=raw_settings.get("exam") is True, difficulty=settings.get("difficulty"),
                           formats=pp.effective_formats(profile, settings.get("question_types")),
                           excluded=profile.get("excludedFormats") or [],
                           files=(profile.get("source") or {}).get("files") or [], scope=profile.get("scope"))


def planned_quiz_mode(raw_settings, view):
    """The mode her first batch was written in, held for the rest of the request."""
    if any(i.get("kind") == "style" for i in (view or {}).get("items", [])):
        return "nclex"
    mode = (raw_settings or {}).get("quiz_mode")
    return mode if mode in ("knowledge", "nclex", "applied") else "nclex"


class DebriefRequest(BaseModel):
    chat_id: str = Field(min_length=1, max_length=200)
    message_id: str = Field(min_length=1, max_length=200, pattern=r'^[^/]+$')
    language: str = Field(default='en', max_length=20)


class FlashcardTutorRequest(DebriefRequest):
    card_index: int = Field(ge=0, le=199)
    revealed: bool = False
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict] = Field(default_factory=list, max_length=16)


def owner(request, chat_id):
    from firebase_admin import auth, firestore
    token = request.headers.get("authorization", "")
    if not token.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in to continue practice.")
    try:
        uid = auth.verify_id_token(token[7:])["uid"]
    except Exception:
        raise HTTPException(401, "Your session expired. Please sign in again.")
    chat = firestore.client().collection("chats").document(chat_id).get()
    if not chat.exists or chat.to_dict().get("userId") != uid:
        raise HTTPException(403, "You cannot access this conversation.")
    return uid


def reserve(uid, request):
    from firebase_admin import firestore
    db = firestore.client()
    user = db.collection("users").document(uid)
    reservation = user.collection("practiceRequests").document(request.request_id)
    @firestore.transactional
    def run(tx):
        prior = reservation.get(transaction=tx)
        if prior.exists:
            data = prior.to_dict()
            if data.get("chat_id") != request.chat_id:
                raise HTTPException(409, "Request belongs to another conversation.")
            if data.get("status") == "complete":
                return data
            raise HTTPException(409, "This batch is already being prepared.")
        snap = user.get(transaction=tx)
        usage = (snap.to_dict() or {}).get("usage", {})
        now = int(time.time() * 1000)
        start = usage.get("windowStart", 0) or 0
        used = max(0, usage.get("count", 0) or 0) if now - start < usage_guard.WINDOW_MS else 0
        start = start if now - start < usage_guard.WINDOW_MS else now
        pro = usage.get("tier") == "pro"
        count = min(request.count, 200 - len(request.existing_questions), request.count if pro else max(0, usage_guard.FREE_LIMIT - used))
        if count <= 0:
            raise HTTPException(429, usage_guard.QUOTA_MESSAGE if not pro else "This practice has reached 200 questions.")
        data = {"chat_id": request.chat_id, "count": count, "charged": 0 if pro else count,
                "windowStart": start, "status": "pending", "questions": []}
        if not pro:
            tx.set(user, {"usage": {**usage, "windowStart": start, "count": used + count}}, merge=True)
        tx.set(reservation, data)
        return data
    return run(db.transaction())


def finish(uid, request_id, questions):
    from firebase_admin import firestore
    db = firestore.client()
    user = db.collection("users").document(uid)
    reservation = user.collection("practiceRequests").document(request_id)
    @firestore.transactional
    def run(tx):
        data = reservation.get(transaction=tx).to_dict() or {}
        usage = (user.get(transaction=tx).to_dict() or {}).get("usage", {})
        if data.get("status") == "complete":
            return
        refund = max(0, data.get("charged", 0) - len(questions))
        if refund and usage.get("windowStart") == data.get("windowStart"):
            tx.set(user, {"usage": {**usage, "count": max(0, usage.get("count", 0) - refund)}}, merge=True)
        tx.set(reservation, {"status": "complete", "questions": questions}, merge=True)
    run(db.transaction())


async def remember_tutor_settings(chat_id, message, settings):
    """A settings change made through the in-quiz tutor is a change to the chat.

    Formats come from the student's own words (practice_profile.parse_changes),
    not from the tutor model's settings, so a model paraphrase can't quietly
    re-enable an excluded format. Returns the settings the quiz should use.
    """
    from services import practice_profile as pp, practice_profile_store
    try:
        changes = pp.parse_changes(message)
        if 'match_examples' in changes:
            current = await practice_profile_store.load(chat_id)
            changes['generation_instructions'] = (current.get('generationInstructions') or '')[:2500] + '\nLatest correction: ' + message[:1400]
        if settings.get("difficulty") in ("easy", "medium", "hard"):
            changes["difficulty"] = settings["difficulty"]
        if isinstance(settings.get("requested_total"), int):
            changes["requested_total"] = settings["requested_total"]
        profile, changed = pp.merge(await practice_profile_store.load(chat_id), changes, updated_by="practice")
        if changed:
            await practice_profile_store.save(chat_id, profile)
        if settings.get("question_types") or any(k in changes for k in ("exclude", "include", "only")):
            return {**settings, "question_types": pp.effective_formats(profile, settings.get("question_types"), message)}
        return settings
    except Exception as e:
        print(f"practice_profile: tutor settings not remembered for {chat_id}: {e}")
        return settings


def build_router(setup_session):
    router = APIRouter()
    @router.post('/flashcards/tutor')
    async def flashcard_tutor(body: FlashcardTutorRequest, request: Request):
        await asyncio.to_thread(owner, request, body.chat_id)
        if len(json.dumps(body.model_dump(), default=str)) > 40000:
            raise HTTPException(413, 'Card discussion is too large.')
        from services.flashcard_tutor import respond_to_card
        try:
            return await respond_to_card(body)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, 'Your tutor could not respond. Please retry.')

    @router.post('/quiz/debrief')
    async def debrief(body: DebriefRequest, request: Request):
        await asyncio.to_thread(owner, request, body.chat_id)
        from services.practice_debrief import create_debrief
        return await create_debrief(body.chat_id, body.message_id, body.language)

    @router.post("/study/reasoning")
    @router.post("/quiz/tutor")
    async def tutor(body: TutorRequest, request: Request):
        await asyncio.to_thread(owner, request, body.chat_id)
        if len(json.dumps(body.model_dump(), default=str)) > 65000:
            raise HTTPException(413, "Question context is too large.")
        try:
            session = await setup_session(body.chat_id, body.language)
            if request.url.path.endswith("/study/reasoning"):
                return await respond(body, session, reasoning=True)
            result = await respond(body, session)
            if result.get("action") == "configure":
                result["settings"] = await remember_tutor_settings(body.chat_id, body.message, result.get("settings") or {})
            return result
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, "Your tutor could not respond. Your quiz is saved; please retry.")

    @router.post("/quiz/practice-stream")
    async def practice(body: PracticeRequest, request: Request):
        uid = await asyncio.to_thread(owner, request, body.chat_id)
        allocation = await asyncio.to_thread(reserve, uid, body)
        async def stream():
            questions = []
            if allocation.get("status") == "complete":
                for q in allocation.get("questions", []):
                    yield json.dumps({"status": "question_ready", "question": q}) + "\n"
                yield json.dumps({"status": "quiz_complete"}) + "\n"
                return
            try:
                from services.quiz_with_bank import stream_quiz_questions
                from services import practice_profile as pp, practice_profile_store
                session = await setup_session(body.chat_id, body.language)
                settings = sanitize_settings(body.settings)
                # The chat's saved settings apply to every batch, including a
                # quiz that was generated before the student excluded a format.
                profile = await practice_profile_store.load(body.chat_id)
                session.practice_profile = profile
                pasted = profile["source"]["pastedText"] if profile["source"]["kind"] == "pasted" else None
                yield json.dumps({"status": "quiz_generating", "total": allocation["count"]}) + "\n"
                seen = {q.strip().lower() for q in body.existing_questions}
                view = request_view_for_batch(profile, settings, body.settings)
                plan = request_plan_ref(body.settings)
                # "Hard" is relative to her: a student getting most of them
                # right gets a harder next batch, and is told why.
                from services import request_brief as rb
                performance = body.settings.get("performance") if isinstance(body.settings.get("performance"), dict) else {}
                harder = rb.escalation(performance) if rb.is_hard(view) else None
                if harder:
                    yield json.dumps({"status": "quiz_escalated", "answered": performance.get("answered"),
                                      "correct": performance.get("correct")}) + "\n"
                if plan:
                    # A planned request continues from its blueprint by index:
                    # the next ten slots after the questions she already has.
                    from services.planned_practice import PlanUnavailable, stream_planned
                    try:
                        async for chunk in stream_planned(
                                session=session, chat_id=body.chat_id, view=view,
                                total=settings.get("requested_total") or len(body.existing_questions) + allocation["count"],
                                offset=plan["offset"], start=len(body.existing_questions), count=allocation["count"],
                                difficulty=settings.get("difficulty", "medium"),
                                question_types=pp.effective_formats(profile, settings.get("question_types")),
                                quiz_mode=planned_quiz_mode(body.settings, view), learning_objective="exam_prep",
                                existing_questions=body.existing_questions, source_text=pasted,
                                focus_topic=body.settings.get("focus_topic") if isinstance(body.settings.get("focus_topic"), str) else None,
                                plan_id=plan["id"], announce=False, is_cancelled=lambda: _is_cancelled(body.chat_id),
                                escalation=harder):
                            if chunk.get("status") == "question_ready":
                                q = chunk["question"]
                                key = q.get("question", "").strip().lower()
                                if key and key not in seen:
                                    questions.append(q)
                                    seen.add(key)
                                    yield json.dumps({"status": "question_ready", "question": q}) + "\n"
                        if not questions:
                            raise RuntimeError("No new questions returned")
                        await asyncio.to_thread(finish, uid, body.request_id, questions)
                        yield json.dumps({"status": "quiz_complete", "total_generated": len(questions)}) + "\n"
                        return
                    except PlanUnavailable as e:
                        print(f"🗺️ PLAN unavailable for batch in {body.chat_id}: {e}")
                async for chunk in stream_quiz_questions(topic=body.topic, difficulty=settings.get("difficulty", "medium"),
                    num_questions=allocation["count"], source="documents" if session.vectorstore or profile['source']['kind'] == 'uploads' else "scratch", session=session,
                    chat_id=body.chat_id, question_types=pp.effective_formats(profile, settings.get("question_types")),
                    quiz_mode="nclex", learning_objective="exam_prep", existing_questions=body.existing_questions,
                    index_offset=len(body.existing_questions), user_prompt=settings.get("scope", ""),
                    source_text=pasted, guidance="\n\n".join(g for g in (pp.generation_guidance(profile), harder) if g),
                    requested_total=settings.get('requested_total') or profile.get('requestedTotal'),
                    plan_id=settings.get('plan_id'), request=view):
                    if chunk.get("status") == "question_ready":
                        q = chunk["question"]
                        key = q.get("question", "").strip().lower()
                        if key and key not in seen:
                            questions.append(q)
                            seen.add(key)
                            yield json.dumps({"status": "question_ready", "question": q}) + "\n"
                    elif chunk.get("status") in ('practice_plan_ready', 'material_analyzing'):
                        yield json.dumps(chunk) + '\n'
                    elif chunk.get("status") == "error":
                        yield json.dumps(chunk) + '\n'
                        return
                if not questions:
                    raise RuntimeError("No new questions returned")
                await asyncio.to_thread(finish, uid, body.request_id, questions)
                yield json.dumps({"status": "quiz_complete", "total_generated": len(questions)}) + "\n"
            except Exception:
                yield json.dumps({"status": "error", "message": "Couldn't finish this batch. Your existing questions are safe; please retry."}) + "\n"
            finally:
                await asyncio.shield(asyncio.to_thread(finish, uid, body.request_id, questions))
        return StreamingResponse(stream(), media_type="application/x-ndjson")
    return router


def _material_backed(source, session, source_text=None):
    profile = getattr(session, "practice_profile", None) or {}
    kind = (profile.get("source") or {}).get("kind")
    return bool(source_text) or kind == "pasted" or (bool(getattr(session, "vectorstore", None))
                                                     and (source == "documents" or kind == "uploads"))


def _empathetic_events(message):
    """The empathetic preamble the concept-first generator streams, for the planned path too."""
    yield {"status": "empathetic_message_start", "message": "Understanding your learning needs..."}
    words, text = message.split(), ""
    for i, word in enumerate(words):
        text += word + " "
        if (i + 1) % 4 == 0 or i == len(words) - 1:
            yield {"status": "empathetic_message_chunk", "chunk": text.strip(), "progress": int((i + 1) / len(words) * 100)}
    yield {"status": "empathetic_message_complete", "full_message": message}


async def _with_heartbeat(events, every=QUIZ_HEARTBEAT_S):
    """`events`, plus {"status": "heartbeat"} whenever it is silent for `every` seconds."""
    iterator = events.__aiter__()
    pending = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=every)
            if not done:
                yield {"status": "heartbeat"}
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            yield event
            pending = asyncio.ensure_future(iterator.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration, Exception):  # noqa: BLE001
                pass
        await iterator.aclose()


def _is_cancelled(chat_id):
    from tools.quiztools import get_connection_manager
    manager = get_connection_manager()
    return bool(manager and chat_id and manager.is_cancelled(chat_id))


async def metered_chat_quiz_stream(**kwargs):
    """Initial chat quizzes use the same atomic question budget as continuation.

    A request she made (`request`, services/request_brief) over her own
    material is written from a blueprint of that material, PLANNED_BATCH at a
    time (services/planned_practice). Anything else, or a plan that cannot be
    built, goes to the concept-first generator, which still receives her
    requirements and checks every question against them.
    """
    from types import SimpleNamespace
    from uuid import uuid4
    from services.quiz_with_bank import stream_quiz_questions
    request_view = kwargs.pop("request", None)
    focus_topic = kwargs.pop("focus_topic", None)
    chat_id = kwargs["chat_id"]
    session = kwargs["session"]
    uid = await asyncio.to_thread(usage_guard._resolve_uid, chat_id)
    if not uid:
        yield {"status": "error", "message": "Save your conversation before starting practice."}
        return
    quota = await asyncio.to_thread(usage_guard.check_quota, chat_id)
    requested = max(1, min(200, kwargs.get("requested_total") or kwargs.get("num_questions") or 5))
    allowed = requested if quota.get("tier") == "pro" else min(requested, quota.get("remaining", 0))
    planned = bool(request_view) and os.getenv("PLANNED_PRACTICE", "1") != "0" \
        and _material_backed(kwargs.get("source"), session, kwargs.get("source_text"))
    first_batch = PLANNED_BATCH if planned else CHAT_QUIZ_FIRST_BATCH
    request = SimpleNamespace(chat_id=chat_id, request_id=str(uuid4()), count=min(first_batch, max(1, allowed)), existing_questions=[])
    try:
        allocation = await asyncio.to_thread(reserve, uid, request)
    except HTTPException as exc:
        yield {"status": "error", "message": exc.detail, "code": "quota_exceeded" if exc.status_code == 429 else "practice_failed"}
        return
    if request_view:
        # What she asked for against what her plan allows, before any
        # question: a free student who asks for 100 is told what she gets.
        yield {"status": "quiz_allowance", "requested": requested, "allowed": allowed,
               "pro": quota.get("tier") == "pro", "batch": allocation["count"]}
    questions = []
    stamp = {"quota_charged": True, "requested_total": max(allocation["count"], allowed)}
    try:
        if planned:
            from services.planned_practice import PlanUnavailable, stream_planned
            empathetic = kwargs.get("empathetic_message")
            try:
                async for chunk in _with_heartbeat(stream_planned(
                        session=session, chat_id=chat_id, view=request_view, total=allowed,
                        count=allocation["count"], difficulty=kwargs.get("difficulty") or "medium",
                        question_types=kwargs.get("question_types"), quiz_mode=kwargs.get("quiz_mode") or "knowledge",
                        learning_objective=kwargs.get("learning_objective") or "general",
                        existing_questions=kwargs.get("existing_questions"), source_text=kwargs.get("source_text"),
                        focus_topic=focus_topic, scope_text=kwargs.get("topic"),
                        additional_context=kwargs.get("additional_context"),
                        is_cancelled=lambda: _is_cancelled(chat_id))):
                    if empathetic and chunk.get("status") == "quiz_planning":
                        for event in _empathetic_events(empathetic):
                            yield {**event, **stamp}
                        empathetic = None
                    if chunk.get("status") == "question_ready":
                        questions.append(chunk["question"])
                    if chunk.get("status") == "quiz_complete":
                        await asyncio.to_thread(finish, uid, request.request_id, questions)
                    yield {**chunk, **stamp}
                return
            except PlanUnavailable as e:
                # Raised only before the first question, so nothing is half-delivered.
                print(f"🗺️ PLAN unavailable for {chat_id}, using the concept-first generator: {e}")
        kwargs["requested_total"] = allowed
        kwargs["num_questions"] = allocation["count"]
        kwargs["request"] = request_view
        async for chunk in _with_heartbeat(stream_quiz_questions(**kwargs)):
            if chunk.get("status") == "question_ready":
                questions.append(chunk["question"])
            if chunk.get("status") == "quiz_complete":
                await asyncio.to_thread(finish, uid, request.request_id, questions)
            yield {**chunk, **stamp}
    finally:
        await asyncio.shield(asyncio.to_thread(finish, uid, request.request_id, questions))


async def metered_extracted_questions(**kwargs):
    """Document extraction must respect the same budget as generated questions."""
    from types import SimpleNamespace
    from uuid import uuid4
    from services.doc_extraction import stream_extracted_questions
    uid = await asyncio.to_thread(usage_guard._resolve_uid, kwargs['chat_id'])
    if not uid:
        yield {'status': 'error', 'message': 'Save your conversation before starting practice.'}
        return
    request = SimpleNamespace(chat_id=kwargs['chat_id'], request_id=str(uuid4()), count=200, existing_questions=[])
    try:
        allocation = await asyncio.to_thread(reserve, uid, request)
    except HTTPException as exc:
        yield {'status': 'error', 'message': exc.detail}
        return
    questions = []
    try:
        async for chunk in stream_extracted_questions(**kwargs):
            if chunk.get('status') == 'question_ready':
                questions.append(chunk['question'])
                yield {**chunk, 'quota_charged': True}
                if len(questions) >= allocation['count']:
                    break
            elif chunk.get('status') != 'quiz_complete':
                yield chunk
        await asyncio.to_thread(finish, uid, request.request_id, questions)
        yield {'status': 'quiz_complete', 'total_generated': len(questions), 'quota_charged': True}
    finally:
        await asyncio.shield(asyncio.to_thread(finish, uid, request.request_id, questions))
