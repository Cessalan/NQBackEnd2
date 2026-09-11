"""Authenticated practice tutoring and metered continuation of existing chat quizzes."""
import asyncio
import json
import time
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from services import usage_guard
from services.quiz_tutor import respond, sanitize_settings


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
    count: int = Field(default=5, ge=1, le=5)
    settings: dict = Field(default_factory=dict)
    existing_questions: list[str] = Field(default_factory=list, max_length=200)
    language: str = Field(default="en", max_length=20)


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


def build_router(setup_session):
    router = APIRouter()
    @router.post("/quiz/tutor")
    async def tutor(body: TutorRequest, request: Request):
        await asyncio.to_thread(owner, request, body.chat_id)
        if len(json.dumps(body.model_dump(), default=str)) > 65000:
            raise HTTPException(413, "Question context is too large.")
        try:
            session = await setup_session(body.chat_id, body.language)
            return await respond(body, session)
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
                session = await setup_session(body.chat_id, body.language)
                settings = sanitize_settings(body.settings)
                yield json.dumps({"status": "quiz_generating", "total": allocation["count"]}) + "\n"
                seen = {q.strip().lower() for q in body.existing_questions}
                async for chunk in stream_quiz_questions(topic=body.topic, difficulty=settings.get("difficulty", "medium"),
                    num_questions=allocation["count"], source="documents" if session.vectorstore else "scratch", session=session,
                    chat_id=body.chat_id, question_types=settings.get("question_types", ["mcq", "sata", "casestudy"]),
                    quiz_mode="nclex", learning_objective="exam_prep", existing_questions=body.existing_questions,
                    index_offset=len(body.existing_questions), user_prompt=settings.get("scope", "")):
                    if chunk.get("status") == "question_ready":
                        q = chunk["question"]
                        key = q.get("question", "").strip().lower()
                        if key and key not in seen:
                            questions.append(q)
                            seen.add(key)
                            yield json.dumps({"status": "question_ready", "question": q}) + "\n"
                    elif chunk.get("status") == "error":
                        raise RuntimeError("Question generation failed")
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


async def metered_chat_quiz_stream(**kwargs):
    """Initial chat quizzes use the same atomic question budget as continuation."""
    from types import SimpleNamespace
    from uuid import uuid4
    from services.quiz_with_bank import stream_quiz_questions
    chat_id = kwargs["chat_id"]
    uid = await asyncio.to_thread(usage_guard._resolve_uid, chat_id)
    if not uid:
        yield {"status": "error", "message": "Save your conversation before starting practice."}
        return
    quota = await asyncio.to_thread(usage_guard.check_quota, chat_id)
    requested = max(1, min(200, kwargs.get("requested_total") or kwargs.get("num_questions") or 5))
    allowed = requested if quota.get("tier") == "pro" else min(requested, quota.get("remaining", 0))
    request = SimpleNamespace(chat_id=chat_id, request_id=str(uuid4()), count=min(5, max(1, allowed)), existing_questions=[])
    try:
        allocation = await asyncio.to_thread(reserve, uid, request)
    except HTTPException as exc:
        yield {"status": "error", "message": exc.detail, "code": "quota_exceeded" if exc.status_code == 429 else "practice_failed"}
        return
    questions = []
    kwargs.pop("requested_total", None)
    kwargs["num_questions"] = allocation["count"]
    try:
        async for chunk in stream_quiz_questions(**kwargs):
            if chunk.get("status") == "question_ready":
                questions.append(chunk["question"])
            if chunk.get("status") == "quiz_complete":
                await asyncio.to_thread(finish, uid, request.request_id, questions)
            yield {**chunk, "quota_charged": True, "requested_total": max(allocation["count"], allowed)}
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
