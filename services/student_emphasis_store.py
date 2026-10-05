"""Firestore side of services/student_emphasis.py.

Reads the chat's own user messages and its practiceProfile, and asks the
vectorstore for the passages of her uploads most likely to carry an exam flag.
Fails OPEN like practice_profile_store: if anything here breaks, the plan is
built exactly as it was before this existed, never an error in front of a
student who is waiting for her plan.
"""
import asyncio

from services import practice_profile as pp
from services import student_emphasis as se

# The vectorstore query that finds instructor flags inside uploaded notes
# (including notes pasted through the Paste-notes modal, which become an
# upload). One embedding call; the plan prompt already costs seconds.
CUE_QUERY = "must know, will be on the exam, high-yield, know the following, be able to"
CUE_CHUNKS = 30


def _user_messages_sync(chat_id):
    from firebase_admin import firestore
    db = firestore.client()
    chat = db.collection("chats").document(chat_id)
    snap = chat.get()
    profile = pp.normalize((snap.to_dict() or {}).get("practiceProfile")) if snap.exists else pp.normalize({})
    # Single-field filter, no order_by: avoids needing a composite index.
    # Chat order is restored from the timestamp afterwards.
    rows = []
    for doc in chat.collection("messages").where(
            filter=firestore.FieldFilter("role", "==", "user")).stream():
        data = doc.to_dict() or {}
        if isinstance(data.get("content"), str):
            rows.append((str(data.get("timestamp") or ""), data["content"]))
    rows.sort(key=lambda r: r[0])
    return [content for _, content in rows], profile


def _upload_text_sync(vectorstore):
    if not vectorstore:
        return ""
    docs = vectorstore.similarity_search(CUE_QUERY, k=CUE_CHUNKS)
    return "\n".join(d.page_content for d in docs)


async def load(chat_id, vectorstore=None, document_text=None):
    """She-told-us signals for this chat. Empty signals on any failure.

    Pass `document_text` when the caller already holds her upload text (the
    /study/plan path does); otherwise the vectorstore is searched for it.
    """
    try:
        messages, profile = await asyncio.to_thread(_user_messages_sync, chat_id)
    except Exception as e:
        print(f"student_emphasis: chat read failed for {chat_id}: {e}")
        messages, profile = [], {}
    if document_text is None:
        try:
            document_text = await asyncio.to_thread(_upload_text_sync, vectorstore)
        except Exception as e:
            print(f"student_emphasis: upload scan failed for {chat_id}: {e}")
            document_text = ""
    try:
        signals = se.gather(messages, profile, document_text)
    except Exception as e:
        print(f"student_emphasis: gather failed for {chat_id}: {e}")
        return se.gather()
    if not se.is_empty(signals):
        print(f"🖍️  Student emphasis for {chat_id}: {len(signals['cues'])} exam flags, "
              f"{len(signals['pasted_text'])} pasted chars, more={signals['more']} less={signals['less']}")
    return signals
