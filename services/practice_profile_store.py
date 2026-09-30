"""Firestore side of services/practice_profile.py: read and write `chats/{id}.practiceProfile`.

Kept apart from the pure module so its rules stay testable without Firebase.
Every function fails OPEN: a profile that can't be read or saved means the
batch is generated the way it was before profiles existed, never an error in
front of a student mid-practice.
"""
import asyncio

from services import practice_profile as pp


def _chat_ref(chat_id):
    from firebase_admin import firestore
    return firestore.client().collection("chats").document(chat_id)


def load_sync(chat_id):
    try:
        snap = _chat_ref(chat_id).get()
        return pp.normalize((snap.to_dict() or {}).get("practiceProfile")) if snap.exists else pp.normalize({})
    except Exception as e:
        print(f"practice_profile: load failed for {chat_id}: {e}")
        return pp.normalize({})


async def load(chat_id):
    return await asyncio.to_thread(load_sync, chat_id)


def save_sync(chat_id, profile):
    try:
        _chat_ref(chat_id).set({"practiceProfile": pp.normalize(profile)}, merge=True)
        return True
    except Exception as e:
        print(f"practice_profile: save failed for {chat_id}: {e}")
        return False


async def save(chat_id, profile):
    return await asyncio.to_thread(save_sync, chat_id, profile)


def from_context(context):
    """The profile from a pre-fetched chat context, seeded if the chat has none."""
    context = context or {}
    saved = context.get("practice_profile")
    if isinstance(saved, dict) and saved:
        return pp.normalize(saved), True
    return pp.seed_from_quiz_settings(context.get("latest_quiz_settings")), False
