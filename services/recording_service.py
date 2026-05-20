"""
Recording Service — handles class lecture recording sessions.

Architecture: client-side chunking.
The frontend records audio with MediaRecorder, rotates the recorder every
~5 minutes to produce independently-decodable webm segments, and POSTs each
segment to /recordings/{id}/chunk. Each chunk is transcribed independently
via Whisper (≤25MB), then stitched on finalize.

Firestore schema (collection: recordings)
{
  recording_id: str
  user_id: str
  chat_id: str | null            # set when finalize attaches recording to a chat
  topic: str
  language: str
  status: "recording" | "transcribing" | "complete" | "cancelled" | "error"
  created_at: timestamp
  updated_at: timestamp
  duration_ms: int               # total accumulated duration
  total_chunks: int              # incremented as chunks arrive
  chunks: [
    {
      index: int,
      text: str,
      duration_ms: int,
      bytes: int,
      storage_path: str,
      transcribed_at: timestamp
    }
  ]
  final_transcript: str | null   # populated on finalize
  transcript_storage_path: str | null
  error: str | null
}

Storage layout:
  recordings/{recording_id}/chunks/chunk_{index:04d}.webm
  chats/{chat_id}/uploads/recording_{recording_id}.txt   (on finalize → chat)
"""

import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import uuid4

from openai import OpenAI
from firebase_admin import firestore, storage

# Whisper hard limit
WHISPER_MAX_BYTES = 25 * 1024 * 1024
RECORDINGS_COLLECTION = "recordings"
CHATS_COLLECTION = "chats"

STATUS_RECORDING = "recording"
STATUS_TRANSCRIBING = "transcribing"
STATUS_COMPLETE = "complete"
STATUS_CANCELLED = "cancelled"
STATUS_ERROR = "error"


def _db():
    return firestore.client()


def _bucket():
    return storage.bucket()


def _now():
    return datetime.now(timezone.utc)


def start_recording(user_id: str, topic: str = "", chat_id: Optional[str] = None,
                    language: str = "en") -> Dict[str, Any]:
    """Create a new recording session. Returns {recording_id}."""
    if not user_id:
        raise ValueError("user_id is required")

    recording_id = uuid4().hex
    db = _db()
    doc = {
        "recording_id": recording_id,
        "user_id": user_id,
        "chat_id": chat_id,
        "topic": topic or "",
        "language": language or "en",
        "status": STATUS_RECORDING,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "duration_ms": 0,
        "total_chunks": 0,
        "chunks": [],
        "final_transcript": None,
        "transcript_storage_path": None,
        "error": None,
    }
    db.collection(RECORDINGS_COLLECTION).document(recording_id).set(doc)
    print(f"🎙️ [Recording] Started {recording_id} for user={user_id} topic='{topic}'")
    return {"recording_id": recording_id}


def _get_recording(recording_id: str) -> Dict[str, Any]:
    snap = _db().collection(RECORDINGS_COLLECTION).document(recording_id).get()
    if not snap.exists:
        raise LookupError(f"Recording {recording_id} not found")
    return snap.to_dict()


def transcribe_chunk(recording_id: str, audio_bytes: bytes, chunk_index: int,
                     duration_ms: int, filename: str = "chunk.webm") -> Dict[str, Any]:
    """
    Transcribe a single audio chunk and append to the recording document.
    Returns {chunk_index, text, total_chunks, accumulated_duration_ms}.
    """
    if len(audio_bytes) == 0:
        raise ValueError("Empty audio chunk")
    if len(audio_bytes) > WHISPER_MAX_BYTES:
        raise ValueError(
            f"Chunk too large ({len(audio_bytes)} bytes). "
            f"Max {WHISPER_MAX_BYTES} bytes. Rotate the MediaRecorder more frequently."
        )

    # Verify the recording exists and is in a valid state
    rec = _get_recording(recording_id)
    if rec["status"] in (STATUS_CANCELLED, STATUS_COMPLETE):
        raise ValueError(f"Recording is {rec['status']}; cannot append chunks")

    # 1. Upload chunk to Storage (best-effort archive for retry/debugging)
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "webm"
    storage_path = f"recordings/{recording_id}/chunks/chunk_{chunk_index:04d}.{ext}"
    try:
        blob = _bucket().blob(storage_path)
        blob.upload_from_string(audio_bytes, content_type="audio/webm")
    except Exception as e:
        # Storage failure shouldn't kill transcription — log and continue
        print(f"⚠️ [Recording] Storage upload failed for {storage_path}: {e}")
        storage_path = None

    # 2. Transcribe via Whisper
    suffix = f".{ext}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        client = OpenAI()
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
            )
        text = transcript.strip() if isinstance(transcript, str) else str(transcript)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. Append chunk to Firestore doc
    chunk_record = {
        "index": int(chunk_index),
        "text": text,
        "duration_ms": int(duration_ms or 0),
        "bytes": len(audio_bytes),
        "storage_path": storage_path,
        "transcribed_at": _now().isoformat(),
    }

    doc_ref = _db().collection(RECORDINGS_COLLECTION).document(recording_id)
    # Use array_union for idempotency on retries
    doc_ref.update({
        "chunks": firestore.ArrayUnion([chunk_record]),
        "total_chunks": firestore.Increment(1),
        "duration_ms": firestore.Increment(int(duration_ms or 0)),
        "status": STATUS_TRANSCRIBING,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

    # Re-read to get the live count for the response
    updated = _get_recording(recording_id)
    print(f"🎙️ [Recording] {recording_id} chunk {chunk_index} ({len(audio_bytes)} bytes) → "
          f"{len(text)} chars. Total chunks: {updated.get('total_chunks')}.")

    return {
        "chunk_index": chunk_index,
        "text": text,
        "total_chunks": updated.get("total_chunks", 0),
        "accumulated_duration_ms": updated.get("duration_ms", 0),
    }


def _stitch_transcript(chunks: List[Dict[str, Any]]) -> str:
    """Order chunks by index and join with paragraph breaks."""
    ordered = sorted(chunks or [], key=lambda c: c.get("index", 0))
    parts = [c.get("text", "").strip() for c in ordered if c.get("text")]
    return "\n\n".join(parts)


def _generate_title_from_transcript(transcript: str, language: str = "en") -> str:
    """
    Generate a short chat title (3-6 words) from the lecture transcript.
    Uses gpt-4.1-nano to match the existing /chat/generate-title behavior.
    Falls back to a generic title if generation fails.
    """
    if not transcript or not transcript.strip():
        return "Recorded lecture"

    # Use just the first ~1500 chars — enough signal, cheap to send
    snippet = transcript.strip()[:1500]

    try:
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate(
            template=(
                "You are an AI assistant for nursing students.\n\n"
                "Below is the opening of a recorded class lecture. Generate a short, "
                "descriptive title in 3 to 6 words.\n\n"
                "Requirements:\n"
                "- Write the title in the same language as the transcript\n"
                "- Be concise and clinically specific\n"
                "- No quotes, no punctuation at the end\n"
                "- Max 6 words\n\n"
                "Transcript:\n{transcript}"
            ),
            input_variables=["transcript"],
        )
        llm = ChatOpenAI(temperature=0.6, model="gpt-4.1-nano", streaming=False)
        chain = prompt | llm | StrOutputParser()
        title = chain.invoke({"transcript": snippet})
        title = (title or "").replace('"', "").replace("'", "").strip()
        # Sanity bounds
        if not title or len(title) > 80:
            return "Recorded lecture"
        return title
    except Exception as e:
        print(f"⚠️ [Recording] Title generation failed: {e}")
        return "Recorded lecture"


def _create_chat_for_recording(user_id: str, title: str, language: str) -> str:
    """Create a new chat document seeded for this recording. Returns chat_id."""
    db = _db()
    chat_doc = {
        "userId": user_id,
        "title": title or "Recorded lecture",
        "description": "Class recording transcribed by NurseQuizAI.",
        "language": language or "en",
        "createdFromRecording": True,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    ref = db.collection(CHATS_COLLECTION).add(chat_doc)
    # add() returns (timestamp, DocumentReference)
    doc_ref = ref[1] if isinstance(ref, tuple) else ref
    return doc_ref.id


def finalize_recording(recording_id: str, topic: Optional[str] = None,
                       action: str = "save", language: Optional[str] = None) -> Dict[str, Any]:
    """
    Stitch the chunk transcripts, save the final transcript to Storage,
    optionally attach it to a (new or existing) chat, and mark the
    recording as complete.

    action:
      - "save":  Just persist the recording + transcript. No chat.
      - "chat":  Create a new chat seeded with the transcript file.
      - "study": Same as "chat" for now — frontend triggers study path from there.

    Returns: {recording_id, chat_id?, transcript_url, transcript_preview, duration_ms}
    """
    valid_actions = ("save", "chat", "study")
    if action not in valid_actions:
        raise ValueError(f"action must be one of {valid_actions}")

    rec = _get_recording(recording_id)
    if rec["status"] == STATUS_CANCELLED:
        raise ValueError("Recording was cancelled; cannot finalize")

    user_id = rec["user_id"]
    lang = language or rec.get("language") or "en"

    # Stitch the transcript
    transcript_text = _stitch_transcript(rec.get("chunks", []))
    if not transcript_text:
        # Empty transcript — still finalize but flag it
        transcript_text = ""
        print(f"⚠️ [Recording] {recording_id} has no transcribed chunks at finalize")

    # Resolve title: user-supplied > stored topic > AI-generated from transcript
    user_topic = (topic or rec.get("topic") or "").strip()
    if user_topic:
        final_topic = user_topic
    else:
        final_topic = _generate_title_from_transcript(transcript_text, lang)
    print(f"🎙️ [Recording] {recording_id} title='{final_topic}'")

    # Decide where to attach the recording
    chat_id = rec.get("chat_id")
    if action in ("chat", "study") and not chat_id:
        chat_id = _create_chat_for_recording(user_id, final_topic, lang)
        print(f"🎙️ [Recording] {recording_id} created new chat {chat_id}")

    # Save transcript to Storage
    if chat_id:
        transcript_path = f"chats/{chat_id}/uploads/recording_{recording_id}.txt"
    else:
        transcript_path = f"recordings/{recording_id}/transcript.txt"

    header = (
        f"# {final_topic}\n"
        f"# Recorded: {_now().isoformat()}\n"
        f"# Duration: {rec.get('duration_ms', 0) / 1000:.0f}s\n"
        f"# Chunks: {len(rec.get('chunks', []))}\n\n"
    )
    full_text = header + transcript_text

    transcript_url = None
    try:
        blob = _bucket().blob(transcript_path)
        blob.upload_from_string(full_text, content_type="text/plain; charset=utf-8")
        try:
            blob.make_public()
            transcript_url = blob.public_url
        except Exception:
            # Private bucket — return signed URL or just the path
            transcript_url = None
    except Exception as e:
        print(f"⚠️ [Recording] Failed to save transcript to {transcript_path}: {e}")

    # Update Firestore
    update = {
        "status": STATUS_COMPLETE,
        "topic": final_topic,
        "final_transcript": transcript_text,
        "transcript_storage_path": transcript_path,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }
    if chat_id and not rec.get("chat_id"):
        update["chat_id"] = chat_id

    _db().collection(RECORDINGS_COLLECTION).document(recording_id).update(update)

    preview = transcript_text[:300] + ("…" if len(transcript_text) > 300 else "")
    print(f"🎙️ [Recording] Finalized {recording_id}: {len(transcript_text)} chars, "
          f"chat={chat_id}, action={action}")

    return {
        "recording_id": recording_id,
        "chat_id": chat_id,
        "title": final_topic,
        "transcript_url": transcript_url,
        "transcript_storage_path": transcript_path,
        "transcript_preview": preview,
        "duration_ms": rec.get("duration_ms", 0),
        "total_chunks": len(rec.get("chunks", [])),
        "action": action,
    }


def cancel_recording(recording_id: str, delete_chunks: bool = True) -> Dict[str, Any]:
    """Mark recording cancelled and (optionally) delete uploaded chunk blobs."""
    rec = _get_recording(recording_id)

    if delete_chunks:
        for chunk in rec.get("chunks", []):
            path = chunk.get("storage_path")
            if not path:
                continue
            try:
                _bucket().blob(path).delete()
            except Exception as e:
                print(f"⚠️ [Recording] Failed to delete chunk {path}: {e}")

    _db().collection(RECORDINGS_COLLECTION).document(recording_id).update({
        "status": STATUS_CANCELLED,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })
    print(f"🎙️ [Recording] Cancelled {recording_id}")
    return {"recording_id": recording_id, "status": STATUS_CANCELLED}


def get_recording(recording_id: str) -> Dict[str, Any]:
    """Return the recording document (for polling clients that don't use Firestore listeners)."""
    rec = _get_recording(recording_id)
    # Convert any Firestore timestamps to ISO for JSON safety
    for key in ("created_at", "updated_at"):
        val = rec.get(key)
        if hasattr(val, "isoformat"):
            rec[key] = val.isoformat()
    return rec
