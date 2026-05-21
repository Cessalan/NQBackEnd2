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


def _extract_live_key_points(text: str, language: str = "en") -> List[str]:
    """
    Extract 1-2 key points (bullet points) from a chunk transcript.
    Uses gpt-4.1-nano for fast/cheap summary.
    """
    if not text or len(text.strip()) < 30:
        return []

    try:
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        import json

        prompt = PromptTemplate(
            template=(
                "You are an assistant for nursing students listening to a live class lecture.\n"
                "Extract 1 to 2 key concepts or points from this transcript segment.\n"
                "Requirements:\n"
                "- Keep each point extremely concise (5-10 words)\n"
                "- Write in the same language as the transcript\n"
                "- Focus on clinical/medical/nursing facts\n"
                "- Return ONLY a JSON list of strings, e.g. [\"point 1\", \"point 2\"]\n"
                "- If there are no clear key concepts or clinical points in the segment, return an empty list: []\n\n"
                "Transcript Segment:\n{text}"
            ),
            input_variables=["text"],
        )
        llm = ChatOpenAI(temperature=0.2, model="gpt-4.1-nano", streaming=False)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"text": text})

        # Parse result
        result_json = result.strip()
        if result_json.startswith("```"):
            parts = result_json.split("```")
            if len(parts) >= 3:
                result_json = parts[1]
                if result_json.startswith("json"):
                    result_json = result_json[4:]
            result_json = result_json.strip()

        # Fallback parsing in case JSON is wrapped in quotes
        if result_json.startswith('"') and result_json.endswith('"'):
            try:
                result_json = json.loads(result_json)
            except Exception:
                pass

        key_points = json.loads(result_json)
        if isinstance(key_points, list):
            return [str(kp).strip() for kp in key_points if kp]
        return []
    except Exception as e:
        print(f"⚠️ [Recording] Failed to extract key points: {e}")
        return []


def transcribe_chunk(recording_id: str, audio_bytes: bytes, chunk_index: int,
                     duration_ms: int, filename: str = "chunk.webm") -> Dict[str, Any]:
    """
    Transcribe a single audio chunk and append to the recording document.
    Returns {chunk_index, text, total_chunks, accumulated_duration_ms, key_points}.
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

    # Generate live key points
    key_points = _extract_live_key_points(text, rec.get("language") or "en")

    return {
        "chunk_index": chunk_index,
        "text": text,
        "total_chunks": updated.get("total_chunks", 0),
        "accumulated_duration_ms": updated.get("duration_ms", 0),
        "key_points": key_points,
    }


def _stitch_transcript(chunks: List[Dict[str, Any]]) -> str:
    """Order chunks by index and join with paragraph breaks."""
    ordered = sorted(chunks or [], key=lambda c: c.get("index", 0))
    parts = [c.get("text", "").strip() for c in ordered if c.get("text")]
    return "\n\n".join(parts)


def _stitch_transcript_with_events(chunks: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> str:
    """Order chunks by index, map event timestamps to chunks, and insert markers."""
    ordered_chunks = sorted(chunks or [], key=lambda c: c.get("index", 0))
    
    # Calculate starting and ending times for each chunk
    current_time = 0
    chunk_windows = []
    for chunk in ordered_chunks:
        duration = chunk.get("duration_ms", 0)
        chunk_windows.append({
            "chunk": chunk,
            "start_ms": current_time,
            "end_ms": current_time + duration
        })
        current_time += duration

    # Map events to chunk indices
    chunk_annotations = {c["chunk"]["index"]: [] for c in chunk_windows}
    
    for event in (events or []):
        ts = event.get("timestamp_ms", 0)
        etype = event.get("type", "important")
        
        # Find matching chunk window
        matched_chunk_idx = None
        for win in chunk_windows:
            if win["start_ms"] <= ts <= win["end_ms"]:
                matched_chunk_idx = win["chunk"]["index"]
                break
                
        # Fallback to last chunk if timestamp is out of bounds
        if matched_chunk_idx is None and chunk_windows:
            matched_chunk_idx = chunk_windows[-1]["chunk"]["index"]
            
        if matched_chunk_idx is not None:
            icon = "⭐ IMPORTANT CONCEPT MARKED BY STUDENT" if etype == "important" else "🚩 STUDENT FLAGGED CONFUSION"
            chunk_annotations[matched_chunk_idx].append(f"\n[{icon}]\n")

    # Stitch the transcript
    parts = []
    for win in chunk_windows:
        chunk = win["chunk"]
        text = chunk.get("text", "").strip()
        idx = chunk["index"]
        annots = chunk_annotations.get(idx, [])
        
        if annots:
            annot_str = "".join(annots)
            if text:
                text = f"{annot_str}{text}"
            else:
                text = annot_str
                
        if text:
            parts.append(text)
            
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
                       action: str = "save", language: Optional[str] = None,
                       events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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

    # Stitch the transcript (incorporating highlight annotations if present)
    if events:
        transcript_text = _stitch_transcript_with_events(rec.get("chunks", []), events)
    else:
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
        "events": events or [],
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


async def cancel_recording(recording_id: str, delete_chunks: bool = True) -> Dict[str, Any]:
    """
    Mark a recording cancelled and clean up everything it left behind.

    Pre-finalize (user stopped before transcript was stitched & embedded):
      - Delete chunk audio blobs from Storage
      - Mark recording cancelled in Firestore

    Post-finalize (user pressed "Discard recording" after the transcript was
    embedded into a chat): also delete the transcript file, the per-file
    vectorstore, the recording's vectors from the chat's combined
    vectorstore, evict the in-memory session so cached vectors don't haunt
    the next chat load, and — if the chat was created specifically for this
    recording and has no other uploads — delete the orphan chat doc.
    """
    rec = _get_recording(recording_id)
    chat_id = rec.get("chat_id")
    is_finalized = (
        rec.get("status") == STATUS_COMPLETE
        or bool(rec.get("transcript_storage_path"))
    )

    # 1. Delete chunk audio blobs
    if delete_chunks:
        for chunk in rec.get("chunks", []):
            path = chunk.get("storage_path")
            if not path:
                continue
            try:
                _bucket().blob(path).delete()
            except Exception as e:
                print(f"⚠️ [Recording] Failed to delete chunk {path}: {e}")

    # 2. Post-finalize cleanup — only when a transcript was produced
    if is_finalized and chat_id:
        filename = f"recording_{recording_id}.txt"
        bucket = _bucket()

        # 2a. Delete transcript file from Storage
        transcript_path = rec.get("transcript_storage_path") or f"chats/{chat_id}/uploads/{filename}"
        try:
            bucket.blob(transcript_path).delete()
            print(f"🗑️  [Recording] Deleted transcript at {transcript_path}")
        except Exception as e:
            print(f"⚠️ [Recording] Failed to delete transcript: {e}")

        # 2b. Delete per-file vectorstore folder
        try:
            prefix = f"FileVectorStore/{chat_id}/{filename}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            for b in blobs:
                b.delete()
            if blobs:
                print(f"🗑️  [Recording] Deleted per-file vectorstore at {prefix}")
        except Exception as e:
            print(f"⚠️ [Recording] Failed to delete per-file vectorstore: {e}")

        # 2c. Remove the recording's vectors from the chat's combined
        # vectorstore. If nothing remains, drop the combined entirely.
        try:
            from services.vectorstore_manager import vectorstore_manager
            combined = await vectorstore_manager.load_combined_vectorstore_from_firebase(chat_id)
            if combined is not None:
                ids_to_delete = [
                    doc_id for doc_id, doc in combined.docstore._dict.items()
                    if (doc.metadata or {}).get("source") == filename
                ]
                if ids_to_delete:
                    combined.delete(ids_to_delete)
                    print(f"🗑️  [Recording] Removed {len(ids_to_delete)} vectors from combined")

                # Re-upload if anything's left, otherwise nuke the combined folder.
                if combined.docstore._dict:
                    await vectorstore_manager.upload_combined_vectorstore_to_firebase(chat_id, combined)
                else:
                    combined_blobs = list(bucket.list_blobs(prefix=f"vectorstores/{chat_id}/"))
                    for b in combined_blobs:
                        b.delete()
                    print(f"🗑️  [Recording] Combined vectorstore now empty — deleted")
        except Exception as e:
            print(f"⚠️ [Recording] Failed to update combined vectorstore: {e}")

        # 2d. Evict the in-memory session so the next interaction reloads
        # the updated vectorstore + file list from Storage.
        try:
            from main import ACTIVE_SESSIONS  # local import to avoid a circular dep
            if chat_id in ACTIVE_SESSIONS:
                ACTIVE_SESSIONS.pop(chat_id, None)
                print(f"🗑️  [Recording] Evicted in-memory session for {chat_id}")
        except Exception as e:
            print(f"⚠️ [Recording] Failed to evict session: {e}")

        # 2e. If this chat was created just for this recording and now has
        # no other uploads, the chat itself is dead weight — delete it too.
        try:
            chat_ref = _db().collection(CHATS_COLLECTION).document(chat_id)
            chat_doc = chat_ref.get()
            if chat_doc.exists:
                chat_data = chat_doc.to_dict() or {}
                if chat_data.get("createdFromRecording"):
                    remaining = list(bucket.list_blobs(prefix=f"chats/{chat_id}/uploads/"))
                    if not remaining:
                        chat_ref.delete()
                        # Also drop the (now-empty) combined vectorstore folder
                        for b in bucket.list_blobs(prefix=f"vectorstores/{chat_id}/"):
                            b.delete()
                        print(f"🗑️  [Recording] Deleted orphan chat {chat_id}")
        except Exception as e:
            print(f"⚠️ [Recording] Failed to delete orphan chat: {e}")

    # 3. Mark recording cancelled in Firestore
    _db().collection(RECORDINGS_COLLECTION).document(recording_id).update({
        "status": STATUS_CANCELLED,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })
    print(f"🎙️ [Recording] Cancelled {recording_id} (finalized={is_finalized})")
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
