"""Uploads the browser has already put in Firebase Storage (2026-10-22).

WHY THIS EXISTS

An upload used to be one multipart POST to this service. Cloud Run's front
end refuses any request body over 32 MiB with a 413 that carries no CORS
header, so the browser saw a bare network error a few seconds in: 9 of the
last 20 logged failures were over 32 MB, most of them a single slide deck.
The other common failure was a phone locked mid-send, which kills a fetch
but only pauses a Firebase resumable upload.

So the browser now writes each file to `chats/{chat_id}/uploads/{name}` (the
same place the background task used to copy it to) and sends only the names.
This module reads them back. Files sent the old way still work.

MAX_STAGED_FILE_BYTES mirrors MAX_UPLOAD_FILE_BYTES in the frontend's
src/Services/storageUpload.js. The frontend refuses a bigger file before
uploading it; this is the server's copy of the same rule.
"""
import asyncio
import json

MAX_STAGED_FILE_BYTES = 150 * 1024 * 1024
MAX_STAGED_FILES = 30


def staged_names(raw):
    """Parse the `storage_files` form field into safe file names.

    Names become Storage paths under the chat's own uploads folder, so anything
    that could step outside it (a slash, a backslash, "..") is refused rather
    than cleaned up. Returns None when the field is absent or malformed.
    """
    if not raw:
        return None
    try:
        names = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(names, list) or not 0 < len(names) <= MAX_STAGED_FILES:
        return None
    clean = []
    for name in names:
        if (not isinstance(name, str) or not name.strip() or len(name) > 255
                or "/" in name or "\\" in name or name in (".", "..") or "\x00" in name):
            return None
        if name not in clean:
            clean.append(name)
    return clean


def _read_one(bucket, chat_id, name):
    blob = bucket.get_blob(f"chats/{chat_id}/uploads/{name}")
    if blob is None:
        return {"filename": name, "error": "File was not found in storage"}
    if (blob.size or 0) > MAX_STAGED_FILE_BYTES:
        return {"filename": name, "error": "File is too large", "code": "file_too_large"}
    data = blob.download_as_bytes()
    return {"filename": name, "content_type": blob.content_type, "bytes": data,
            "size": len(data), "staged": True}


async def read_staged_files(bucket, chat_id, names):
    """Download each staged file in parallel, in the same shape the multipart
    path builds (`filename`, `bytes`, `size`, or `error`). `staged: True` tells
    the background task the file is already in Storage."""
    async def one(name):
        try:
            return await asyncio.to_thread(_read_one, bucket, chat_id, name)
        except Exception as e:
            return {"filename": name, "error": str(e)}
    return await asyncio.gather(*(one(n) for n in names))
