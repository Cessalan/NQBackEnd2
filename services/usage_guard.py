"""
usage_guard.py
Server-side enforcement of the free-tier generation quota (monetization).

The frontend already gates generations in the UI (ragfrontend
src/Services/UsageService.js), but that gate runs in the browser and is
bypassable. This module re-checks the SAME Firestore state on the server so a
direct API/WebSocket caller can't skip the limit.

Model (must stay in sync with UsageService.js — change both together):
  - users/{uid}.usage = { tier: 'free'|'pro', windowStart: ms epoch, count }
  - Free tier: FREE_LIMIT question-units per rolling WINDOW_MS window.
  - Window elapsed => bucket is considered empty (the client resets it on its
    next write; we only need to *read* correctly here).
  - Pro tier: unlimited.

Who is asking? Generation requests carry no auth, only chat_id, so the user is
resolved via chats/{chat_id}.userId. An attacker can only ever use a chat_id
that maps to some real user, whose own quota then applies. Unknown chat_ids
fail open (see below).

Charging still happens client-side (consumeGeneration). This guard is
VALIDATION ONLY — it never writes. That keeps a single writer for the usage
map and avoids double-charging; the residual gap (a scripted caller who never
runs the client and therefore never increments count) is accepted for now.

Fail-open philosophy (matches the client): any lookup error, missing doc, or
malformed data ALLOWS the request. Never block a paying or working user
because of our own read failure.
"""

import time
from firebase_admin import firestore

# ── Tunables — keep identical to FREE_LIMIT / WINDOW_MS in UsageService.js ──
FREE_LIMIT = 30                      # question-units per window (free tier)
WINDOW_MS = 3 * 60 * 60 * 1000       # rolling window length (3 hours, in ms)

# User-facing rejection copy (the frontend may show its own localized copy;
# this is the fallback that lands in the error bubble).
QUOTA_MESSAGE = (
    "You've used all your free questions for this 3-hour window. "
    "Upgrade to Pro for unlimited practice, or wait for the window to reset."
)

# chat_id -> uid. Chat ownership never changes, so entries never go stale.
# Cleared wholesale if it somehow grows unbounded (long-lived instance).
_uid_cache: dict = {}
_UID_CACHE_MAX = 10000


def _resolve_uid(chat_id: str):
    """Look up the owner of a chat (chats/{chat_id}.userId), with caching."""
    if not chat_id:
        return None
    cached = _uid_cache.get(chat_id)
    if cached:
        return cached
    snap = firestore.client().collection("chats").document(chat_id).get()
    uid = (snap.to_dict() or {}).get("userId") if snap.exists else None
    if uid:
        if len(_uid_cache) >= _UID_CACHE_MAX:
            _uid_cache.clear()
        _uid_cache[chat_id] = uid
    return uid


def check_quota(chat_id: str) -> dict:
    """
    Read-only quota check for the user who owns `chat_id`.

    Returns {"allowed": bool, "reason": str|None, "tier": str|None}.
    reason is "quota_exceeded" when blocked; other reasons are diagnostic
    (fail-open paths) and always come with allowed=True.
    """
    try:
        uid = _resolve_uid(chat_id)
        if not uid:
            # Chat not in Firestore (or has no owner) — likely a brand-new
            # chat racing its own creation write. Fail open.
            return {"allowed": True, "reason": "no_user", "tier": None}

        snap = firestore.client().collection("users").document(uid).get()
        usage = ((snap.to_dict() or {}).get("usage") or {}) if snap.exists else {}

        if usage.get("tier") == "pro":
            return {"allowed": True, "reason": None, "tier": "pro"}

        now_ms = int(time.time() * 1000)
        window_start = usage.get("windowStart")
        count = usage.get("count")
        if not isinstance(window_start, (int, float)):
            window_start = 0
        if not isinstance(count, (int, float)):
            count = 0

        # Window elapsed (or never started) => fresh bucket.
        if not window_start or now_ms - window_start >= WINDOW_MS:
            count = 0

        if count < FREE_LIMIT:
            return {"allowed": True, "reason": None, "tier": "free"}
        return {"allowed": False, "reason": "quota_exceeded", "tier": "free"}

    except Exception as e:
        print(f"⚠️ usage_guard: quota check failed for chat {chat_id}: {e} — failing open")
        return {"allowed": True, "reason": "check_failed", "tier": None}
