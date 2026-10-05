"""
email_sender.py
Outbound email: the return channel the product has never had.

WHY THIS EXISTS

Measured 2026-09-06: of 288 students active in the 35 days since checkout went
live, 232 (81%) used the product on exactly ONE day. Conversion among students
who came back at least once is 16.4%; among one-day students it is 0.9%. The
single biggest constraint on revenue is that people do not return — and until
now there was no way to say one word to any of them. StudyReminderService.js on
the frontend is browser-local by its own admission: it can only fire while the
tab is open.

1,956 of 2,006 accounts (97.5%) carry a usable address, so the audience exists.

WHAT THIS MODULE REFUSES TO DO

Sending mail to ~2,000 people is easy to do once and impossible to undo. 78% of
these addresses are Gmail, and Gmail's judgement about a new sending domain is
formed in the first few days and is expensive to reverse. So the defaults here
are deliberately timid, and every one of these is a guard against a mistake
that cannot be walked back:

  - DISABLED BY DEFAULT. Nothing leaves the process unless EMAIL_ENABLED is
    explicitly "true". Import it, call it, run the cron — with the flag unset
    you get a full dry run with the exact payload logged. The failure mode of
    a misconfigured deploy is silence, not 2,000 emails.

  - SUPPRESSION IS CHECKED EVEN IN DRY RUN, so a dry run tells you the truth
    about who would actually be contacted.

  - EVERY SEND NEEDS AN IDEMPOTENCY KEY. A cron that reruns — because Cloud
    Scheduler retried, or someone redeployed — must not mail the same student
    twice. The key is the document id in `emailLog`, so the write itself is
    the lock.

  - DAILY CAP. Domain warm-up is not optional on a cold domain. The cap is
    read from the environment so it can be raised over days without a deploy.

CONSENT

CAN-SPAM requires a working opt-out and a physical postal address in every
commercial message. Some of this list is Canadian/French (CASL: implied consent
runs ~6 months from signup, so the oldest accounts are outside it). Every
message therefore carries a one-click unsubscribe that needs no login, backed
by an HMAC token so the link cannot be forged or enumerated.

`transactional=True` skips the marketing opt-out check but STILL honours a
global unsubscribe. Use it only for things a student asked for.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone

import httpx
from firebase_admin import firestore

RESEND_ENDPOINT = "https://api.resend.com/emails"

# Firestore collections
LOG_COLLECTION = "emailLog"
USERS = "users"


def _env(name, default=""):
    return (os.getenv(name) or default).strip()


def is_enabled():
    """Master switch. Anything other than an explicit 'true' means dry run."""
    return _env("EMAIL_ENABLED").lower() == "true"


def _api_key():
    return _env("RESEND_API_KEY")


def _from_address():
    # Send from a SUBDOMAIN, never the root domain: a deliverability problem on
    # marketing mail must not be able to poison password resets or receipts.
    return _env("EMAIL_FROM", "NurseQuizAI <hello@mail.nursequizai.com>")


def _app_url():
    """Where the STUDENT goes — the React app. Resume links, CTAs."""
    return _env("APP_URL", "https://docai-efb03.web.app").rstrip("/")


def _api_url():
    """
    Where OUR OWN routes live — this FastAPI service. The unsubscribe link.

    THIS IS NOT APP_URL, and conflating them silently breaks the one thing that
    has to work before anything may be mailed at all.

    /api/email/unsubscribe is a route on this Cloud Run service. APP_URL points
    at Firebase Hosting, whose only rewrite is `** -> /index.html` (checked in
    firebase.json 2026-09-12) — there is no /api/** proxy to Cloud Run. So an
    unsubscribe link built on APP_URL returns the React shell: a student
    clicking "Unsubscribe" would get the app's loading screen, stay subscribed,
    and have no way to tell us to stop except a spam complaint. The same URL
    goes in the List-Unsubscribe header that Gmail's one-click control calls,
    so Gmail would be hitting the SPA too.

    Either add an /api/** rewrite to Firebase Hosting or leave this pointing
    straight at the service. Pointing straight at it is fewer moving parts and
    what the default does.
    """
    return _env("EMAIL_LINK_BASE",
                "https://ragfastapi-1075876064685.europe-west1.run.app").rstrip("/")


def _postal_address():
    # Required by CAN-SPAM in every commercial message. Set it before the
    # first marketing send; the footer says so loudly if it is missing.
    return _env("EMAIL_POSTAL_ADDRESS", "")


def daily_cap():
    try:
        return int(_env("EMAIL_DAILY_CAP", "50"))
    except ValueError:
        return 50


def configuration_issues():
    """Actionable setup failures, without exposing credentials."""
    issues = []
    if not _api_key():
        issues.append('Set RESEND_API_KEY in the backend environment.')
    if not _postal_address():
        issues.append('Set EMAIL_POSTAL_ADDRESS for the email footer.')
    if _secret() == 'dev-only-insecure':
        issues.append('Set EMAIL_TOKEN_SECRET to secure unsubscribe links.')
    if daily_cap() <= 0:
        issues.append('Set EMAIL_DAILY_CAP to a positive number.')
    return issues


# ── Unsubscribe tokens ────────────────────────────────────────────────────
# Signed with a dedicated secret so a link cannot be forged and uids cannot be
# enumerated by walking the URL space. Falls back to the Stripe webhook secret
# only so a missing config does not silently produce unsigned links.

def _secret():
    return _env("EMAIL_TOKEN_SECRET") or _env("STRIPE_WEBHOOK_SECRET") or "dev-only-insecure"


def make_unsubscribe_token(uid: str) -> str:
    payload = base64.urlsafe_b64encode(uid.encode()).decode().rstrip("=")
    sig = hmac.new(_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{sig}"


def verify_unsubscribe_token(token: str):
    """Return the uid, or None. Never raises on malformed input."""
    try:
        payload, sig = str(token).split(".", 1)
        expected = hmac.new(_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
        # compare_digest: token checking must not leak timing information.
        if not hmac.compare_digest(sig, expected):
            return None
        pad = "=" * (-len(payload) % 4)
        return base64.urlsafe_b64decode(payload + pad).decode()
    except Exception:
        return None


def unsubscribe_url(uid: str) -> str:
    # _api_url, not _app_url — see the docstring on _api_url.
    return f"{_api_url()}/api/email/unsubscribe?token={make_unsubscribe_token(uid)}"


# ── Consent ───────────────────────────────────────────────────────────────

def is_suppressed(db, uid: str, transactional: bool = False):
    """
    Why this student must not be mailed, or None if they may be.

    Returns a reason string so a dry run can report exactly who was skipped
    and why, rather than just a smaller number.
    """
    try:
        snap = db.collection(USERS).document(uid).get()
        if not snap.exists:
            return "no_user_doc"
        d = snap.to_dict() or {}

        prefs = d.get("emailPrefs") or {}
        if prefs.get("unsubscribedAt"):
            return "unsubscribed"
        # Marketing opt-out never blocks a message the student asked for, but a
        # global unsubscribe (above) blocks everything.
        if not transactional and prefs.get("marketing") is False:
            return "marketing_opt_out"

        if d.get("hardBounced"):
            return "hard_bounced"
        if not (d.get("email") or "").strip():
            return "no_address"
        return None
    except Exception as e:
        # Fail CLOSED. An unreadable consent record is not permission.
        return f"suppression_check_failed:{e}"


def set_unsubscribed(db, uid: str):
    db.collection(USERS).document(uid).set(
        {"emailPrefs": {"marketing": False, "unsubscribedAt": firestore.SERVER_TIMESTAMP}},
        merge=True,
    )


# ── Sending ───────────────────────────────────────────────────────────────

def _already_sent(db, key: str) -> bool:
    """
    Has this exact message already gone out?

    Only a row with status 'sent' counts. A 'failed' row must stay retryable,
    and a 'dry_run' row is written under a prefixed key (see send_email) so it
    can never occupy the real one — an afternoon of testing must not silently
    mark the whole list as already-mailed and make the first real send a no-op.
    """
    try:
        snap = db.collection(LOG_COLLECTION).document(key).get()
        if not snap.exists:
            return False
        return ((snap.to_dict() or {}).get("status")) == "sent"
    except Exception:
        # Fail CLOSED: if we cannot prove we have NOT sent it, do not send.
        return True


def _record(db, key: str, uid: str, campaign: str, to: str, status: str, extra=None):
    try:
        db.collection(LOG_COLLECTION).document(key).set({
            "uid": uid,
            "campaign": campaign,
            "to": to,
            "status": status,
            "sentAt": firestore.SERVER_TIMESTAMP,
            "clientTs": int(time.time() * 1000),
            **(extra or {}),
        })
    except Exception as e:
        print(f"⚠️  emailLog write failed ({key}): {e}")


# A count we could not read must never read as "plenty of budget left". Any
# number above every plausible cap makes the cap trip instead.
_CAP_UNKNOWN = 10 ** 9


def sent_today(db, campaign: str = None) -> int:
    """
    How many have gone out since UTC midnight — the warm-up budget.

    THE FILTER IS DELIBERATELY ONE FIELD, and that is the whole point of this
    function's history. It used to range-filter `sentAt` AND equality-filter
    `status`, which Firestore cannot serve without a composite index that does
    not exist on this project. The query raised FailedPrecondition every time,
    and the bare `except: return 0` turned that into "nothing sent today" — so
    `used >= cap` was `0 >= 50` forever and the daily cap NEVER TRIPPED.

    Measured against production 2026-09-12: two rows written with status "sent"
    and sentAt=now returned 0. The one guard standing between a cold domain and
    a 2,000-message first run was inert, on a list we cannot re-acquire.

    So: range-filter on `sentAt` alone, which needs no composite index, and
    match status/campaign here. Today's log is at most a few hundred rows.

    Fails CLOSED. If we cannot count what has already gone out, we cannot know
    there is budget left, so this returns a number that exhausts any cap.
    """
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        q = db.collection(LOG_COLLECTION).where(
            filter=firestore.FieldFilter("sentAt", ">=", start))
        n = 0
        for snap in q.stream():
            d = snap.to_dict() or {}
            if d.get("status") != "sent":
                continue
            if campaign and d.get("campaign") != campaign:
                continue
            n += 1
        return n
    except Exception as e:
        print(f"⚠️  sent_today failed, treating the daily cap as exhausted: {e}")
        return _CAP_UNKNOWN


def remaining_today(db) -> int:
    """
    How many more may go out today.

    Campaign runners check this ONCE before their loop instead of letting
    send_email reject each candidate in turn: a 2,000-candidate announcement
    against a 50/day cap would otherwise rescan today's log 2,000 times to
    refuse 1,950 of them.
    """
    return max(0, daily_cap() - sent_today(db))


def already_sent_uids(db, campaign: str):
    """
    Which uids have already received this campaign. None if we cannot tell.

    A campaign larger than the daily cap needs this BEFORE the limit is
    applied, not after. With 2,000 recipients and a 50/day cap, a selector that
    simply takes the first 50 each day hands send_email the same 50 people it
    already mailed; all 50 are skipped as already_sent and day two delivers
    nothing. The campaign stalls at 50 recipients forever and looks like it ran.

    Equality on `campaign` alone needs no composite index — verified against
    production 2026-09-12, where adding `status` to it did.

    Returns None rather than an empty set on failure, because those two mean
    opposite things: an empty set says "nobody has been mailed yet, send to
    everyone", which is the one wrong answer that cannot be taken back. Callers
    must treat None as "abort the run".
    """
    try:
        out = set()
        q = db.collection(LOG_COLLECTION).where(
            filter=firestore.FieldFilter("campaign", "==", campaign))
        for snap in q.stream():
            d = snap.to_dict() or {}
            if d.get("status") == "sent" and d.get("uid"):
                out.add(d["uid"])
        return out
    except Exception as e:
        print(f"⚠️  already_sent_uids({campaign}) failed: {e}")
        return None


def _footer(uid: str) -> str:
    addr = _postal_address()
    addr_html = f"<div>{addr}</div>" if addr else (
        "<div style='color:#b00'>[SET EMAIL_POSTAL_ADDRESS — required by CAN-SPAM]</div>"
    )
    return (
        "<hr style='border:none;border-top:1px solid #eee;margin:28px 0 14px'>"
        "<div style='font:12px -apple-system,Segoe UI,sans-serif;color:#888;line-height:1.6'>"
        f"<a href='{unsubscribe_url(uid)}' style='color:#888'>Unsubscribe</a>"
        " · You're receiving this because you created a NurseQuizAI account."
        f"{addr_html}</div>"
    )


def send_email(db, uid: str, to: str, subject: str, html: str, campaign: str,
               idempotency_key: str = None, transactional: bool = False,
               ignore_cap: bool = False):
    """
    Send one email, or explain why it was not sent.

    Returns {"status": ..., "reason": ...}. Never raises: a failed send is a
    row in a log, not an exception that takes down the caller's loop.

    status is one of: sent | dry_run | skipped | failed
    """
    real_key = idempotency_key or f"{uid}_{campaign}_{datetime.now(timezone.utc):%Y%m%d}"
    # Dry runs log under their own namespace so reviewing a campaign can never
    # consume the key the real send needs.
    key = real_key if is_enabled() else f"dryrun_{real_key}"

    if is_enabled() and configuration_issues():
        return {"status": "failed", "reason": ' '.join(configuration_issues()), "key": key}

    reason = is_suppressed(db, uid, transactional=transactional)
    if reason:
        return {"status": "skipped", "reason": reason, "key": key}

    if _already_sent(db, key):
        return {"status": "skipped", "reason": "already_sent", "key": key}

    if not ignore_cap and not transactional:
        used = sent_today(db)
        cap = daily_cap()
        if used >= cap:
            return {"status": "skipped", "reason": f"daily_cap_reached:{used}/{cap}", "key": key}

    body = html + _footer(uid)

    if not is_enabled():
        # The whole payload, so a dry run is genuinely reviewable.
        print(f"📧 [DRY RUN] to={to} campaign={campaign} subject={subject!r} key={real_key}")
        _record(db, key, uid, campaign, to, "dry_run",
                {"subject": subject, "wouldUseKey": real_key})
        return {"status": "dry_run", "reason": "EMAIL_ENABLED not true", "key": real_key}

    api_key = _api_key()
    if not api_key:
        return {"status": "failed", "reason": "RESEND_API_KEY missing", "key": key}

    try:
        r = httpx.post(
            RESEND_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            content=json.dumps({
                "from": _from_address(),
                "to": [to],
                "subject": subject,
                "html": body,
                # Gmail surfaces this as a native unsubscribe control, which is
                # the difference between an opt-out and a spam complaint.
                "headers": {
                    "List-Unsubscribe": f"<{unsubscribe_url(uid)}>",
                    "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                },
            }),
            timeout=20.0,
        )
        if r.status_code >= 300:
            _record(db, key, uid, campaign, to, "failed",
                    {"error": r.text[:400], "code": r.status_code})
            return {"status": "failed", "reason": f"{r.status_code}: {r.text[:200]}", "key": key}

        msg_id = (r.json() or {}).get("id")
        _record(db, key, uid, campaign, to, "sent", {"providerId": msg_id, "subject": subject})
        return {"status": "sent", "id": msg_id, "key": key}

    except Exception as e:
        _record(db, key, uid, campaign, to, "failed", {"error": str(e)[:400]})
        return {"status": "failed", "reason": str(e), "key": key}


def preflight():
    """Config readiness — call before the first real send."""
    return {
        "enabled": is_enabled(),
        "api_key_present": bool(_api_key()),
        "from": _from_address(),
        "app_url": _app_url(),
        # Where unsubscribe links point. If this is the hosting domain rather
        # than the API service, every opt-out in every message is dead.
        "link_base": _api_url(),
        "unsubscribe_example": unsubscribe_url("preflight-check"),
        "postal_address_set": bool(_postal_address()),
        "daily_cap": daily_cap(),
        "token_secret_dedicated": bool(_env("EMAIL_TOKEN_SECRET")),
    }
