"""
email_campaigns.py
Who gets mailed, and with which numbers.

This is the half of automated email that goes wrong. Sending is a solved
problem; deciding who deserves a message — and being certain the numbers in it
are true for that person — is not.

THE RULE THIS FILE ENFORCES

    No student is mailed unless we can state something TRUE and SPECIFIC
    about her.

The winback message says "you're averaging 83%, but 40% on Fluid &
Electrolytes." If those numbers are not computable for a given student, she is
skipped — not sent a version with the placeholders left in, and not sent a
generic "come back!" instead. A generic re-engagement mail from an unfamiliar
domain is the exact shape spam filters and people discard, and it burns a
send on the one list we cannot re-acquire.

So `select_*` returns candidates already carrying their own personalisation,
and anyone whose evidence is too thin never becomes a candidate at all.

WHY DORMANCY IS A BAND, NOT A THRESHOLD

Winback targets students dormant between MIN and MAX days:

  - Too recent (< ~3 days) and it is nagging someone who has not left.
  - Too old (> ~45 days) and she has forgotten signing up, which produces
    spam complaints rather than returns. It also drifts toward the edge of
    CASL's implied-consent window (~6 months from signup) for a Canadian
    list, and the oldest addresses are the least likely to still be valid —
    bounces on a cold domain are expensive.

COST

Selection reads chat metadata for every chat, then study maps only for the
handful of users who survive filtering. Full study documents are large; pulling
5,000 of them to find 40 recipients would be slow and pointless.
"""

import os
from datetime import datetime, timedelta, timezone

from services import email_sender, email_templates

MIN_EVIDENCE_QUESTIONS = 8      # below this, "your average" is noise
MIN_TOPIC_QUESTIONS = 4         # below this, a weak topic is not a finding
WEAK_MAX_PCT = 70               # only call a topic weak if it actually is

# The message contrasts her average with her worst subject. If those two
# numbers are close, there is no contrast — "you average 70%, and 70% on X"
# reads as a broken mail merge. Require a real gap or send nothing.
MIN_GAP_POINTS = 10
MIN_DISTINCT_TOPICS = 2

DORMANT_MIN_DAYS = 3
DORMANT_MAX_DAYS = 45
EXAM_WINDOW_DAYS = 7

# Not every key in studyPerformance is a SUBJECT. The adaptive layer writes
# node labels into the same map ("Review", "Comprehensive Final Test",
# "Testing a theory: prioritization"), and "Review is costing you marks" is
# not a sentence to send anybody. These are the shapes to refuse.
_JUNK_TOPIC_EXACT = {
    "review", "quick review", "comprehensive final test", "final test",
    "document overview", "overview", "mini-test", "general", "untitled",
    "practice", "drill", "test", "quiz", "summary",
}
_JUNK_TOPIC_PREFIXES = (
    "testing a theory", "targeted practice", "focused drill", "harder",
    "fix this", "challenge", "retake", "adaptive", "warm-up", "warmup",
)


def _is_real_subject(name):
    """A topic a student would recognise as a subject, not a node label."""
    n = str(name or "").strip().lower()
    if len(n) < 4:
        return False
    if n in _JUNK_TOPIC_EXACT:
        return False
    if any(n.startswith(p) for p in _JUNK_TOPIC_PREFIXES):
        return False
    # "Foo: bar" from the adaptive layer — a subject rarely has a colon.
    if ":" in n and any(n.split(":")[0].strip().startswith(p) for p in _JUNK_TOPIC_PREFIXES):
        return False
    return True


def _app_url():
    return (os.getenv("APP_URL") or "https://docai-efb03.web.app").rstrip("/")


def _aggregate_performance(db, uid):
    """
    Her record across every session: overall accuracy and weakest topic.

    Returns None when there is not enough to say anything true.
    """
    correct = total = 0
    per_topic = {}
    try:
        for sp in db.collection("users").document(uid).collection("studyPerformance").stream():
            for tname, tp in ((sp.to_dict() or {}).get("topics") or {}).items():
                c = tp.get("questionsCorrect") or 0
                t = tp.get("questionsTotal") or 0
                if not t:
                    continue
                correct += c
                total += t
                # Strip the node-kind suffix ("X - Mini-Test") so one subject
                # is one bucket, matching baseTopicName on the frontend.
                base = str(tname).split(" - ")[0].strip()
                acc = per_topic.setdefault(base, [0, 0])
                acc[0] += c
                acc[1] += t
    except Exception:
        return None

    if total < MIN_EVIDENCE_QUESTIONS:
        return None

    # Only real subjects are eligible to be named in the message.
    named = {n: v for n, v in per_topic.items() if _is_real_subject(n)}
    if len(named) < MIN_DISTINCT_TOPICS:
        # With one subject there is no "your average vs this topic" to draw.
        return None

    weak = None
    for name, (c, t) in named.items():
        if t < MIN_TOPIC_QUESTIONS:
            continue
        pct = round(100.0 * c / t)
        if pct > WEAK_MAX_PCT:
            continue
        if weak is None or pct < weak[1]:
            weak = (name, pct)

    if not weak:
        # No weak subject. Good news, and not a reason to email — there is no
        # honest gap to point at.
        return None

    overall = round(100.0 * correct / total)
    if overall - weak[1] < MIN_GAP_POINTS:
        # The contrast the email is built on does not exist for her.
        return None

    return {
        "overall_pct": overall,
        "weak_topic": weak[0],
        "weak_pct": weak[1],
        "questions": total,
    }


def _unfinished_plan(db, uid, chat_ids):
    """The student's most recent plan with steps left: (chatId, steps_left)."""
    best = None
    for cid in chat_ids:
        try:
            d = db.collection("chats").document(cid).get().to_dict() or {}
        except Exception:
            continue
        study = d.get("study") or {}
        nodes = (study.get("path") or {}).get("nodes") or []
        if not nodes:
            continue
        left = sum(1 for n in nodes
                   if isinstance(n, dict)
                   and n.get("type") not in ("banner",)
                   and n.get("status") != "completed")
        if left <= 0:
            continue
        ts = d.get("updatedAt")
        if best is None or (ts and best[2] and ts > best[2]) or best[2] is None:
            best = (cid, left, ts)
    return (best[0], best[1]) if best else (None, 0)


def _scan_users_and_chats(db):
    """One pass each over users and chat metadata. Study maps are read later."""
    users = {}
    for d in db.collection("users").select(["email", "displayName", "emailPrefs",
                                            "onboarding", "usage"]).stream():
        x = d.to_dict() or {}
        users[d.id] = {
            "email": (x.get("email") or "").strip(),
            "name": (x.get("displayName") or "").split(" ")[0] or None,
            "tier": ((x.get("usage") or {}).get("tier")) or "free",
            "examDate": (x.get("onboarding") or {}).get("examDate"),
        }

    chats = {}
    last_seen = {}
    for d in db.collection("chats").select(["userId", "updatedAt", "isStudySession"]).stream():
        x = d.to_dict() or {}
        uid = x.get("userId")
        if not uid:
            continue
        ts = x.get("updatedAt")
        if ts and (uid not in last_seen or ts > last_seen[uid]):
            last_seen[uid] = ts
        if x.get("isStudySession"):
            chats.setdefault(uid, []).append(d.id)
    return users, chats, last_seen


def select_winback(db, limit=50, verbose=False):
    """Dormant students for whom we can state a real, specific gap."""
    users, study_chats, last_seen = _scan_users_and_chats(db)
    now = datetime.now(timezone.utc)
    lo, hi = now - timedelta(days=DORMANT_MAX_DAYS), now - timedelta(days=DORMANT_MIN_DAYS)

    out, skipped = [], {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for uid, u in users.items():
        if not u["email"]:
            skip("no_email"); continue
        if u["tier"] == "pro":
            skip("already_pro"); continue           # nothing to sell here
        seen = last_seen.get(uid)
        if not seen:
            skip("never_active"); continue
        if not (lo <= seen <= hi):
            skip("outside_dormancy_window"); continue
        if uid not in study_chats:
            skip("no_study_plan"); continue

        reason = email_sender.is_suppressed(db, uid)
        if reason:
            skip(reason); continue

        perf = _aggregate_performance(db, uid)
        if not perf:
            # The message would have nothing true to say. Do not send one.
            skip("insufficient_evidence"); continue

        chat_id, steps_left = _unfinished_plan(db, uid, study_chats[uid])
        if not chat_id:
            skip("plan_already_finished"); continue

        out.append({
            "uid": uid, "to": u["email"], "name": u["name"],
            "resume_url": f"{_app_url()}/c/{chat_id}",
            "steps_left": steps_left, **perf,
        })
        if len(out) >= limit:
            break

    if verbose:
        print("  selection skips:")
        for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>5}  {k}")
    return out


def run_winback(db, limit=50, verbose=True):
    """Select, render and send. Honours dry-run, suppression, cap, idempotency."""
    postal = os.getenv("EMAIL_POSTAL_ADDRESS", "")
    candidates = select_winback(db, limit=limit, verbose=verbose)

    results = {"selected": len(candidates), "sent": 0, "dry_run": 0,
               "skipped": 0, "failed": 0, "detail": []}

    for c in candidates:
        msg = email_templates.winback_gap(
            first_name=c["name"], weak_topic=c["weak_topic"],
            overall_pct=c["overall_pct"], weak_pct=c["weak_pct"],
            steps_left=c["steps_left"], resume_url=c["resume_url"],
            unsubscribe_url=email_sender.unsubscribe_url(c["uid"]),
            postal_address=postal,
        )
        r = email_sender.send_email(
            db, uid=c["uid"], to=c["to"], subject=msg["subject"],
            html=msg["html"], campaign="winback_gap",
            # One per student per campaign, ever — not per day. A winback that
            # repeats monthly is a newsletter nobody subscribed to.
            idempotency_key=f"{c['uid']}_winback_gap",
        )
        results[r["status"]] = results.get(r["status"], 0) + 1
        results["detail"].append({"uid": c["uid"][:10], "topic": c["weak_topic"],
                                  "status": r["status"], "reason": r.get("reason")})
        if verbose:
            print(f"  {r['status']:<8} {c['to'][:34]:<34} "
                  f"{c['weak_pct']}% {c['weak_topic'][:26]}  {r.get('reason','')}")
    return results


CAMPAIGN_RUNNERS = {"winback_gap": run_winback}
