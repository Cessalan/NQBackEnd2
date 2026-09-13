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

THE CAMPAIGNS

    winback_gap      her average vs her worst subject. Needs answered
                     questions, so it reaches 28 of 348 dormant students.
    plan_unstarted   plan built, never opened. The other 228.
    exam_countdown   an exam inside 7 days and a plan with steps left.
    announcement     the same authored words to an audience. Content and the
                     approval gate live in email_announcements.py.

Only `announcement` takes a slug; the rest take just a limit.

COST

Selection reads chat metadata for every chat, then study maps only for the
handful of users who survive filtering. Full study documents are large; pulling
5,000 of them to find 40 recipients would be slow and pointless.
"""

import os
from datetime import datetime, timedelta, timezone

from services import email_announcements, email_sender, email_templates

# The planner emits "section_banner" pseudo-nodes as section headers. They are
# not steps, and the frontend has always known it: isRealNode in
# StudyMode/firstBlock.js is exactly `node.type !== 'section_banner'`.
#
# This module used to filter `type not in ("banner",)` — a type the planner
# does not emit. Every section header therefore counted as a step, and
# "your plan has 9 steps left" overstated the work by however many headers the
# plan carried. A number in an email is a promise, and this one was wrong in
# the worst direction: it made the plan look LONGER, to precisely the students
# who had not started it because it already looked long.
NON_STEP_NODE_TYPES = ("section_banner", "banner")

# Mirrors NODE_MINUTES in ragfrontend/src/Components/StudyMode/planFormatting.js.
# Mirrored rather than guessed so the "about 4 minutes" in an email is the same
# number the app shows her when she arrives. If they drift, the mail promises a
# length the product then contradicts on the first screen.
NODE_MINUTES = {
    "lesson": 5, "quiz": 4, "flashcard": 3,
    "audio": 6, "mindmap": 5, "review": 2, "exam": 15,
}
NODE_MINUTES_DEFAULT = 4        # planFormatting.js: NODE_MINUTES[type] || 4

# What to call a step in a sentence. "a quick check on Hepatitis" reads like
# the product; "a quiz node on Hepatitis" reads like the database.
NODE_KIND_LABEL = {
    "quiz": "quick check", "lesson": "lesson", "flashcard": "flashcard set",
    "audio": "audio lesson", "mindmap": "concept map", "exam": "practice exam",
    "review": "review",
}

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
    # Added 2026-09-12 from a scan of 90 real plans. node.tags[0] is usually
    # the clean subject, but on some plans it is the STRUCTURE instead —
    # "lesson", "assessment", "core". Those pass every other check here (long
    # enough, no junk prefix) and would have produced the subject line
    # "Your plan starts with lesson".
    "lesson", "lessons", "assessment", "assessments", "core", "core topics",
    "pre-test", "pretest", "quick check", "topic", "topics", "section",
    "module", "unit", "chapter", "document", "notes", "study", "exam",
    "final", "midterm", "material", "content", "intro", "introduction",
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


def _real_nodes(study):
    """The steps in a plan — section headers are not steps. See NON_STEP_NODE_TYPES."""
    return [n for n in ((study or {}).get("path") or {}).get("nodes") or []
            if isinstance(n, dict) and n.get("type") not in NON_STEP_NODE_TYPES]


def _node_minutes(nodes):
    return sum(NODE_MINUTES.get(n.get("type"), NODE_MINUTES_DEFAULT) for n in nodes)


def _node_topic(node):
    """
    The subject this step is about, or None if it cannot be named honestly.

    Two sources, checked in order, because production carries both shapes:
    `tags[0]` is normally the clean subject ("Electrolyte Disorders") while
    `label` is the decorated one ("Electrolyte Disorders and Their Clinical
    Manifestations - Quick Check"). On some plans tags[0] is structural
    instead ("lesson", "assessment"), so each candidate has to clear
    _is_real_subject rather than being trusted for its position.

    Returning None is a real outcome and the caller must drop the student. A
    subject line built from an unnameable topic is how you mail somebody
    "Your plan starts with assessment".
    """
    tags = node.get("tags") or []
    first_tag = str(tags[0]).strip() if tags and tags[0] else ""
    if first_tag and _is_real_subject(first_tag):
        return first_tag

    # "Topic - Quick Check" → "Topic". The decoration is always a suffix.
    label = str(node.get("label") or "").strip()
    base = label.split(" - ")[0].strip()
    if base and _is_real_subject(base):
        return base
    return None


def _days_since(ts, now):
    try:
        return max(0, (now - ts).days)
    except Exception:
        return None


def _as_datetime(value):
    """
    An exam date as an aware datetime, or None.

    Production stores this field in BOTH shapes — a Firestore Timestamp on some
    accounts and an ISO-8601 string ("2026-09-09T04:00:00.000Z") on others,
    confirmed by a scan on 2026-09-12. Reading only one shape silently drops
    half the students with an exam booked, which for a countdown campaign is
    the entire audience.
    """
    if not value:
        return None
    # Firestore Timestamp / datetime
    if hasattr(value, "year") and hasattr(value, "month"):
        dt = value
        try:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None

    parsed = None
    try:
        # fromisoformat rejects the trailing Z in Python < 3.11.
        parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            parsed = datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None

    # ALWAYS aware. A bare date ("2026-09-09") parses fine and comes back
    # naive, and every caller immediately subtracts an aware `now` from this —
    # which raises TypeError and takes down the whole selector, so one student
    # storing a date-only exam date would stop the campaign for everybody.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _unfinished_plan(db, uid, chat_ids):
    """The student's most recent plan with steps left: (chatId, steps_left)."""
    best = None
    for cid in chat_ids:
        try:
            d = db.collection("chats").document(cid).get().to_dict() or {}
        except Exception:
            continue
        study = d.get("study") or {}
        nodes = _real_nodes(study)
        if not nodes:
            continue
        left = sum(1 for n in nodes if n.get("status") != "completed")
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
        prefs = x.get("emailPrefs") or {}
        users[d.id] = {
            "email": (x.get("email") or "").strip(),
            "name": (x.get("displayName") or "").split(" ")[0] or None,
            "tier": ((x.get("usage") or {}).get("tier")) or "free",
            "examDate": (x.get("onboarding") or {}).get("examDate"),
            # A cheap pre-filter off the same read, so a 2,000-candidate
            # announcement does not do 2,000 extra single-document gets just to
            # discover who opted out. email_sender.is_suppressed stays the
            # authoritative check on whoever survives — it fails closed and
            # sees fields this projection does not (hardBounced).
            "opted_out": bool(prefs.get("unsubscribedAt")) or prefs.get("marketing") is False,
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


def run_winback(db, limit=50, verbose=True, **_):
    """Select, render and send. Honours dry-run, suppression, cap, idempotency."""
    postal = os.getenv("EMAIL_POSTAL_ADDRESS", "")
    # Ask for the budget once rather than letting send_email reject candidates
    # one at a time — each rejection re-scans today's log.
    budget = email_sender.remaining_today(db)
    candidates = select_winback(db, limit=min(limit, budget), verbose=verbose)

    results = {"selected": len(candidates), "budget_today": budget,
               "sent": 0, "dry_run": 0, "skipped": 0, "failed": 0, "detail": []}

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


# ══════════════════════════════════════════════════════════════════════════
# PLAN UNSTARTED — the cohort dormancy actually has here
#
# Measured 2026-09-12 across the 348 dormant students who are free, opted-in
# and reachable:
#
#     plan built, ZERO steps done .... 228
#     chat but no plan ............... 120
#     plan partly done ................. 0
#     plan finished .................... 0
#     clear winback's evidence bar .... 28
#
# winback_gap needs answered questions, so it was speaking to 28 of 348 while
# the 228 — students who uploaded their material, got a plan, and never opened
# it — had no message at all. That is the group this campaign exists for.
#
# The 120 with no plan are deliberately NOT served here. A scan of 60 of them
# found no upload rows and nothing else nameable: the only honest thing we
# could say is "you signed up", which is the generic re-engagement mail this
# package refuses to send. They are reachable by announcement, where the
# content does not pretend to be about them.
# ══════════════════════════════════════════════════════════════════════════

def _plan_first_step(db, uid, chat_ids, now):
    """
    Her most recent plan that has not been started, described honestly.

    Returns None unless every claim the template makes is computable: the step
    count, the first step's subject, its kind and its length.
    """
    best = None
    for cid in chat_ids:
        try:
            d = db.collection("chats").document(cid).get().to_dict() or {}
        except Exception:
            continue
        study = d.get("study") or {}
        nodes = _real_nodes(study)
        if not nodes:
            continue
        if any(n.get("status") == "completed" for n in nodes):
            continue            # started; winback_gap's territory, not this one
        ts = d.get("updatedAt")
        if best is None or (ts and best["ts"] and ts > best["ts"]):
            best = {"cid": cid, "nodes": nodes, "ts": ts,
                    "startedAt": study.get("startedAt")}

    if not best:
        return None

    first = best["nodes"][0]
    topic = _node_topic(first)
    if not topic:
        # Cannot name the first step. Drop the student rather than send
        # "Your plan starts with assessment".
        return None

    kind = NODE_KIND_LABEL.get(first.get("type"), "step")
    minutes = NODE_MINUTES.get(first.get("type"), NODE_MINUTES_DEFAULT)

    # How long ago she built it. Prefer study.startedAt (present on 78% of
    # plans); fall back to the chat's updatedAt, which for an untouched plan is
    # when it was generated.
    since = _days_since(_as_datetime(best["startedAt"]) or best["ts"], now)

    return {
        "chat_id": best["cid"],
        "steps_total": len(best["nodes"]),
        "minutes_total": _node_minutes(best["nodes"]),
        "first_topic": topic,
        "first_kind": kind,
        "first_minutes": minutes,
        "days_since": since,
    }


def select_plan_unstarted(db, limit=50, verbose=False):
    """Dormant students whose plan was built and never opened."""
    users, study_chats, last_seen = _scan_users_and_chats(db)
    now = datetime.now(timezone.utc)
    lo, hi = now - timedelta(days=DORMANT_MAX_DAYS), now - timedelta(days=DORMANT_MIN_DAYS)

    already = email_sender.already_sent_uids(db, "plan_unstarted")
    if already is None:
        if verbose:
            print("  ABORT: cannot read who already received plan_unstarted")
        return []

    out, skipped = [], {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for uid, u in users.items():
        if not u["email"]:
            skip("no_email"); continue
        if u["opted_out"]:
            skip("opted_out"); continue
        if u["tier"] == "pro":
            skip("already_pro"); continue
        if uid in already:
            skip("already_received"); continue
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

        step = _plan_first_step(db, uid, study_chats[uid], now)
        if not step:
            skip("no_unstarted_nameable_plan"); continue

        out.append({"uid": uid, "to": u["email"], "name": u["name"],
                    "resume_url": f"{_app_url()}/c/{step['chat_id']}", **step})
        if len(out) >= limit:
            break

    if verbose:
        print("  selection skips:")
        for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>5}  {k}")
    return out


def run_plan_unstarted(db, limit=50, verbose=True, **_):
    postal = os.getenv("EMAIL_POSTAL_ADDRESS", "")
    budget = email_sender.remaining_today(db)
    candidates = select_plan_unstarted(db, limit=min(limit, budget), verbose=verbose)

    results = {"selected": len(candidates), "budget_today": budget,
               "sent": 0, "dry_run": 0, "skipped": 0, "failed": 0, "detail": []}

    for c in candidates:
        msg = email_templates.plan_unstarted(
            first_name=c["name"], first_topic=c["first_topic"],
            first_kind=c["first_kind"], first_minutes=c["first_minutes"],
            steps_total=c["steps_total"], days_since=c["days_since"],
            resume_url=c["resume_url"],
            unsubscribe_url=email_sender.unsubscribe_url(c["uid"]),
            postal_address=postal,
        )
        r = email_sender.send_email(
            db, uid=c["uid"], to=c["to"], subject=msg["subject"],
            html=msg["html"], campaign="plan_unstarted",
            # Once per student, ever. She either comes back or she does not;
            # asking monthly is a newsletter nobody subscribed to.
            idempotency_key=f"{c['uid']}_plan_unstarted",
        )
        results[r["status"]] = results.get(r["status"], 0) + 1
        results["detail"].append({"uid": c["uid"][:10], "topic": c["first_topic"],
                                  "status": r["status"], "reason": r.get("reason")})
        if verbose:
            print(f"  {r['status']:<8} {c['to'][:34]:<34} "
                  f"{c['steps_total']:>2} steps  {c['first_topic'][:30]}  {r.get('reason','')}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# EXAM COUNTDOWN
#
# The template for this shipped without a selector, so it had never been
# sendable. 35 students have an exam inside the next 14 days (2026-09-12), and
# a student sitting an exam this week is the most time-sensitive message this
# package can send — and the one where being wrong costs the most, which is
# why `measured` gates the "losing the most marks" claim.
# ══════════════════════════════════════════════════════════════════════════

def select_exam_countdown(db, limit=50, verbose=False):
    users, study_chats, last_seen = _scan_users_and_chats(db)
    now = datetime.now(timezone.utc)

    already = email_sender.already_sent_uids(db, "exam_countdown")
    if already is None:
        if verbose:
            print("  ABORT: cannot read who already received exam_countdown")
        return []

    out, skipped = [], {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for uid, u in users.items():
        if not u["email"]:
            skip("no_email"); continue
        if u["opted_out"]:
            skip("opted_out"); continue
        if uid in already:
            skip("already_received"); continue

        exam = _as_datetime(u["examDate"])
        if not exam:
            skip("no_exam_date"); continue
        days = (exam - now).days
        # Past exams are the whole reason this is a window and not a threshold:
        # a "your exam is in -40 days" mail is the clearest possible signal
        # that nobody is reading what we send.
        if days < 0 or days > EXAM_WINDOW_DAYS:
            skip("exam_outside_window"); continue
        if uid not in study_chats:
            skip("no_study_plan"); continue

        reason = email_sender.is_suppressed(db, uid)
        if reason:
            skip(reason); continue

        chat_id, steps_left = _unfinished_plan(db, uid, study_chats[uid])
        if not chat_id:
            skip("plan_already_finished"); continue

        # A measured weak topic if she has one; otherwise the plan's next step,
        # introduced as such. Either way the topic is named from real data.
        perf = _aggregate_performance(db, uid)
        if perf:
            topic, measured = perf["weak_topic"], True
        else:
            step = _plan_first_step(db, uid, study_chats[uid], now)
            if not step:
                skip("no_nameable_topic"); continue
            topic, measured = step["first_topic"], False

        out.append({"uid": uid, "to": u["email"], "name": u["name"],
                    "days_away": max(0, days), "weak_topic": topic,
                    "measured": measured, "steps_left": steps_left,
                    "resume_url": f"{_app_url()}/c/{chat_id}"})
        if len(out) >= limit:
            break

    if verbose:
        print("  selection skips:")
        for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>5}  {k}")
    return out


def run_exam_countdown(db, limit=50, verbose=True, **_):
    postal = os.getenv("EMAIL_POSTAL_ADDRESS", "")
    budget = email_sender.remaining_today(db)
    candidates = select_exam_countdown(db, limit=min(limit, budget), verbose=verbose)

    results = {"selected": len(candidates), "budget_today": budget,
               "sent": 0, "dry_run": 0, "skipped": 0, "failed": 0, "detail": []}

    for c in candidates:
        msg = email_templates.exam_countdown(
            days_away=c["days_away"], weak_topic=c["weak_topic"],
            steps_left=c["steps_left"], measured=c["measured"],
            resume_url=c["resume_url"],
            unsubscribe_url=email_sender.unsubscribe_url(c["uid"]),
            postal_address=postal,
        )
        r = email_sender.send_email(
            db, uid=c["uid"], to=c["to"], subject=msg["subject"],
            html=msg["html"], campaign="exam_countdown",
            # Keyed on the exam, not the day: one countdown per exam sitting.
            # A student who books a later exam is a new, legitimate send.
            idempotency_key=f"{c['uid']}_exam_countdown_{c['days_away']}d",
        )
        results[r["status"]] = results.get(r["status"], 0) + 1
        results["detail"].append({"uid": c["uid"][:10], "days": c["days_away"],
                                  "status": r["status"], "reason": r.get("reason")})
        if verbose:
            print(f"  {r['status']:<8} {c['to'][:34]:<34} exam in {c['days_away']}d  "
                  f"{'measured' if c['measured'] else 'next-step'}  {r.get('reason','')}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# ANNOUNCEMENTS — the same words to everybody
# ══════════════════════════════════════════════════════════════════════════

AUDIENCE_ACTIVE_MAX_DAYS = 3        # last active inside this = "active"
AUDIENCE_COLD_MIN_DAYS = 46         # last active beyond this = "cold"


def resolve_audience(users, last_seen, audience, now=None):
    """
    uids in an audience. See email_announcements for the measured sizes and for
    why `cold` must not be the first thing a new sending domain touches.
    """
    now = now or datetime.now(timezone.utc)
    out = []
    for uid, u in users.items():
        if not u["email"] or u["opted_out"]:
            continue
        seen = last_seen.get(uid)
        days = _days_since(seen, now) if seen else None
        tier = u["tier"]

        if audience == "all":
            ok = True
        elif audience == "pro":
            ok = tier == "pro"
        elif audience == "free":
            ok = tier != "pro"
        elif audience == "active":
            ok = tier != "pro" and days is not None and days < AUDIENCE_ACTIVE_MAX_DAYS
        elif audience == "dormant":
            ok = (tier != "pro" and days is not None
                  and DORMANT_MIN_DAYS <= days <= DORMANT_MAX_DAYS)
        elif audience == "cold":
            ok = days is not None and days >= AUDIENCE_COLD_MIN_DAYS
        else:
            raise ValueError(f"unknown audience: {audience}")

        if ok:
            out.append(uid)
    return out


def select_announcement(db, slug, limit=50, verbose=False):
    """
    Who has not yet received this announcement.

    THE already-received FILTER RUNS BEFORE THE LIMIT, and that ordering is the
    whole reason a send larger than the daily cap works at all. The audience is
    2,079 and the cap starts at 50, so this runs for weeks. A selector that
    took the first 50 every day would hand send_email the same 50 people it
    mailed on day one, all 50 would come back skipped as already_sent, and the
    campaign would stall at 50 recipients while reporting a clean run.
    """
    # The draft gate is re-checked here, not only in run_announcement. This
    # function is the one an ad-hoc script or a future caller would reach for,
    # and a selector that happily returns 2,000 addresses for unapproved copy
    # is a gate with a door beside it.
    ok, reason = email_announcements.is_sendable(slug)
    if not ok:
        raise ValueError(reason)
    a = email_announcements.get(slug)

    users, _study_chats, last_seen = _scan_users_and_chats(db)
    audience = resolve_audience(users, last_seen, a["audience"])

    campaign = f"announce_{slug}"
    already = email_sender.already_sent_uids(db, campaign)
    if already is None:
        if verbose:
            print(f"  ABORT: cannot read who already received {campaign}")
        return []

    out = []
    for uid in audience:
        if uid in already:
            continue
        if email_sender.is_suppressed(db, uid):
            continue
        out.append({"uid": uid, "to": users[uid]["email"], "name": users[uid]["name"]})
        if len(out) >= limit:
            break

    if verbose:
        print(f"  audience={a['audience']} size={len(audience)} "
              f"already_received={len(already)} selected={len(out)}")
    return out


def run_announcement(db, slug=None, limit=50, verbose=True, **_):
    """
    Send one announcement to the next slice of its audience.

    Safe to run daily: idempotent per student per slug, so each run picks up
    where the last one stopped. Refuses outright unless the slug is marked
    approved in email_announcements — EMAIL_ENABLED alone is not consent to
    send a particular set of words.
    """
    if not slug:
        return {"error": "slug is required", "available":
                [a["slug"] for a in email_announcements.listing()]}

    ok, reason = email_announcements.is_sendable(slug)
    if not ok:
        # A refusal, not a failure: this is the review gate doing its job.
        if verbose:
            print(f"  REFUSED: {reason}")
        return {"slug": slug, "refused": reason, "selected": 0,
                "sent": 0, "dry_run": 0, "skipped": 0, "failed": 0}

    a = email_announcements.get(slug)
    postal = os.getenv("EMAIL_POSTAL_ADDRESS", "")
    app = _app_url()

    cta_url = a.get("cta_url") or ""
    if cta_url.startswith("/"):
        cta_url = f"{app}{cta_url}"

    budget = email_sender.remaining_today(db)
    candidates = select_announcement(db, slug, limit=min(limit, budget), verbose=verbose)

    results = {"slug": slug, "audience": a["audience"], "budget_today": budget,
               "selected": len(candidates), "sent": 0, "dry_run": 0,
               "skipped": 0, "failed": 0, "detail": []}

    for c in candidates:
        msg = email_templates.announcement(
            title=a["title"], body=a.get("body") or (),
            bullets_list=a.get("bullets") or (),
            cta_label=a.get("cta_label"), cta_url=cta_url or None,
            preheader=a.get("preheader") or "",
            first_name=c["name"], sign_off=a.get("sign_off"),
            unsubscribe_url=email_sender.unsubscribe_url(c["uid"]),
            postal_address=postal,
        )
        r = email_sender.send_email(
            db, uid=c["uid"], to=c["to"], subject=msg["subject"],
            html=msg["html"], campaign=f"announce_{slug}",
            # Scoped to the slug, so the NEXT announcement reaches the same
            # student while this one can never reach her twice.
            idempotency_key=f"{c['uid']}_announce_{slug}",
        )
        results[r["status"]] = results.get(r["status"], 0) + 1
        results["detail"].append({"uid": c["uid"][:10], "status": r["status"],
                                  "reason": r.get("reason")})
        if verbose:
            print(f"  {r['status']:<8} {c['to'][:40]:<40} {r.get('reason','')}")
    return results


def audience_report(db, include_campaigns=False):
    """
    How many people each audience holds right now, sending nothing.

    This is the thing to read before turning EMAIL_ENABLED on. A campaign that
    selects zero is not a campaign — which was literally true of exam_countdown
    (its template shipped with no selector at all) and nearly true of
    winback_gap (28 of 348 dormant students) with nothing in the system able to
    show it.

    `include_campaigns` is OFF by default because it is slow, not because it is
    unimportant. Counting a campaign means running its real selector: a full
    pass over users and chats, then per-candidate reads of study maps and
    studyPerformance subcollections. That is minutes, not milliseconds, so the
    HTTP route leaves it off and the CLI turns it on.
    """
    users, _study_chats, last_seen = _scan_users_and_chats(db)
    out = {"users": len(users), "audiences": {},
           "announcements": email_announcements.listing(),
           "daily_cap": email_sender.daily_cap(),
           "sent_today": email_sender.sent_today(db)}
    for name in ("all", "active", "dormant", "cold", "free", "pro"):
        try:
            out["audiences"][name] = len(resolve_audience(users, last_seen, name))
        except Exception as e:
            out["audiences"][name] = f"error: {e}"

    if include_campaigns:
        out["campaigns"] = {
            "winback_gap": len(select_winback(db, limit=10 ** 6)),
            "plan_unstarted": len(select_plan_unstarted(db, limit=10 ** 6)),
            "exam_countdown": len(select_exam_countdown(db, limit=10 ** 6)),
        }
    return out


CAMPAIGN_RUNNERS = {
    "winback_gap": run_winback,
    "plan_unstarted": run_plan_unstarted,
    "exam_countdown": run_exam_countdown,
    "announcement": run_announcement,
}
