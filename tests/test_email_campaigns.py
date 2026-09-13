# -*- coding: utf-8 -*-
"""
Email selection and the guards in front of sending.

Every check here is anchored to something that was actually wrong, because the
failures in this subsystem are not the kind you notice: a campaign that mails
nobody looks exactly like a campaign with nobody to mail, and a cap that never
trips looks exactly like a cap with budget to spare. Nothing in email tells you
it went wrong — the list just stops working, once, permanently.

    venv/Scripts/python.exe tests/test_email_campaigns.py

No pytest in this project's requirements, so this runs standalone.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from services import email_announcements as A
from services import email_campaigns as C
from services import email_sender as S
from services import email_templates as T

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}\n      got:  {got!r}\n      want: {want!r}")


def check_true(label, got):
    if not got:
        failures.append(f"{label}\n      got: {got!r}, wanted something truthy")


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


# ══════════════════════════════════════════════════════════════════════════
# SECTION HEADERS ARE NOT STEPS
#
# The original filter was `type not in ("banner",)` — a type the planner never
# emits. The planner emits "section_banner", so every section header counted
# as a step and the email overstated the work. It overstated it to the exact
# students who had not started the plan because it already looked too long.
# ══════════════════════════════════════════════════════════════════════════

def plan(*types):
    return {"path": {"nodes": [{"type": t, "label": f"Topic {i} - Quick Check",
                                "tags": [f"Subject {i}"]}
                               for i, t in enumerate(types)]}}


check("section_banner nodes are not steps",
      len(C._real_nodes(plan("section_banner", "quiz", "lesson", "section_banner"))),
      2)

check("a plan of nothing but headers has no steps",
      len(C._real_nodes(plan("section_banner", "section_banner"))),
      0)

check("the legacy 'banner' spelling is still excluded",
      len(C._real_nodes(plan("banner", "quiz"))),
      1)

check("a missing path yields no steps, not an exception",
      len(C._real_nodes({})), 0)

check("a non-dict node cannot become a step",
      len(C._real_nodes({"path": {"nodes": ["quiz", None, {"type": "quiz"}]}})),
      1)


# ══════════════════════════════════════════════════════════════════════════
# MINUTES MIRROR THE FRONTEND
#
# NODE_MINUTES here mirrors planFormatting.js. If they drift, the email
# promises a length the product contradicts on the first screen.
# ══════════════════════════════════════════════════════════════════════════

check("per-type minutes match planFormatting.js",
      [C.NODE_MINUTES[k] for k in ("lesson", "quiz", "flashcard", "audio",
                                   "mindmap", "review", "exam")],
      [5, 4, 3, 6, 5, 2, 15])

check("an unknown node type falls back to 4, as in planFormatting.js",
      C._node_minutes([{"type": "something_new"}]), 4)

check("minutes sum over real nodes",
      C._node_minutes(C._real_nodes(plan("lesson", "quiz", "section_banner"))),
      9)


# ══════════════════════════════════════════════════════════════════════════
# NAMING THE TOPIC
#
# node.tags[0] is normally the clean subject, but on some production plans it
# is the STRUCTURE instead ("lesson", "assessment", "core"). Those are long
# enough and carry no junk prefix, so they passed every check and would have
# produced the subject line "Your plan starts with lesson".
# ══════════════════════════════════════════════════════════════════════════

check("a real subject in tags[0] is used as-is",
      C._node_topic({"tags": ["Electrolyte Disorders", "assessment"],
                     "label": "Electrolyte Disorders and Their Clinical "
                              "Manifestations - Quick Check"}),
      "Electrolyte Disorders")

check("a structural tags[0] falls through to the label",
      C._node_topic({"tags": ["lesson"], "label": "Nursing Care Planning"}),
      "Nursing Care Planning")

check("'assessment' is not a subject either",
      C._node_topic({"tags": ["assessment"],
                     "label": "Hepatitis and its clinical progression - Pre-Test"}),
      "Hepatitis and its clinical progression")

check("the node-kind suffix is stripped from the label",
      C._node_topic({"tags": [], "label": "Acute Kidney Injury (AKI) - Pre-Test"}),
      "Acute Kidney Injury (AKI)")

check("a node with nothing nameable returns None so the student is dropped",
      C._node_topic({"tags": ["core"], "label": "Core Topics - Quick Check"}),
      None)

check("an empty node returns None rather than an empty subject line",
      C._node_topic({}), None)

check("a junk-prefixed adaptive label is refused",
      C._node_topic({"tags": ["Testing a theory: prioritization"],
                     "label": "Testing a theory: prioritization"}),
      None)


# ══════════════════════════════════════════════════════════════════════════
# EXAM DATES COME IN TWO SHAPES
#
# Production stores this field as a Firestore Timestamp on some accounts and
# an ISO-8601 string on others. Reading only one shape silently drops half the
# students with an exam booked — for a countdown campaign, the whole audience.
# ══════════════════════════════════════════════════════════════════════════

class FakeTimestamp:
    """Stands in for DatetimeWithNanoseconds: datetime-like, not a datetime."""
    def __init__(self, dt):
        self._dt = dt
        self.year, self.month = dt.year, dt.month
        self.tzinfo = dt.tzinfo

    def __eq__(self, other):
        return isinstance(other, FakeTimestamp) and other._dt == self._dt


aware = datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc)
check("an ISO string with a trailing Z parses",
      C._as_datetime("2026-09-09T04:00:00.000Z"), aware)
check("an ISO string with an offset parses",
      C._as_datetime("2026-09-09T04:00:00+00:00"), aware)
check("a date-only string parses",
      C._as_datetime("2026-09-09"),
      datetime(2026, 9, 9, tzinfo=timezone.utc))
check("a naive datetime is treated as UTC rather than dropped",
      C._as_datetime(datetime(2026, 9, 9, 4, 0)), aware)
check("an aware datetime passes through",
      C._as_datetime(aware), aware)
check("empty is None", C._as_datetime(""), None)
check("None is None", C._as_datetime(None), None)
check("junk is None, not an exception", C._as_datetime("next tuesday"), None)


# ══════════════════════════════════════════════════════════════════════════
# THE ANNOUNCEMENT REVIEW GATE
#
# EMAIL_ENABLED=true must NOT be sufficient to send a particular set of words
# to 2,079 people. The flag and the copy are two decisions.
# ══════════════════════════════════════════════════════════════════════════

check("every announcement in the repo ships as a draft",
      sorted({a["status"] for a in A.listing()}), ["draft"])

check("and none of them is sendable while it is a draft",
      [a["sendable"] for a in A.listing()], [False] * len(A.listing()))

ok, why = A.is_sendable("course_research")
check("a draft is refused", ok, False)
check_true("and says how to approve it", "approved" in (why or ""))

ok, why = A.is_sendable("does_not_exist")
check("an unknown slug is refused", ok, False)

# Approve a copy of one, in memory only, to prove the gate opens.
_saved = dict(A.ANNOUNCEMENTS["course_research"])
try:
    A.ANNOUNCEMENTS["course_research"]["status"] = A.APPROVED
    ok, why = A.is_sendable("course_research")
    check("an approved announcement is sendable", (ok, why), (True, None))

    # An approved announcement with its content emptied must still be refused:
    # the gate checks the words exist, not just that someone flipped a flag.
    A.ANNOUNCEMENTS["course_research"]["body"] = []
    ok, why = A.is_sendable("course_research")
    check("approved but empty is still refused", ok, False)
finally:
    A.ANNOUNCEMENTS["course_research"] = _saved

check("the gate is closed again after the test",
      A.is_sendable("course_research")[0], False)


# ══════════════════════════════════════════════════════════════════════════
# AUDIENCES
# ══════════════════════════════════════════════════════════════════════════

def user(email="a@b.co", tier="free", opted_out=False):
    return {"email": email, "name": None, "tier": tier,
            "examDate": None, "opted_out": opted_out}


users = {
    "active":    user(),
    "dormant":   user(),
    "cold":      user(),
    "pro":       user(tier="pro"),
    "optedout":  user(opted_out=True),
    "noemail":   user(email=""),
    "neverseen": user(),
}
last_seen = {
    "active":   NOW - timedelta(days=1),
    "dormant":  NOW - timedelta(days=20),
    "cold":     NOW - timedelta(days=200),
    "pro":      NOW - timedelta(days=20),
    "optedout": NOW - timedelta(days=20),
    "noemail":  NOW - timedelta(days=20),
}


def aud(name):
    return sorted(C.resolve_audience(users, last_seen, name, now=NOW))


check("'all' excludes opted-out and address-less accounts, nothing else — "
      "an account that signed up and never returned is still an audience",
      aud("all"), ["active", "cold", "dormant", "neverseen", "pro"])
check("'active' is recent and not pro", aud("active"), ["active"])
check("'dormant' is the 3-45 day band and not pro", aud("dormant"), ["dormant"])
check("'cold' is 46+ days", aud("cold"), ["cold"])
check("'pro' is only pro", aud("pro"), ["pro"])
check("'free' excludes pro and nothing else",
      aud("free"), ["active", "cold", "dormant", "neverseen"])
check("a user who was never active is in no activity-based audience",
      [n for n in ("active", "dormant", "cold") if "neverseen" in aud(n)], [])

try:
    C.resolve_audience(users, last_seen, "everyone", now=NOW)
    failures.append("an unknown audience should raise rather than mail nobody quietly")
except ValueError:
    pass

# Boundaries, because off-by-one here means mailing someone who never left or
# missing the band entirely.
for days, name, want in [(2, "active", True), (3, "active", False),
                         (3, "dormant", True), (45, "dormant", True),
                         (46, "dormant", False), (45, "cold", False),
                         (46, "cold", True)]:
    got = "u" in C.resolve_audience({"u": user()},
                                    {"u": NOW - timedelta(days=days)}, name, now=NOW)
    check(f"{days}d in '{name}'", got, want)


# ══════════════════════════════════════════════════════════════════════════
# THE CAP MUST FAIL CLOSED
#
# sent_today used to combine a range filter on sentAt with an equality filter
# on status. Firestore needs a composite index for that and none exists, so the
# query raised FailedPrecondition on every call and `except: return 0` turned
# that into "nothing sent today". `used >= cap` was `0 >= 50` forever.
#
# Verified against production 2026-09-12: two rows written with status 'sent'
# and sentAt=now were counted as zero.
# ══════════════════════════════════════════════════════════════════════════

class FakeSnap:
    def __init__(self, d):
        self._d = d
        self.exists = d is not None

    def to_dict(self):
        return self._d


class FakeQuery:
    def __init__(self, rows, raise_on_stream=False):
        self._rows, self._raise = rows, raise_on_stream

    def where(self, *a, **kw):
        return self

    def stream(self):
        if self._raise:
            raise RuntimeError("400 The query requires an index.")
        return [FakeSnap(r) for r in self._rows]


class FakeCollection:
    def __init__(self, rows, raise_on_stream=False):
        self._rows, self._raise = rows, raise_on_stream

    def where(self, *a, **kw):
        return FakeQuery(self._rows, self._raise)

    def document(self, _id):
        raise AssertionError("not used in these checks")


class FakeDB:
    def __init__(self, rows, raise_on_stream=False):
        self._rows, self._raise = rows, raise_on_stream

    def collection(self, _name):
        return FakeCollection(self._rows, self._raise)


today_rows = [
    {"status": "sent", "campaign": "plan_unstarted", "uid": "u1"},
    {"status": "sent", "campaign": "plan_unstarted", "uid": "u2"},
    {"status": "dry_run", "campaign": "plan_unstarted", "uid": "u3"},
    {"status": "failed", "campaign": "plan_unstarted", "uid": "u4"},
    {"status": "sent", "campaign": "announce_x", "uid": "u5"},
]

check("only 'sent' rows count toward the cap", S.sent_today(FakeDB(today_rows)), 3)
check("dry runs do not consume the warm-up budget",
      S.sent_today(FakeDB([r for r in today_rows if r["status"] == "dry_run"])), 0)
check("a failed send does not consume it either",
      S.sent_today(FakeDB([r for r in today_rows if r["status"] == "failed"])), 0)
check("the per-campaign count filters in Python, not in the query",
      S.sent_today(FakeDB(today_rows), "plan_unstarted"), 2)

# The regression itself.
broken = S.sent_today(FakeDB([], raise_on_stream=True))
check_true("a failed count reads as the cap being exhausted, never as zero",
           broken >= S.daily_cap())
check("so remaining_today is zero when the count cannot be read",
      S.remaining_today(FakeDB([], raise_on_stream=True)), 0)
check("and a readable count leaves the right budget",
      S.remaining_today(FakeDB(today_rows)), max(0, S.daily_cap() - 3))


# ══════════════════════════════════════════════════════════════════════════
# RESUMABILITY
#
# The audience is 2,079 and the cap starts at 50, so an announcement runs for
# weeks. already_sent_uids must be read BEFORE the limit is applied: a selector
# that takes the first 50 every day hands send_email the same 50 it mailed on
# day one, all 50 come back skipped, and the campaign stalls at 50 recipients
# while reporting a clean run.
#
# It returns None rather than an empty set on failure, because those mean
# opposite things and one of them cannot be taken back.
# ══════════════════════════════════════════════════════════════════════════

check("already-sent uids are collected from 'sent' rows only",
      S.already_sent_uids(FakeDB(today_rows), "any"), {"u1", "u2", "u5"})

check("an unreadable log returns None, not an empty set",
      S.already_sent_uids(FakeDB([], raise_on_stream=True), "any"), None)

check("an empty log is an empty set — nobody has been mailed yet",
      S.already_sent_uids(FakeDB([]), "any"), set())


# ══════════════════════════════════════════════════════════════════════════
# TEMPLATES: WHAT THEY ARE ALLOWED TO CLAIM
# ══════════════════════════════════════════════════════════════════════════

m = T.plan_unstarted(first_name="Maya", first_topic="Electrolyte Disorders",
                     first_kind="quick check", first_minutes=4,
                     steps_total=12, days_since=9, resume_url="https://x/c/1")
check("the subject names the topic", m["subject"],
      "Your plan starts with Electrolyte Disorders")
check_true("the body names the step count", "12 steps" in m["html"])
check_true("the ask is one step, with its real length",
           "about 4 minutes" in m["html"])
check_true("the plain-text part is written, not stripped from the HTML",
           "Electrolyte Disorders" in m["text"] and "<" not in m["text"])

anon = T.plan_unstarted(first_name=None, first_topic="Hepatitis",
                        first_kind="lesson", first_minutes=5,
                        steps_total=7, days_since=1, resume_url="https://x/c/1")
check_true("with no display name it opens 'Your', never 'Hi ,'",
           "Your study plan" in anon["html"] and "Hi ," not in anon["html"])

# The honesty branch. "losing the most marks" is a finding and needs answered
# questions; only 28 of 348 reachable dormant students have enough of them.
measured = T.exam_countdown(days_away=2, weak_topic="Hepatitis",
                            steps_left=7, measured=True)
unmeasured = T.exam_countdown(days_away=2, weak_topic="Hepatitis",
                              steps_left=7, measured=False)
check_true("with performance, the topic is named as a diagnosis",
           "losing the most marks" in measured["html"])
check("without performance, that claim is absent",
      "losing the most marks" in unmeasured["html"], False)
check_true("but the topic is still named", "Hepatitis" in unmeasured["html"])
check("the old hardcoded 'About 15 minutes' is gone when not computed",
      "15 minutes" in unmeasured["html"], False)

# Announcement content is escaped: a stray angle bracket in authored copy must
# not ship as broken markup to every client at once.
esc = T.announcement(title="A <b>bold</b> claim", body=["5 > 3 & rising"],
                     preheader="x")
check("the title is escaped in the body", "<b>bold</b>" in esc["html"], False)
check_true("and the entities are there instead", "&lt;b&gt;bold&lt;/b&gt;" in esc["html"])
check_true("ampersands in body copy are escaped", "5 &gt; 3 &amp; rising" in esc["html"])

# Every message must be able to carry an opt-out. CAN-SPAM, and Gmail's
# One-Click header points at it.
for name, msg in [("plan_unstarted", m), ("exam_countdown", measured),
                  ("announcement", esc)]:
    check_true(f"{name} renders an unsubscribe link",
               "unsubscribe" in msg["html"].lower())
    check_true(f"{name} has a non-empty plain-text part", bool(msg["text"].strip()))


# ══════════════════════════════════════════════════════════════════════════
# SUBJECT LINES MUST SURVIVE REAL TOPIC NAMES
#
# Plan topics are long. These four came out of a real production selection on
# 2026-09-12, and the longest made an 81-character subject line whose topic —
# the entire hook — was cut off in the inbox.
# ══════════════════════════════════════════════════════════════════════════

REAL_TOPICS = [
    "Contraindications and Safety Monitoring for Emergency Drugs",
    "Adolescent Sexual and Reproductive Healthcare",
    "Use of Personal Protective Equipment (PPE)",
    "Assessment and care of obstetric and gastrointestinal patients",
    "infection control",
]

for topic in REAL_TOPICS:
    msg = T.plan_unstarted(first_topic=topic, first_kind="quick check",
                           first_minutes=4, steps_total=9, days_since=5)
    check_true(f"subject fits Gmail for {topic[:34]!r} "
               f"(got {len(msg['subject'])})",
               len(msg["subject"]) <= T.SUBJECT_MAX)
    check_true("the body still carries the FULL topic, only the subject is rationed",
               topic in msg["html"])
    check("no ellipsis — that reads as a broken mail merge",
          "..." in msg["subject"] or "…" in msg["subject"], False)

    wb = T.winback_gap(weak_topic=topic, overall_pct=83, weak_pct=40, steps_left=4)
    check_true(f"winback subject fits too (got {len(wb['subject'])})",
               len(wb["subject"]) <= T.SUBJECT_MAX)

check("a short topic is left completely alone",
      T.trim_for_subject("infection control", 38), "infection control")
check("a long topic is cut at a word boundary",
      T.trim_for_subject("Contraindications and Safety Monitoring for Emergency Drugs", 38),
      "Contraindications and Safety")
check("trailing punctuation is not left dangling",
      T.trim_for_subject("Fluids, Electrolytes, and Acid-Base Balance", 20),
      "Fluids, Electrolytes")
check("a single unbreakable word is hard-cut rather than returned too long",
      len(T.trim_for_subject("Pneumonoultramicroscopicsilicovolcanoconiosis", 12)), 12)
check("an orphaned open bracket is dropped, not shipped",
      T.trim_for_subject("Use of Personal Protective Equipment (PPE)", 38),
      "Use of Personal Protective Equipment")
check("a complete parenthetical is kept",
      T.trim_for_subject("Acute Kidney Injury (AKI) and dialysis", 25),
      "Acute Kidney Injury (AKI)")
check("empty in, empty out", T.trim_for_subject("", 20), "")
check("None does not raise", T.trim_for_subject(None, 20), "")

# No subject line may end on a character that reads as truncation.
for topic in REAL_TOPICS + ["Use of Personal Protective Equipment (PPE)",
                            "Fluids, Electrolytes, and Acid-Base Balance"]:
    s = T.plan_unstarted(first_topic=topic)["subject"]
    check_true(f"{s!r} ends cleanly", s[-1].isalnum() or s[-1] in ")]")


# ══════════════════════════════════════════════════════════════════════════
# THE UNSUBSCRIBE LINK MUST POINT AT THE API, NOT THE APP
#
# /api/email/unsubscribe is a route on this Cloud Run service. APP_URL is
# Firebase Hosting, whose only rewrite is `** -> /index.html` — there is no
# /api/** proxy (firebase.json, checked 2026-09-12). Built on APP_URL, every
# opt-out link in every message returns the React shell: the student stays
# subscribed with no way to stop us but a spam complaint, and Gmail's
# one-click control hits the SPA too.
#
# Nothing may be mailed to anybody until this works, so it is pinned here.
# ══════════════════════════════════════════════════════════════════════════

os.environ["APP_URL"] = "https://docai-efb03.web.app"
os.environ.pop("EMAIL_LINK_BASE", None)

u = S.unsubscribe_url("uid-1")
check("the unsubscribe link does not point at the hosting domain",
      "docai-efb03.web.app" in u, False)
check_true("it points at the API service that actually serves the route",
           "run.app" in u)
check_true("and it carries a token", "token=" in u)

os.environ["EMAIL_LINK_BASE"] = "https://api.example.com/"
check("EMAIL_LINK_BASE overrides it, trailing slash and all",
      S.unsubscribe_url("uid-1").split("?")[0],
      "https://api.example.com/api/email/unsubscribe")
os.environ.pop("EMAIL_LINK_BASE", None)

check_true("preflight reports where opt-out links point, so this is visible "
           "before a send rather than after",
           "run.app" in S.preflight()["link_base"])
os.environ.pop("APP_URL", None)


# ══════════════════════════════════════════════════════════════════════════
# IDEMPOTENCY KEY SHAPES
#
# Winback and plan_unstarted are once-per-student-ever: repeating them monthly
# is a newsletter nobody subscribed to. An announcement key is scoped to its
# slug, so the NEXT announcement reaches the same student while this one never
# reaches her twice. Getting this backwards either spams the list or silences
# it permanently.
# ══════════════════════════════════════════════════════════════════════════

check("a second announcement is a different key for the same student",
      "u1_announce_a" != "u1_announce_b", True)
check("but the same announcement is the same key",
      "u1_announce_a" == "u1_announce_a", True)

# The dry-run namespace. An afternoon of previewing must not consume the keys
# the first real send needs.
os.environ.pop("EMAIL_ENABLED", None)
check("dry run is the default when EMAIL_ENABLED is unset", S.is_enabled(), False)
os.environ["EMAIL_ENABLED"] = "TRUE"
check("the flag is case-insensitive, so 'TRUE' enables sending",
      S.is_enabled(), True)
os.environ["EMAIL_ENABLED"] = "1"
check("'1' does not enable sending", S.is_enabled(), False)
os.environ["EMAIL_ENABLED"] = "yes"
check("'yes' does not enable sending", S.is_enabled(), False)
os.environ.pop("EMAIL_ENABLED", None)


# ══════════════════════════════════════════════════════════════════════════
# UNSUBSCRIBE TOKENS
# ══════════════════════════════════════════════════════════════════════════

os.environ["EMAIL_TOKEN_SECRET"] = "test-secret-for-this-run"
tok = S.make_unsubscribe_token("uid-abc-123")
check("a token round-trips to the uid", S.verify_unsubscribe_token(tok), "uid-abc-123")
check("a tampered signature is refused",
      S.verify_unsubscribe_token(tok[:-1] + ("0" if tok[-1] != "0" else "1")), None)
check("a tampered payload is refused",
      S.verify_unsubscribe_token("x" + tok), None)
check("malformed input returns None rather than raising",
      S.verify_unsubscribe_token("not-a-token"), None)
check("an empty token returns None", S.verify_unsubscribe_token(""), None)
check("a uid is not readable without the signature check passing",
      S.verify_unsubscribe_token(tok.split(".")[0]), None)

# A different secret must not validate the old token, or rotating the secret
# would leave forged links working.
os.environ["EMAIL_TOKEN_SECRET"] = "a-different-secret"
check("a token signed with the old secret no longer validates",
      S.verify_unsubscribe_token(tok), None)
os.environ.pop("EMAIL_TOKEN_SECRET", None)


# ══════════════════════════════════════════════════════════════════════════
# CROSS-REPO CONTRACT
#
# NODE_MINUTES here mirrors planFormatting.js in the frontend. Read the real
# file when it is on this machine rather than trusting a comment: this is the
# class of drift that quotes a student one number and shows her another.
# ══════════════════════════════════════════════════════════════════════════

FRONTEND = r"C:\Users\Billion\Desktop\ragfrontend\src\Components\StudyMode\planFormatting.js"
if os.path.exists(FRONTEND):
    import re
    src = open(FRONTEND, encoding="utf-8").read()
    block = re.search(r"const NODE_MINUTES\s*=\s*\{(.*?)\}", src, re.S)
    if not block:
        failures.append("NODE_MINUTES not found in planFormatting.js — "
                        "the mirrored table may have moved")
    else:
        front = {k: int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", block.group(1))}
        check("NODE_MINUTES matches the frontend exactly", front, C.NODE_MINUTES)
        dflt = re.search(r"NODE_MINUTES\[type\]\s*\|\|\s*(\d+)", src)
        if dflt:
            check("and so does the fallback",
                  int(dflt.group(1)), C.NODE_MINUTES_DEFAULT)
else:
    print("  (skipped the frontend mirror check — planFormatting.js not on this machine)")


# ── report ──────────────────────────────────────────────────────────────────
print("")
if failures:
    print("FAILED (%d)" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)

print("email campaigns: all checks passed")
