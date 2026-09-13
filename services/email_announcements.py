"""
email_announcements.py
The words that go to everybody, and the review gate in front of them.

WHY ANNOUNCEMENTS ARE THE DANGEROUS KIND

Every other campaign in this package derives its copy from one student's own
record and refuses to send when the numbers are not there — the honesty check
is mechanical. An announcement has no such check. The same sentences go to
2,079 people, nothing in them can be verified per recipient, and the entire
burden of whether they are true sits on whoever typed them.

So the content lives here, in reviewed code, rather than in a Firestore
document. Two reasons, and the second is the real one:

  1. The preview tool renders exactly these bytes, so what you approve is what
     ships.
  2. `emailLog` and the rest of that database are client-writable under the
     current transitional firestore.rules. Announcement copy read from
     Firestore would be attacker-controlled text mailed from our domain to our
     entire list. Code is the only authoring surface here that is not.

THE DRAFT GATE

`status` must be "approved" before a send is possible. EMAIL_ENABLED=true is
NOT sufficient. This exists so copy can be written, previewed and left in the
repo without sitting one environment variable away from 2,000 inboxes — the
flag and the words should never be the same decision.

Flipping a slug to "approved" is therefore the actual go/no-go, and it is a
diff someone has to read.

AUDIENCES

Resolved in email_campaigns.resolve_audience(). Deliberately coarse: an
announcement is a product update, and slicing it finely is how you end up
sending the wrong one to the wrong cohort. Measured 2026-09-12:

    all          2,079 addressable (has an address, not unsubscribed)
    dormant        348 (last active 3-45 days ago, free)
    cold         1,498 (last active 46+ days ago)   ← see the warning below
    active          75 (last active under 3 days, free)
    pro              17

COLD IS NOT A DEFAULT. 1,498 of these accounts have been silent for 46+ days
and 1,245 for over 90. CASL implied consent runs about six months from signup,
so the oldest of them are outside it; on a domain that has never sent a single
message, mailing the least engaged 70% of the list first is the fastest way to
lose the channel. Warm up on `active` and `dormant`, watch the bounce and
complaint rates, and only then consider the rest.
"""

# ── The one hard gate ─────────────────────────────────────────────────────
DRAFT = "draft"
APPROVED = "approved"


ANNOUNCEMENTS = {

    # ──────────────────────────────────────────────────────────────────────
    # DRAFT. Describes what shipped in course intelligence. Every claim below
    # is checkable against the code that implements it:
    #
    #   - "looks up your course"  -> services/course_intelligence.py runs three
    #     concurrent web-search passes (course, instructor, public resources)
    #     before the plan is built.
    #   - "orders your plan by what's likely to be tested" -> the report's
    #     study_strategy.ordered_topics drives _order_units_by_priority, which
    #     is what stops the plan following the order of the uploaded slides.
    #   - "instead of the order of your slides" -> that was the previous
    #     behaviour, so the contrast is real and not invented.
    #
    # Nothing here promises accuracy we do not deliver: the report marks each
    # finding verified / public / inference, and a researched section that
    # cites nothing is dropped rather than softened.
    # ──────────────────────────────────────────────────────────────────────
    "course_research": {
        "status": DRAFT,
        "audience": "dormant",
        "title": "Your plan now starts from your actual course",
        "preheader": "We look up your course and exam before building the plan.",
        "body": [
            "When you upload your material now, NurseQuizAI looks up your "
            "actual course and exam before it builds anything — the syllabus, "
            "the exam format, what your program tends to test.",
            "Then it orders your study plan by what is most likely to be on "
            "the exam, instead of following the order of your slides. The "
            "first thing you study is the thing that matters most, not "
            "whatever happened to be on page one.",
        ],
        "bullets": [
            "Takes about 20 seconds, while your files are still uploading",
            "Every finding is labelled — confirmed from your material, "
            "found in a public source, or flagged as our inference",
            "Works from your material alone if there is nothing public to find",
        ],
        "cta_label": "Upload your material",
        "cta_url": "/",
        "sign_off": None,
    },

    # ──────────────────────────────────────────────────────────────────────
    # DRAFT. The post-node readout. Claims check against StudyMode's
    # nodeReadout.js and the backend's practice_debrief.py: after each step the
    # student gets result -> insight -> recommendation, and every node type now
    # carries a written note rather than only quizzes.
    # ──────────────────────────────────────────────────────────────────────
    "step_feedback": {
        "status": DRAFT,
        "audience": "dormant",
        "title": "You now get a read after every step",
        "preheader": "What you missed, why, and what to do next.",
        "body": [
            "Finishing a step used to just tick a box. Now every step ends "
            "with a short read on how it went: what you got wrong, the "
            "pattern behind it, and which one thing to do next.",
            "It applies to every kind of step, not just quizzes — lessons, "
            "flashcards and practice exams all end with a note.",
        ],
        "bullets": [],
        "cta_label": "Open your study plan",
        "cta_url": "/",
        "sign_off": None,
    },
}


def get(slug):
    """One announcement, or None."""
    return ANNOUNCEMENTS.get(slug)


def is_sendable(slug):
    """
    (ok, reason). The gate every send path must pass through.

    Checks only what is knowable from the content itself. Consent, the daily
    cap and idempotency are enforced downstream in email_sender.
    """
    a = ANNOUNCEMENTS.get(slug)
    if not a:
        return False, f"unknown announcement: {slug}"
    if a.get("status") != APPROVED:
        return False, (f"announcement '{slug}' is status="
                       f"{a.get('status')!r}; set it to {APPROVED!r} to send")
    if not a.get("title"):
        return False, "no title (it is also the subject line)"
    if not a.get("body"):
        return False, "no body"
    if not a.get("audience"):
        return False, "no audience"
    return True, None


def listing():
    """Every announcement and whether it could send. For the preflight route."""
    out = []
    for slug, a in ANNOUNCEMENTS.items():
        ok, reason = is_sendable(slug)
        out.append({
            "slug": slug,
            "status": a.get("status"),
            "audience": a.get("audience"),
            "title": a.get("title"),
            "sendable": ok,
            "blocked_because": reason,
        })
    return out
