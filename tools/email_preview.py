"""
Render every email template to a file you can open, without sending anything.

Usage (from NQBackEnd2 root):
    venv\\Scripts\\python tools\\email_preview.py
    venv\\Scripts\\python tools\\email_preview.py --open
    venv\\Scripts\\python tools\\email_preview.py --campaign winback_gap

WHY A FILE AND NOT A TEST SEND

The send loop is slow, burns reputation on a cold domain, and — because
sending is capped and idempotent by design — is awkward to repeat. Reviewing
copy and layout does not need any of that. This renders the exact bytes that
would be handed to Resend.

WHAT A BROWSER CAN AND CANNOT TELL YOU

It CAN check: copy, hierarchy, the preheader, that the plain-text part reads
like something a person wrote, that the unsubscribe link is present, and that
nothing depends on an image.

It CANNOT check Outlook, which renders through Word and is where table layouts
earn their keep. For that there is no substitute for a real send — so the
recommended loop is: iterate here, then send ONE to yourself before any list.

The index page renders each template with images disabled in one pane, because
most first opens block images and that is the version most people see.
"""

import argparse
import os
import sys
import webbrowser

# Run as `python tools\email_preview.py` from the repo root, so the repo root
# has to be importable for `services.*` to resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import email_announcements as A
from services import email_templates as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_previews")

# Sample data chosen to look like a real student rather than a happy path:
# a weak score, a near exam, an unfinished plan.
SAMPLES = {
    "winback_gap": dict(
        first_name="Sarina", weak_topic="Prioritisation",
        overall_pct=83, weak_pct=40, steps_left=4,
        resume_url="https://example.com/c/abc123",
    ),
    "exam_countdown": dict(
        days_away=3, weak_topic="Fluid & Electrolytes", steps_left=4,
        minutes_left=38, measured=True,
        resume_url="https://example.com/c/abc123",
    ),
    # The same message for a student with no answered questions, where the
    # "losing the most marks" claim is not available. Both variants are
    # previewed because the difference between them is the honesty of the
    # message, and that is exactly the kind of thing a reviewer should see
    # side by side rather than take on trust.
    "exam_countdown_unmeasured": dict(
        days_away=1, weak_topic="Acute Kidney Injury", steps_left=15,
        measured=False, resume_url="https://example.com/c/abc123",
    ),
    # Values taken from a real selection against production on 2026-09-12:
    # a 12-step plan, untouched for 9 days, opening on a 4-minute quick check.
    "plan_unstarted": dict(
        first_name="Maya", first_topic="Electrolyte Disorders",
        first_kind="quick check", first_minutes=4, steps_total=12,
        days_since=9, resume_url="https://example.com/c/abc123",
    ),
    # And the no-name case: 42% of the addressable list has no display name.
    "plan_unstarted_noname": dict(
        first_name=None, first_topic="Hepatitis and Clinical Progression",
        first_kind="lesson", first_minutes=5, steps_total=18,
        days_since=1, resume_url="https://example.com/c/abc123",
    ),
}

# Which template each preview name renders, where the name is not the template.
ALIASES = {
    "exam_countdown_unmeasured": "exam_countdown",
    "plan_unstarted_noname": "plan_unstarted",
}

COMMON = dict(
    unsubscribe_url="https://example.com/api/email/unsubscribe?token=demo",
    postal_address="NurseQuizAI · 123 Example St, Toronto ON, Canada",
)


def build(name):
    """
    Render one preview.

    An `announce_<slug>` name renders the AUTHORED content from
    email_announcements rather than sample data. That is the point of
    previewing an announcement: nothing about it is derived per recipient, so
    the only check available is a human reading the actual words that will
    ship. Sample copy here would defeat the exercise entirely.
    """
    if name.startswith("announce_"):
        slug = name[len("announce_"):]
        a = A.get(slug)
        if not a:
            raise KeyError(f"unknown announcement: {slug}")
        return T.announcement(
            title=a["title"], body=a.get("body") or (),
            bullets_list=a.get("bullets") or (),
            cta_label=a.get("cta_label"),
            cta_url=(a.get("cta_url") or "").replace("/", "https://example.com/", 1)
                    if (a.get("cta_url") or "").startswith("/") else a.get("cta_url"),
            preheader=a.get("preheader") or "",
            first_name="Maya", sign_off=a.get("sign_off"), **COMMON)

    fn = T.CAMPAIGNS[ALIASES.get(name, name)]
    kwargs = dict(SAMPLES.get(name, {}))
    kwargs.update(COMMON)
    return fn(**kwargs)


def all_names():
    """Every derived template, then every authored announcement."""
    derived = [n for n in T.CAMPAIGNS if n != "announcement"]
    derived += [n for n in SAMPLES if n in ALIASES]
    return derived + [f"announce_{slug}" for slug in A.ANNOUNCEMENTS]


def _utf8_stdout():
    """
    The Windows console here is cp1252, and this tool prints ✓ and ⚠️ — so it
    died with UnicodeEncodeError partway through the first campaign and had
    never once rendered a full set. Subject lines carry en dashes and student
    topics carry accents, so stripping the glyphs would only move the crash.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the index in a browser")
    ap.add_argument("--campaign", help="render only this one")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    names = [args.campaign] if args.campaign else all_names()

    cards = ""
    for name in names:
        msg = build(name)

        html_path = os.path.join(OUT, f"{name}.html")
        text_path = os.path.join(OUT, f"{name}.txt")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(msg["html"])
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(msg["text"])

        # Cheap lint for the mistakes that actually bite.
        warn = []
        gate = ""
        if name.startswith("announce_"):
            slug = name[len("announce_"):]
            ok, why = A.is_sendable(slug)
            a = A.get(slug) or {}
            gate = f"{a.get('status', '?')} → audience: {a.get('audience', '?')}"
            if ok:
                # Not a defect — but the only difference between copy sitting
                # in the repo and copy in 2,000 inboxes is this one field, so
                # it should never be something you scroll past.
                warn.append("APPROVED — this sends once EMAIL_ENABLED=true")
            else:
                gate += f"  ({why})"
            if not a.get("cta_url"):
                warn.append("no CTA — an announcement with nothing to click "
                            "is a notification, not an update")
        if "EMAIL_POSTAL_ADDRESS" in msg["html"]:
            warn.append("postal address missing (CAN-SPAM)")
        if "unsubscribe" not in msg["html"].lower():
            warn.append("NO UNSUBSCRIBE LINK")
        if "<img" in msg["html"]:
            warn.append("contains an image — check it reads with images off")
        if len(msg["subject"]) > 60:
            warn.append(f"subject is {len(msg['subject'])} chars (Gmail cuts ~60)")
        if not msg["text"].strip():
            warn.append("empty plain-text part (hurts deliverability)")

        print(f"\n{name}")
        if gate:
            print(f"  gate      : {gate}")
        print(f"  subject   : {msg['subject']}  ({len(msg['subject'])} chars)")
        print(f"  preheader : {msg['preheader']}")
        print(f"  html      : {html_path}")
        print(f"  text      : {text_path}")
        for w in warn:
            print(f"  ⚠️  {w}")
        if not warn:
            print("  ✓ no lint warnings")

        warn_html = ("".join(f"<li>{w}</li>" for w in warn)) or "<li>none</li>"
        cards += f"""
        <section>
          <h2>{name}</h2>
          <p class="meta"><b>Subject:</b> {msg['subject']}<br>
             <b>Preheader:</b> {msg['preheader']}
             {f"<br><b>Gate:</b> {gate}" if gate else ""}</p>
          <ul class="warn">{warn_html}</ul>
          <div class="panes">
            <div><h3>As rendered</h3><iframe src="{name}.html"></iframe></div>
            <div><h3>Plain-text part</h3><iframe src="{name}.txt"></iframe></div>
          </div>
        </section>"""

    index = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Email previews</title><style>
 body{{font:15px/1.6 -apple-system,Segoe UI,sans-serif;background:#f4f1ec;
      margin:0;padding:28px;color:#3d3d3d}}
 h1{{margin:0 0 6px}} h2{{margin:0 0 4px;color:#c46a5a}}
 .lead{{color:#6b6b6b;margin:0 0 26px;max-width:70ch}}
 section{{background:#fff;border:1px solid #e6ded5;border-radius:14px;
          padding:18px;margin-bottom:26px}}
 .meta{{color:#6b6b6b;font-size:13px;margin:0 0 8px}}
 .warn{{margin:0 0 12px;padding-left:18px;color:#b3541e;font-size:13px}}
 .panes{{display:flex;gap:16px;flex-wrap:wrap}}
 .panes>div{{flex:1;min-width:320px}}
 h3{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
     color:#9a9a9a;margin:0 0 6px}}
 iframe{{width:100%;height:620px;border:1px solid #e6ded5;border-radius:10px;
         background:#fff}}
</style></head><body>
<h1>Email previews</h1>
<p class="lead">Rendered from the same code that builds the real message.
A browser will not tell you how Outlook renders this — it renders mail through
Word — so treat this as the fast loop and send one real test to yourself before
any list send.</p>
{cards}
</body></html>"""

    index_path = os.path.join(OUT, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index)
    print(f"\nIndex: {index_path}")

    if args.open:
        webbrowser.open("file:///" + index_path.replace("\\", "/"))


if __name__ == "__main__":
    main()
