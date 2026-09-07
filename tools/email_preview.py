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
        resume_url="https://example.com/c/abc123",
    ),
}

COMMON = dict(
    unsubscribe_url="https://example.com/api/email/unsubscribe?token=demo",
    postal_address="NurseQuizAI · 123 Example St, Toronto ON, Canada",
)


def build(name):
    fn = T.CAMPAIGNS[name]
    kwargs = dict(SAMPLES.get(name, {}))
    kwargs.update(COMMON)
    return fn(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the index in a browser")
    ap.add_argument("--campaign", help="render only this one")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    names = [args.campaign] if args.campaign else list(T.CAMPAIGNS)

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
             <b>Preheader:</b> {msg['preheader']}</p>
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
