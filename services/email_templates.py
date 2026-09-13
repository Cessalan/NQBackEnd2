"""
email_templates.py
Branded HTML email, built for the clients that actually exist.

WHY THIS IS NOT JUST HTML

An email is not a web page and the habits do not transfer:

  - Gmail STRIPS <head>, so a <style> block is thrown away. Every rule has to
    be an inline style attribute on the element it affects.
  - Outlook renders through Word. No flexbox, no grid, no background-image,
    unreliable padding on divs. Layout is <table>, as it was in 2005.
  - Images are blocked by default in most clients on first open, so a design
    that carries its meaning in images arrives blank. Nothing here needs an
    image to make sense — the brand is carried by colour and type.
  - Web fonts do not load in Outlook and are inconsistent elsewhere, so the
    stack degrades to system fonts on purpose rather than by accident.

WHAT MAKES MAIL LAND

Two things here are deliverability features, not design ones:

  - A PLAIN-TEXT ALTERNATIVE. A multipart message with a real text part is
    treated as more legitimate than HTML alone; an html-only blast from a cold
    domain to a 78%-Gmail list is a spam-filter tell. Every template returns
    both, and the text part is generated from the same content blocks rather
    than by stripping tags, so it reads like something a person would write.

  - A PREHEADER. The grey snippet Gmail shows next to the subject line. Left
    unset the client fills it with whatever text comes first — usually "View
    in browser" or the unsubscribe line. It is the second thing a recipient
    reads and it is free.

BRAND

Colours mirror src/index.css in the frontend (warm coral on cream). Kept as
plain hex rather than imported: this file has to run server-side with no
access to the stylesheet, and email cannot use CSS variables anyway.
"""

# ── Palette (mirrors ragfrontend/src/index.css) ───────────────────────────
CORAL = "#e88d7d"
CORAL_DARK = "#c46a5a"
PEACH = "#f8c8c4"
PAPER = "#fffef9"
CREAM = "#fdf8f3"
INK = "#3d3d3d"
INK_SOFT = "#6b6b6b"
INK_MUTED = "#9a9a9a"
LINE = "#efe6dd"

# Outfit is the product's face but will not load in Outlook or most desktop
# clients; the rest of the stack is what most recipients will actually see.
FONT = ("'Outfit',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
        "Helvetica,Arial,sans-serif")


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# Gmail shows roughly the first 60 characters of a subject on desktop and
# fewer on a phone. Anything past that is not short — it is invisible.
SUBJECT_MAX = 60


def trim_for_subject(text, budget):
    """
    Shorten a topic to fit a subject line, at a word boundary, with no ellipsis.

    Real plan topics are long: a production selection on 2026-09-12 produced
    "Contraindications and Safety Monitoring for Emergency Drugs" (59 chars),
    which made the subject "Your plan starts with Contraindications and Safety
    Monitoring for Emergency Drugs" — 81 characters, so the part naming the
    topic was cut off in the inbox and the hook was lost.

    No ellipsis on purpose. A trailing "..." in a subject line reads as a
    broken mail merge, which is the exact impression this whole package is
    built to avoid. Cutting at a word boundary reads like a choice.

    The BODY always keeps the full topic. Only the subject is rationed.
    """
    t = str(text or "").strip()
    if len(t) <= budget:
        return t

    window = t[:budget]
    nxt = t[budget]
    # Back up to the previous boundary ONLY when the cut landed inside a word.
    # If the next character is a space or punctuation, the last word in the
    # window is already whole — "Fluids, Electrolytes" should not lose
    # "Electrolytes" just because a comma follows it.
    if nxt.isalnum() and window[-1:].isalnum():
        # rsplit on a string with no space returns the whole string, which is
        # the right answer for a single word longer than the budget: a hard cut
        # beats returning something longer than the caller asked for.
        window = window.rsplit(" ", 1)[0]

    window = window.rstrip(" ,;:-–—/&([{")

    # Real topics carry parenthetical abbreviations — "Use of Personal
    # Protective Equipment (PPE)" trimmed to 38 characters ends on the open
    # bracket. Drop the orphaned group rather than ship "...Equipment (".
    for opener, closer in (("(", ")"), ("[", "]")):
        if window.count(opener) > window.count(closer):
            window = window[:window.rfind(opener)].rstrip(" ,;:-–—")

    return window or t[:budget].rstrip()


# ── Content blocks ────────────────────────────────────────────────────────
# Each returns (html, text) so the plain-text part is written alongside the
# markup and cannot drift out of sync with it.

def paragraph(text):
    html = (f"<p style=\"margin:0 0 16px;font:16px/1.65 {FONT};color:{INK}\">"
            f"{_esc(text)}</p>")
    return html, f"{text}\n\n"


def heading(text):
    html = (f"<h1 style=\"margin:0 0 14px;font:600 23px/1.3 {FONT};color:{INK}\">"
            f"{_esc(text)}</h1>")
    return html, f"{text}\n{'=' * len(str(text))}\n\n"


def stat_row(items):
    """
    A row of numbers — "83% recall / 40% prioritisation".

    Table-based and centred per cell rather than flex, because flex silently
    collapses to a vertical stack in Outlook and takes the comparison (which
    is the entire point of putting two numbers side by side) with it.
    """
    cells = ""
    for label, value, tone in items:
        colour = CORAL_DARK if tone == "bad" else INK
        cells += (
            f"<td align=\"center\" style=\"padding:14px 10px\">"
            f"<div style=\"font:700 27px/1.1 {FONT};color:{colour}\">{_esc(value)}</div>"
            f"<div style=\"font:12px/1.4 {FONT};color:{INK_MUTED};"
            f"text-transform:uppercase;letter-spacing:.07em;padding-top:5px\">"
            f"{_esc(label)}</div></td>"
        )
    html = (f"<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" "
            f"width=\"100%\" style=\"margin:6px 0 20px;background:{CREAM};"
            f"border:1px solid {LINE};border-radius:12px\"><tr>{cells}</tr></table>")
    text = " | ".join(f"{v} {l}" for l, v, _ in items) + "\n\n"
    return html, text


def bullets(items):
    lis = "".join(
        f"<li style=\"margin:0 0 8px;font:16px/1.6 {FONT};color:{INK}\">{_esc(i)}</li>"
        for i in items
    )
    html = f"<ul style=\"margin:0 0 18px;padding-left:20px\">{lis}</ul>"
    text = "".join(f"  - {i}\n" for i in items) + "\n"
    return html, text


def button(label, url):
    """
    A table with a background colour, not a <button> or a styled <div>.
    Outlook will not render either as a clickable block.
    """
    html = (
        f"<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" "
        f"style=\"margin:8px 0 6px\"><tr>"
        f"<td align=\"center\" bgcolor=\"{CORAL}\" style=\"border-radius:10px\">"
        f"<a href=\"{url}\" style=\"display:inline-block;padding:14px 30px;"
        f"font:600 16px/1 {FONT};color:#ffffff;text-decoration:none;border-radius:10px\">"
        f"{_esc(label)}</a></td></tr></table>"
    )
    return html, f"{label}: {url}\n\n"


def spacer(px=8):
    return f"<div style=\"height:{px}px\"></div>", ""


# ── Shell ─────────────────────────────────────────────────────────────────

def render(preheader, blocks, unsubscribe_url="#", postal_address=""):
    """
    Wrap content blocks in the branded shell.

    Returns (html, text). `blocks` is a list of (html, text) tuples from the
    helpers above.
    """
    body_html = "".join(b[0] for b in blocks)
    body_text = "".join(b[1] for b in blocks)

    addr = (f"<div style=\"padding-top:6px\">{_esc(postal_address)}</div>"
            if postal_address else
            "<div style=\"padding-top:6px;color:#c0392b\">"
            "[SET EMAIL_POSTAL_ADDRESS — required by CAN-SPAM]</div>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>NurseQuizAI</title>
</head>
<body style="margin:0;padding:0;background:{PAPER}">

<!-- Preheader: the grey snippet next to the subject in the inbox list.
     Hidden in the body, then padded with zero-width joiners so the client
     cannot pull footer text in after it to fill the space. -->
<div style="display:none;max-height:0;overflow:hidden;opacity:0;
            mso-hide:all;font-size:1px;line-height:1px;color:{PAPER}">
{_esc(preheader)}{'&#8204;&nbsp;' * 60}
</div>

<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"
       style="background:{PAPER}">
  <tr><td align="center" style="padding:32px 16px">

    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"
           style="max-width:560px;background:{PAPER};border:1px solid {LINE};
                  border-radius:16px;overflow:hidden">

      <!-- Brand bar. A coral rule rather than a logo image, so the identity
           survives images-off, which is how most first opens render. -->
      <tr><td style="height:4px;background:{CORAL};font-size:0;line-height:0">&nbsp;</td></tr>

      <tr><td style="padding:26px 30px 8px">
        <div style="font:700 17px/1 {FONT};color:{CORAL_DARK};letter-spacing:-.2px">
          NurseQuizAI
        </div>
      </td></tr>

      <tr><td style="padding:10px 30px 26px">
        {body_html}
      </td></tr>

      <tr><td style="padding:0 30px 26px">
        <div style="border-top:1px solid {LINE};padding-top:16px;
                    font:12px/1.6 {FONT};color:{INK_MUTED}">
          <a href="{unsubscribe_url}" style="color:{INK_MUTED}">Unsubscribe</a>
          &nbsp;·&nbsp; You're receiving this because you created a NurseQuizAI account.
          {addr}
        </div>
      </td></tr>

    </table>
  </td></tr>
</table>
</body></html>"""

    text = (f"{body_text}"
            f"---\nUnsubscribe: {unsubscribe_url}\n"
            f"You're receiving this because you created a NurseQuizAI account.\n"
            f"{postal_address}\n")

    return html, text


# ══════════════════════════════════════════════════════════════════════════
# CAMPAIGNS
#
# Each returns {subject, preheader, html, text}. The copy leads with the
# student's own numbers rather than with a request to come back: a generic
# re-engagement mail from an unfamiliar domain is exactly the shape spam
# filters — and people — are trained to discard. What we can say and a
# competitor cannot is what her own material showed about her.
# ══════════════════════════════════════════════════════════════════════════

def winback_gap(first_name=None, weak_topic="Fluid & Electrolytes",
                overall_pct=83, weak_pct=40, steps_left=4,
                resume_url="#", unsubscribe_url="#", postal_address=""):
    """
    Her own average against her own worst topic.

    NOTE ON WHAT THIS IS ALLOWED TO CLAIM. An earlier draft framed this as
    "recall vs judgement" — 83% on recall, 40% on prioritisation. That is the
    product's real thesis and it is a better hook, but question FORMAT is not
    persisted on quiz messages, so per-format accuracy cannot be computed per
    student. Sending it anyway would have meant printing a number we made up,
    to the exact people whose trust we are trying to win back.

    So it says the true version instead: her overall average against her
    weakest topic, both computed from studyPerformance. If `questionType` is
    ever persisted, the sharper framing becomes available honestly.
    """
    hi = f"{first_name}, you" if first_name else "You"
    blocks = [
        heading(f"{weak_topic} is costing you marks"),
        paragraph(f"{hi}'re averaging {overall_pct}% across everything you've "
                  f"studied — but {weak_pct}% on {weak_topic}. That gap is where "
                  f"the marks go."),
        stat_row([("Your average", f"{overall_pct}%", "good"),
                  (weak_topic, f"{weak_pct}%", "bad")]),
        paragraph(f"Your plan has {steps_left} steps left, starting there."),
        button("Pick up where you left off", resume_url),
    ]
    html, text = render(
        preheader=f"{weak_pct}% on {weak_topic} — {steps_left} steps left in your plan.",
        blocks=blocks, unsubscribe_url=unsubscribe_url, postal_address=postal_address)
    lead = "Your weak spot: "
    return {"subject": lead + trim_for_subject(weak_topic, SUBJECT_MAX - len(lead)),
            "preheader": f"{weak_pct}% on {weak_topic}", "html": html, "text": text}


def exam_countdown(days_away=3, weak_topic="Prioritisation", steps_left=4,
                   minutes_left=None, measured=True,
                   resume_url="#", unsubscribe_url="#", postal_address=""):
    """
    Only for students with a real exam date. Never guess this one.

    `measured` is what keeps the middle sentence honest, and it is not
    cosmetic. "X is where you're losing the most marks" is a FINDING: it
    requires answered questions, and only 28 of 348 reachable dormant students
    have enough of them (measured 2026-09-12). For everyone else the same
    sentence would be a number we invented, sent to someone sitting an exam
    this week — the worst possible person to be wrong to.

    So when there is no performance to draw on, the topic is introduced as
    what the plan starts with rather than as a diagnosis. Same urgency, same
    named topic, no claim we cannot support.

    `minutes_left` replaces the old hardcoded "About 15 minutes", which was
    printed regardless of how much plan was actually left.
    """
    when = ("today" if days_away == 0 else
            "tomorrow" if days_away == 1 else f"in {days_away} days")
    if measured:
        lead = (f"There's still time for the part that matters most. "
                f"{weak_topic} is where you're losing the most marks.")
        second = f"Starting with {weak_topic}"
    else:
        lead = (f"There's still time for the part that matters most. Your "
                f"plan opens on {weak_topic}.")
        second = f"Starting with {weak_topic}"

    facts = [f"{steps_left} steps left in your plan", second]
    if minutes_left:
        facts.append(f"About {minutes_left} minutes end to end")

    blocks = [
        heading(f"Your exam is {when}"),
        paragraph(lead),
        bullets(facts),
        button("Finish your plan", resume_url),
        paragraph("If you've already sat it — good luck, and tell us how it went."),
    ]
    html, text = render(
        preheader=f"{steps_left} steps left. Start with {weak_topic}.",
        blocks=blocks, unsubscribe_url=unsubscribe_url, postal_address=postal_address)
    return {"subject": f"Your exam is {when} — {steps_left} steps left",
            "preheader": f"Start with {weak_topic}", "html": html, "text": text}


def plan_unstarted(first_name=None, first_topic="Fluid & Electrolytes",
                   first_kind="quick check", first_minutes=4, steps_total=12,
                   days_since=9, resume_url="#", unsubscribe_url="#",
                   postal_address=""):
    """
    For the student whose plan was built and never opened.

    WHY THIS EXISTS SEPARATELY FROM winback_gap

    winback_gap contrasts her average with her worst topic, which needs answered
    questions. Measured against production 2026-09-12: of 348 dormant students
    who are free, opted-in and reachable, 228 have a plan with ZERO completed
    nodes and 0 have a partially completed one. Only 28 of the 348 clear
    winback's evidence bar. The campaign was aimed at a cohort of 28 while the
    cohort of 228 — the actual shape of dormancy here — had no message at all.

    Those 228 have no performance data and never will until they start. So this
    says the true thing that IS computable for them: the plan exists, it was
    built from their own material, and step one is short and named.

    WHY THE MINUTES NUMBER IS THE POINT

    Plan completion was measured at 6.4% with 32.5% of students never
    completing a single node — an activation problem, not a conversion one. The
    plan is not being rejected, it is being deferred, and a 12-step plan reads
    like an evening. So the ask in this message is ONE STEP, with its real
    length. `first_minutes` mirrors NODE_MINUTES in the frontend's
    planFormatting.js so the number here is the number she sees in the app.
    """
    hi = f"{first_name}, your" if first_name else "Your"
    when = (f"{days_since} days ago" if days_since and days_since > 1
            else "the other day")
    blocks = [
        heading(f"Step one: {first_topic}"),
        paragraph(f"{hi} study plan has been sitting ready since you built it "
                  f"{when}. It has {steps_total} steps, all built from the "
                  f"material you uploaded."),
        paragraph(f"You don't have to do the whole thing tonight. The first "
                  f"step is a {first_kind} on {first_topic}, and it takes "
                  f"about {first_minutes} minutes."),
        button(f"Start step one ({first_minutes} min)", resume_url),
    ]
    html, text = render(
        preheader=f"A {first_minutes}-minute {first_kind} on {first_topic}. "
                  f"{steps_total} steps waiting.",
        blocks=blocks, unsubscribe_url=unsubscribe_url, postal_address=postal_address)
    # "Your plan starts with " is 22 characters; the topic gets the rest.
    lead = "Your plan starts with "
    return {"subject": lead + trim_for_subject(first_topic, SUBJECT_MAX - len(lead)),
            "preheader": f"About {first_minutes} minutes",
            "html": html, "text": text}


def announcement(title="What's new", body=(), bullets_list=(),
                 cta_label=None, cta_url=None, preheader="", first_name=None,
                 sign_off=None, unsubscribe_url="#", postal_address=""):
    """
    A product update. The one template whose content is AUTHORED, not derived.

    Every other campaign here computes its copy from one student's record and
    refuses to send when the numbers are not there. An announcement is the
    opposite: the same words go to everybody, so nothing about it can be
    verified per-recipient and the honesty burden sits entirely on whoever
    writes it. That is why the content lives in email_announcements.py as
    reviewed code rather than in a Firestore document — an irreversible send to
    2,000 people should not be one careless edit away.

    `body` is a list of paragraph strings and `bullets_list` an optional list of
    short lines. Plain strings, not HTML: everything is escaped on the way in,
    so an announcement cannot accidentally ship broken markup to every client
    at once, and the plain-text part stays in sync for free.

    The greeting is optional because 878 of the 2,079 addressable accounts have
    no display name — 42%, measured 2026-09-12. A template that requires one
    produces "Hi ," for four recipients in ten.
    """
    blocks = [heading(title)]
    if first_name:
        blocks.append(paragraph(f"Hi {first_name},"))
    for para in body:
        blocks.append(paragraph(para))
    if bullets_list:
        blocks.append(bullets(list(bullets_list)))
    if cta_label and cta_url:
        blocks.append(button(cta_label, cta_url))
    if sign_off:
        blocks.append(paragraph(sign_off))

    html, text = render(
        preheader=preheader or title,
        blocks=blocks, unsubscribe_url=unsubscribe_url, postal_address=postal_address)
    return {"subject": title, "preheader": preheader or title,
            "html": html, "text": text}


CAMPAIGNS = {
    "winback_gap": winback_gap,
    "exam_countdown": exam_countdown,
    "plan_unstarted": plan_unstarted,
    "announcement": announcement,
}
