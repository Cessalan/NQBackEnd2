"""
The blueprint of one requested practice: which part of her material each
question comes from, and which of her requirements it serves.

WHY THIS EXISTS (2026-10-11)

A chat quiz used to choose its topics from whatever fit in front of the
concept picker: the 30 passages most similar to the router's topic string,
cut to 12,000 characters, of which the picker read the first 8,000 (with the
instructions taking the top of that). A student who uploaded two chapters
(126,000 characters, 257 analysed learning goals) asked for "a full exam on
both these chapters with medications questions"; most of both chapters was
never read, and a "targeted practice on Heart Failure with Preserved Ejection
Fraction" came back as staging and echo questions.

The material analysis (services/document_understanding) already reads every
character and keeps each learning goal with its chapter heading, the file it
is in and the passage around its evidence. So a request is planned against
that full catalogue:

- SELECTION: which goals are in scope (all of them unless she narrowed it)
  and which ones can serve each focus she named ("medication classes"). Code
  does what it can (selected files, a targeted topic named in the goals' own
  headings); a model call does only the semantic part and is validated here.
- SCHEDULING: weighted fair queuing over three levels: her focus shares, then
  files, then goals, plus the format mix. It is PREFIX-STABLE: slot i is the
  same whatever the total, so an exam of 50 is the first 50 slots of an exam
  of 100, batches of 10 can be generated one at a time from any index, and a
  change of size never re-plans what she has already answered. Every window
  of ten slots carries the plan's proportions, so a student who stops at 10
  has still practised both chapters and the share of medications she asked
  for.

Pure (no LLM, no Firebase); pinned by tests/test_practice_plan.py against her
real material's goal catalogue.
"""
import hashlib
import json
import re
from collections import Counter

# 2: a topic named in the request is a scope (needs_model_selection), so v1
# plans built over all her material for a named topic are not reused.
PLAN_VERSION = 2
MAX_SELECTION_GOALS = 400
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "of", "for", "with", "in", "on", "a", "an", "to", "patient", "patients", "nursing",
         "management", "care", "create", "short", "targeted", "practice", "questions", "question", "quiz"}


def partly_readable(analysis):
    """True for a file with readable goals but uncertain visual pages.

    document_understanding.apply_readability marks a file not practice-ready
    when any image page has little native text (a cartoon title slide is
    enough), but it has already removed every goal overlapping the uncertain
    transcription. The rest of the file is real text and safe to plan from.
    An UNREADABLE marker is different: the source itself could not be read.
    """
    warnings = " ".join(str(w) for w in analysis.get("warnings", []))
    return analysis.get("practiceReady") is False and bool(analysis.get("goals")) and "UNREADABLE" not in warnings


def compact_goals(analyses, files=None):
    """Every usable goal in document order: id, file, chapter, topic, outcome, reasoning, context.

    `files` limits the catalogue to the files the practice is about. A file
    that could not be read contributes nothing; a partly readable one
    contributes its readable goals (see partly_readable).
    """
    out = []
    allowed = set(files or [])
    for analysis in analyses or []:
        filename = analysis.get("filename")
        if allowed and filename not in allowed:
            continue
        if analysis.get("practiceReady") is False and not partly_readable(analysis):
            continue
        for goal in analysis.get("goals", []):
            if not (goal.get("id") and goal.get("topic") and goal.get("outcome")):
                continue
            out.append({"id": goal["id"], "file": filename,
                        "chapter": goal.get("mainTopic") or goal["topic"],
                        "topic": goal["topic"], "outcome": goal["outcome"],
                        "reasoning": goal.get("reasoning"),
                        "context": (goal.get("context") or goal.get("evidence") or {}).get("quote", "")})
    return out[:MAX_SELECTION_GOALS]


def _key(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _words(text):
    return {w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP and len(w) > 2}


def match_topic(goals, phrase):
    """Goals whose own chapter or topic names a targeted topic, or [].

    Code's half of selection for a "targeted practice on X" button: the button
    copies a heading from her material, so the heading usually matches. Every
    distinctive word of the phrase must appear in the goal's chapter + topic.
    """
    wanted = _words(phrase)
    if not wanted:
        return []
    exact = [g["id"] for g in goals if wanted <= _words(f"{g['chapter']} {g['topic']}")]
    return exact


def selection_key(goals, request_items, scope_text, focus_topic):
    """Cache key for a selection: the catalogue and what she asked for."""
    data = {"goals": [g["id"] for g in goals], "focus_topic": focus_topic or None, "scope": scope_text or None,
            "items": [[i.get("id"), i.get("instruction")] for i in request_items or []
                      if i.get("kind") in ("scope", "focus")], "version": PLAN_VERSION}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]


# Scope words that mean "all of it", so no selection is needed.
_WHOLE_MATERIAL = re.compile(
    r"^\s*(?:(?:on|from|about)\s+)?(?:all\s+(?:of\s+)?)?(?:the|my|her)?\s*(?:uploaded\s+|whole\s+|entire\s+)?"
    r"(?:material|materials|notes|documents?|files?|uploads?|everything|course)\s*$", re.I)


def names_a_scope(scope_text):
    """True when the topic she (or the post-upload button) named is narrower than everything."""
    return bool(scope_text and scope_text.strip()) and not _WHOLE_MATERIAL.match(scope_text)


def needs_model_selection(request_items, focus_topic_matched, scope_text=None):
    """True when choosing goals needs reading, not matching.

    A focus ("medication classes") always does. A scope she named in words
    does, unless it is the derived whole-material scope. A targeted topic that
    code already matched does not. Neither does the topic the quiz was called
    with, unless it names something narrower than all her material: "a quiz
    about kidney disorders" over two files was planned over both and four of
    five questions came from the other file (2026-10-17), and a student who
    clicked the knowledge quiz for five named topics got endotoxin shock.
    """
    for item in request_items or []:
        if item.get("kind") == "focus":
            return True
        if item.get("kind") == "scope" and not item.get("derived") and not focus_topic_matched:
            return True
    return not focus_topic_matched and names_a_scope(scope_text)


def validate_selection(raw, goals, focus_items):
    """The model's selection made safe: known goals only, every focus accounted for.

    Returns {inScope: [ids], serves: {focusId: [ids]}, gaps: [{id, kind, file?}]}.

    Gaps are COMPUTED from what was selected, never written by the model: a
    model asked for gap notes invented one on the first live run ("your heart
    failure file gives little on the other classes", which was false), and a
    false caveat on the loading screen costs the trust the screen is for.
      - kind "none": no goal in her material serves this focus;
      - kind "file": a file she uploaded has none of it (others do);
      - kind "scope": nothing matched the scope she named, so the whole
        material is used instead of an empty exam.
    A focus no goal can serve is shown to her as a gap, never filled with
    something else.
    """
    raw = raw if isinstance(raw, dict) else {}
    alias = {f"g{index}": goal["id"] for index, goal in enumerate(goals)}
    known = {g["id"] for g in goals}
    file_of = {g["id"]: g["file"] for g in goals}

    def ids(values):
        out = []
        for value in values if isinstance(values, list) else []:
            gid = alias.get(str(value), str(value))
            if gid in known and gid not in out:
                out.append(gid)
        return out

    gaps = []
    in_scope = list(known) if raw.get("all_in_scope") is True else ids(raw.get("in_scope"))
    if not in_scope:
        in_scope = [g["id"] for g in goals]
        if raw and raw.get("all_in_scope") is not True:
            gaps.append({"id": "scope", "kind": "scope"})
    in_scope_set = set(in_scope)
    in_scope = [g["id"] for g in goals if g["id"] in in_scope_set]
    scope_files = list(dict.fromkeys(file_of[gid] for gid in in_scope))
    serves = {}
    by_focus = {}
    # Models echo a focus id loosely ("medication-classes" for
    # "focus:medication-classes", or the label). Each spelling maps back to
    # one of her focus items; an entry that maps to none is ignored.
    spellings = {}
    for item in focus_items or []:
        for spelling in (item["id"], item["id"].split(":", 1)[-1], item.get("label")):
            if spelling:
                spellings[_key(spelling)] = item["id"]
    entries = [e for e in raw.get("serves") or [] if isinstance(e, dict) and e.get("focus")]
    for entry in entries:
        focus_id = spellings.get(_key(entry["focus"]))
        if not focus_id and len(focus_items or []) == 1 and len(entries) == 1:
            focus_id = focus_items[0]["id"]
        if focus_id:
            by_focus.setdefault(focus_id, []).extend(entry.get("goals") or [])
    for item in focus_items or []:
        chosen = [gid for gid in ids(by_focus.get(item["id"], [])) if gid in in_scope_set]
        serves[item["id"]] = chosen
        if not chosen:
            gaps.append({"id": item["id"], "kind": "none"})
            continue
        covered = {file_of[gid] for gid in chosen}
        for filename in scope_files:
            if filename not in covered:
                gaps.append({"id": item["id"], "kind": "file", "file": filename})
    return {"inScope": in_scope, "serves": serves, "gaps": gaps}


def schedule(goals, selection, focus_items, formats, start, end):
    """Slots [start, end) of the plan. Prefix-stable: computed from slot 0 every time.

    Levels, each picking the least-served candidate relative to its weight:
      1. bucket: a focus she named (its share) or the core of her material;
      2. file: so both chapters stay in proportion inside every bucket;
      3. heading: so a file's questions spread over all of its sections
         instead of the first few goals in document order;
      4. goal: the least-used goal under that heading, in document order.
    The format rotates independently. A goal used again gets a new `angle`
    so its next question asks something different about it.
    """
    by_id = {g["id"]: g for g in goals}
    if not by_id:
        return []
    in_scope = [gid for gid in (selection or {}).get("inScope", []) if gid in by_id] or [g["id"] for g in goals]
    serves = (selection or {}).get("serves", {})
    buckets = []
    focus_total = 0.0
    for item in focus_items or []:
        pool = [gid for gid in serves.get(item["id"], []) if gid in by_id]
        if pool:
            share = float(item.get("share") or 0.3)
            buckets.append({"id": item["id"], "share": share, "pool": pool})
            focus_total += share
    core_share = max(0.0, 1.0 - focus_total)
    if core_share > 0 or not buckets:
        buckets.append({"id": None, "share": core_share or 1.0, "pool": in_scope})
    formats = [f for f in (formats or ["mcq"]) if f] or ["mcq"]
    order = {gid: index for index, gid in enumerate(g["id"] for g in goals)}
    file_weight = Counter(by_id[gid]["file"] for gid in in_scope)

    file_order = list(file_weight)

    used_bucket, used_file, used_chapter, used_goal, used_format = Counter(), Counter(), Counter(), Counter(), Counter()
    slots = []
    for index in range(max(0, end)):
        bucket = min(buckets, key=lambda b: (used_bucket[b["id"]] / b["share"], buckets.index(b)))
        files = sorted({by_id[gid]["file"] for gid in bucket["pool"]},
                       key=lambda f: file_order.index(f) if f in file_order else len(file_order))
        filename = min(files, key=lambda f: (used_file[f] / max(1, file_weight.get(f, 1)), files.index(f)))
        candidates = [gid for gid in bucket["pool"] if by_id[gid]["file"] == filename]
        chapters = {}
        for gid in candidates:
            chapters.setdefault(by_id[gid]["chapter"], []).append(gid)
        chapter_order = list(chapters)
        chapter = min(chapter_order, key=lambda c: (used_chapter[(filename, c)] / len(chapters[c]),
                                                    chapter_order.index(c)))
        gid = min(chapters[chapter], key=lambda g: (used_goal[g], order[g]))
        fmt = min(formats, key=lambda f: (used_format[f], formats.index(f)))
        if index >= start:
            goal = by_id[gid]
            serving = [b["id"] for b in buckets if b["id"] and gid in b["pool"]]
            slots.append({"index": index, "goalId": gid, "file": goal["file"], "chapter": goal["chapter"],
                          "topic": goal["topic"], "outcome": goal["outcome"], "format": fmt,
                          "reasoning": goal.get("reasoning"),
                          "bucket": bucket["id"], "serves": serving, "angle": used_goal[gid]})
        used_bucket[bucket["id"]] += 1
        used_file[filename] += 1
        used_chapter[(filename, chapter)] += 1
        used_goal[gid] += 1
        used_format[fmt] += 1
    return slots


def coverage(slots, focus_items):
    """What a total delivers: questions per file, per chapter and per focus."""
    files = Counter(s["file"] for s in slots)
    focus = {item["id"]: sum(1 for s in slots if s["bucket"] == item["id"]) for item in focus_items or []}
    chapters = Counter(s["chapter"] for s in slots)
    return {"files": dict(files), "focus": focus, "chapters": len(chapters)}


# The question a slot asks, from the kind of learning goal behind it. Ten
# "hard" questions of 2026-10-11 were all "Which action should the nurse take
# FIRST?"; a test of judgment also asks which finding matters and whether care
# worked. Two shapes per kind, alternated by slot so a batch varies.
STEMS = {
    "priority": ("which action the nurse takes first for this client",
                 "which of several clients the nurse sees first"),
    "evaluation": ("which finding shows the treatment or intervention is working",
                   "which client statement shows the teaching needs to be reinforced"),
    "application": ("which nursing action is indicated for this client now",
                    "which prescribed order the nurse should question before carrying it out"),
    "recognition": ("which finding requires follow-up",
                    "which finding the nurse reports to the provider immediately"),
    "recall": ("which statement about this client's care is accurate",
               "which explanation the nurse gives the client"),
}
# Under "hard", a recall goal is asked as a decision about it.
HARD_UPGRADE = {"recall": "recognition", None: "application"}


def slot_concept(slot, hard=False):
    """The seed a question writer is given for one slot, in the material's words."""
    angle = slot.get("angle") or 0
    seed = f"{slot['topic']}: {slot['outcome']}"
    reasoning = slot.get("reasoning") if slot.get("reasoning") in STEMS else None
    if hard:
        reasoning = HARD_UPGRADE.get(reasoning, reasoning)
    if reasoning and slot.get("format") in (None, "mcq"):
        seed += f". Ask {STEMS[reasoning][slot.get('index', 0) % 2]}"
    if angle:
        seed += (f" (this goal was already tested {angle} time{'s' if angle > 1 else ''}; ask about a "
                 "different aspect, finding or decision than a first question would)")
    return seed
