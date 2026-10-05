"""What the student told us matters, in her own words, applied to her plan.

WHY THIS EXISTS
───────────────
Students paste their instructor's material into the chat box: study guides,
objective lists, notes with the instructor's own flags in them. Measured
2026-09-28, about 25 students did; one Pro student, on the day of her exam,
pasted 27k characters containing "MUST know the bone & muscle landmarks. This
will be on the exam." Her chat quiz then asked nothing about landmarks.

The planner never saw any of it. It built topics from the uploaded files'
insights and the first 8,000 characters of the vectorstore, and the only thing
it read from the chat's practiceProfile was which question formats to leave
out. A plan that ignores "this will be on the exam" is the clearest possible
way to tell a student we were not listening.

WHAT COUNTS AS A FLAG
─────────────────────
Only literal text she gave us. Three sources, all CONFIDENCE "verified" in
course-intelligence terms (her upload, or something she told us herself):

  * exam cues inside pasted or uploaded notes ("MUST know X", "X will be on
    the exam", "high-yield", a study guide's "Know the ..." objective lines);
  * the chat's remembered emphasis ("more EKG rhythms").

The emphasis is a STEER, not an exam flag, and it can point down: "less
pharmacology" must never push pharmacology up. `emphasis_direction` reads the
direction before anything is promoted.

WHAT IT DOES TO THE PLAN
────────────────────────
Flagged topics move to the front of the planner's topic list, and topics she
asked for less of move to the back of the window. It does NOT add a new
PRIORITY_WEIGHTS signal (that table is a cross-repo contract with the
frontend), and it does not override a measured diagnostic gap: the diagnostic
weighting still orders by tier and only uses a flag to break ties inside one.

Nothing here invents a topic. A cue that names nothing in the curriculum still
reaches the path prompt verbatim, so the generator can teach it, but it is
never turned into a topic label we made up.

Pure: no Firebase, no network. The Firestore read lives in
student_emphasis_store.py.
"""
import re

from services.practice_profile import PASTE_MIN_CHARS, looks_like_pasted_material, outline_topics

# How much pasted material reaches topic extraction. The extraction prompt
# reads 8,000 characters in total; pasted notes get at most this share when
# there are uploads too, and all of it when there are none.
PASTED_SHARE_CHARS = 3500
MAX_PASTED_CHARS = 12000
MAX_CUES = 8
MAX_QUOTE_CHARS = 200

CONFIDENCE_VERIFIED = "verified"  # mirrors course_intelligence.CONFIDENCE_VERIFIED

# Phrases an instructor uses to say "this is examinable". Calibrated against
# the 46 long pastes in the 900 most recent chats (2026-10-01): these match the
# real flags ("MUST know the bone & muscle landmarks", "Landmarks (MUST know)",
# "drugs that will be on the exam") and not the near-misses that sit beside
# them ("MUST be administered again", "Providers may NOT know").
_STRONG_CUE = re.compile(
    r"\bmust\s+know\b"
    r"|\bneed\s+to\s+know\b"
    r"|\b(?:will|would|is\s+going\s+to)\s+be\s+(?:on|in)\s+(?:the|your|our|this)?\s*(?:exam|test|final|midterm|quiz|ati|hesi)\b"
    r"|\b(?:is|are)\s+on\s+the\s+(?:exam|test|final|midterm)\b"
    r"|\bwill\s+be\s+(?:tested|examined|asked)\b"
    r"|\bhigh[\s-]?yield\b"
    r"|\b(?:exam|test|final)\s+(?:will\s+)?(?:covers?|focus(?:es)?\s+on|includes?)\b",
    re.I,
)

# A study guide's objective line: "Know the components of ...", "Be able to
# describe ...". Only at the START of a line, after bullets and bold markers,
# so "Providers may NOT know that you are a student" never matches.
_OBJECTIVE_LINE = re.compile(r"^[\s*_#>\-•\d.)]*(?:know|be\s+able\s+to)\s+\w", re.I)

# A line that is nothing but the cue ("This will be on the exam.") names its
# subject in the line before it.
_BARE_CUE = re.compile(
    r"^[\s*_(]*(?:this|these|that|it|they)?\s*(?:will\s+be\s+on\s+the\s+\w+|is\s+on\s+the\s+\w+|must\s+know|"
    r"high[\s-]?yield|will\s+be\s+tested)[\s*_).!:]*$",
    re.I,
)

# Textbook prose uses the same phrases about someone else: "the patient and
# partner need to know that it is normal", "What does the provider need to
# know", "The nurse will be tested for HIV". A cue whose subject is a person in
# the scenario is clinical content, not an instructor's flag.
_THIRD_PARTY_SUBJECT = re.compile(
    r"\b(?:patients?|clients?|partners?|famil(?:y|ies)|providers?|physicians?|parents?|caregivers?|"
    r"nurses?|mothers?|staff|he|she|they)\b[\w\s,']{0,25}$",
    re.I,
)

_NEGATIVE_STEER = re.compile(
    r"^\s*(?:less|fewer|no\s+more|not\s+(?:so\s+)?much|stop|skip|drop|avoid|without|reduce)\b", re.I)
_POSITIVE_STEER = re.compile(r"^\s*(?:more|extra|focus\s+on|emphasi[sz]e|mostly|prioriti[sz]e)\s+(?:on\s+)?", re.I)

# Words that cannot make a topic match on their own: every nursing topic has
# them, so "Nursing care of the cardiac client" would otherwise match any cue.
_GENERIC = {
    "nursing", "nurse", "nurses", "care", "client", "clients", "patient", "patients", "management",
    "principles", "principle", "overview", "introduction", "intro", "basics", "basic", "concepts",
    "concept", "fundamentals", "review", "unit", "chapter", "module", "week", "lecture", "exam",
    "test", "final", "know", "must", "need", "will", "this", "that", "these", "those", "with",
    "from", "into", "about", "your", "their", "what", "when", "which", "able", "describe",
    "understand", "identify", "including", "related", "general", "important", "things", "topics",
    "content", "material", "information", "drug", "drugs", "medication", "medications",
    # Plain English that carries no subject.
    "the", "and", "are", "for", "you", "all", "any", "here", "there", "has", "have", "how", "why",
    "its", "was", "were", "more", "less", "also", "each", "both", "but", "not", "can", "may",
    "should", "very", "most", "much", "many", "some", "such", "than", "then", "them", "they",
    "our", "out", "who", "whom", "own", "per", "via", "like", "make", "sure", "being", "going",
}

_TOKEN = re.compile(r"[a-zA-ZÀ-ÿ]{3,}")


def _clean(line):
    return re.sub(r"[*_`#>|]+", "", str(line or "")).strip(" \t-•:")


def _tokens(text):
    out = set()
    for word in _TOKEN.findall(str(text or "").lower()):
        if word in _GENERIC:
            continue
        # Crude singular: "landmarks" and "landmark" are the same topic.
        out.add(word[:-1] if len(word) > 4 and word.endswith("s") else word)
    return out


def _lines(text):
    # Sentences as well as lines: pasted notes often run a flag into the
    # sentence before it on one line.
    for raw in str(text or "").splitlines():
        for part in re.split(r"(?<=[.!?])\s+", raw):
            if part.strip():
                yield part.strip()


def _window(text, at):
    """At most MAX_QUOTE_CHARS of `text` around position `at`, cut on word
    boundaries. Notes pasted from a PDF often arrive as one enormous line, and
    quoting its first 200 characters quotes something other than the flag."""
    if len(text) <= MAX_QUOTE_CHARS:
        return text
    start = max(0, at - MAX_QUOTE_CHARS // 3)
    end = min(len(text), start + MAX_QUOTE_CHARS)
    snippet = text[start:end]
    if start > 0:
        snippet = snippet.split(" ", 1)[-1]
    if end < len(text):
        snippet = snippet.rsplit(" ", 1)[0]
    return snippet.strip()


def exam_cues(text, source="pasted_notes", limit=MAX_CUES):
    """Lines of her material that say "this is on the exam", verbatim.

    Returns [{"quote", "source", "strength"}], deduplicated, at most `limit`;
    strength is "explicit" for an instructor's flag and "objective" for a
    study guide's "Know the ..." line. A bare "This will be on the exam." is
    joined to the line before it, because that line is what it is about.
    """
    strong, objectives, seen, previous = [], [], set(), ""
    for line in _lines(text):
        cleaned = _clean(line)
        if not cleaned:
            continue
        hit = next((m for m in _STRONG_CUE.finditer(cleaned)
                    if not _THIRD_PARTY_SUBJECT.search(cleaned[:m.start()])), None)
        if hit or _OBJECTIVE_LINE.match(line):
            quote = _window(cleaned, hit.start() if hit else 0)
            joined = bool(_BARE_CUE.match(cleaned) and previous)
            if joined:
                quote = _window(f"{previous} {cleaned}", len(previous))
                # "MUST know the landmarks. This will be on the exam." The
                # first sentence was already a cue; the joined quote replaces
                # it rather than standing beside it as a near-duplicate.
                if strong and strong[-1]["quote"] == _window(previous, 0):
                    seen.discard(strong.pop()["quote"].lower())
            key = quote.lower()
            # A quote with no subject words ("Here are the drugs that will be
            # on the exam.") is still kept for the prompt, where the list
            # after it gives it meaning; it simply never matches a topic.
            if key not in seen and len(quote) >= 12:
                seen.add(key)
                if hit:
                    strong.append({"quote": quote, "source": source, "strength": "explicit"})
                else:
                    objectives.append({"quote": quote, "source": source, "strength": "objective"})
        previous = cleaned
    # An instructor's explicit flag outranks a study guide's routine "Know
    # the ..." line when the cap forces a choice.
    return (strong + objectives)[:limit]


def emphasis_direction(emphasis):
    """("more" | "less", subject) for the chat's remembered steer, else None."""
    text = str(emphasis or "").strip()
    if not text:
        return None
    if _NEGATIVE_STEER.match(text):
        subject = _NEGATIVE_STEER.sub("", text, count=1).strip(" :-")
        return ("less", subject) if _tokens(subject) else None
    subject = _POSITIVE_STEER.sub("", text, count=1).strip(" :-")
    return ("more", subject) if _tokens(subject) else None


def pasted_material(user_messages, profile=None):
    """Her pasted notes, newest first, joined. NOT capped: cues are scanned
    across all of it (gather caps only what the prompts read). The first real
    case, a 27k-character paste, had its "MUST know" flag past the 12k mark.

    Read from the chat's own user messages rather than only from the
    practiceProfile: the profile records a paste only when the chat has no
    uploads and she then asks for a quiz, and clears it when she uploads.
    """
    chunks = []
    for content in reversed([m for m in (user_messages or []) if isinstance(m, str)]):
        if looks_like_pasted_material(content):
            chunks.append(content)
    saved = ((profile or {}).get("source") or {}).get("pastedText")
    if saved and not any(saved[:200] in c for c in chunks):
        chunks.append(saved)
    return "\n\n".join(chunks)


def gather(user_messages=None, profile=None, document_text=""):
    """Everything she told us matters, from the chat and her uploads.

    `user_messages` is a list of her message strings in chat order;
    `document_text` is text from her uploads (pasted-notes.txt included), which
    is scanned for cues but never treated as a paste.
    """
    pasted = pasted_material(user_messages, profile)
    cues = exam_cues(pasted, "pasted_notes", limit=MAX_CUES * 4)
    # Short chat messages can carry a flag too ("pharm and cardiac will be on
    # the final"); a long one was already read as material above.
    for content in (user_messages or []):
        if isinstance(content, str) and len(content) < PASTE_MIN_CHARS and _STRONG_CUE.search(content):
            cues += exam_cues(content, "chat")
    if document_text:
        cues += exam_cues(document_text, "uploaded_notes", limit=MAX_CUES * 4)
    deduped, seen = [], set()
    for cue in cues:
        if cue["quote"].lower() not in seen:
            seen.add(cue["quote"].lower())
            deduped.append(cue)
    # Explicit flags from every source before any objective line; stable, so
    # pasted notes still lead within each strength.
    deduped.sort(key=lambda c: 0 if c["strength"] == "explicit" else 1)
    steer = emphasis_direction((profile or {}).get("emphasis"))
    return {
        "pasted_text": pasted[:MAX_PASTED_CHARS],
        "cues": deduped[:MAX_CUES],
        "more": [steer[1]] if steer and steer[0] == "more" else [],
        "less": [steer[1]] if steer and steer[0] == "less" else [],
        "emphasis": (profile or {}).get("emphasis"),
    }


def is_empty(signals):
    return not signals or not (signals.get("pasted_text") or signals.get("cues")
                               or signals.get("more") or signals.get("less"))


def _matches(topic, phrase):
    """A topic matches a phrase when the phrase covers most of what the topic
    is specifically about. Generic nursing words never count, so a topic that
    is all generic words matches nothing."""
    topic_tokens = _tokens(topic)
    if not topic_tokens:
        return False
    shared = topic_tokens & _tokens(phrase)
    return len(shared) * 2 >= len(topic_tokens)


def apply_to_topics(topics, signals, limit=None):
    """Reorder the planner's topics by what she flagged.

    Flagged topics move to the front, in the order they already had; topics
    she asked for less of move to the back of the kept window. With no
    signals the list comes back exactly as given (cut to `limit`), so a chat
    with nothing pasted gets the plan it always got.

    Returns (topics, flags) where flags is [{topic, quote, source,
    confidence}] for every promoted topic: the evidence the plan can quote
    back to her.
    """
    topics = [t for t in (topics or []) if isinstance(t, str) and t.strip()]
    if is_empty(signals):
        return (topics[:limit] if limit else topics), []

    evidence = [(c["quote"], c["source"]) for c in signals.get("cues") or []]
    evidence += [(f"more {s}", "chat_emphasis") for s in signals.get("more") or []]

    flags, flagged = [], []
    for topic in topics:
        for quote, source in evidence:
            if _matches(topic, quote):
                flagged.append(topic)
                flags.append({"topic": topic, "quote": quote, "source": source,
                              "confidence": CONFIDENCE_VERIFIED})
                break

    ordered = flagged + [t for t in topics if t not in flagged]
    if limit:
        ordered = ordered[:limit]
    less = [t for t in ordered if t not in flagged and any(_matches(t, s) for s in signals.get("less") or [])]
    ordered = [t for t in ordered if t not in less] + less
    return ordered, flags


def fallback_topics(signals, limit=5):
    """Topics for a plan with no upload insights: the pasted notes' own
    headings. Literal, never invented; empty when the paste has no structure."""
    return outline_topics((signals or {}).get("pasted_text") or "")[:limit]


def extraction_material(document_content, signals, total=8000):
    """What the topic-extraction prompt reads: her uploads, plus her pasted
    notes. Pasted notes get a bounded share when there are uploads, so a long
    paste cannot crowd the files out, and everything when there are none."""
    pasted = (signals or {}).get("pasted_text") or ""
    document_content = document_content or ""
    if not pasted:
        return document_content[:total]
    if not document_content:
        return pasted[:total]
    share = min(len(pasted), PASTED_SHARE_CHARS)
    return (document_content[: total - share - 40]
            + "\n\nSTUDENT'S PASTED NOTES:\n" + pasted[:share])


def prompt_block(signals):
    """A briefing for the planner prompts. Her words, quoted, nothing more."""
    if is_empty(signals):
        return ""
    lines = []
    cues = signals.get("cues") or []
    if cues:
        lines.append("THE STUDENT'S OWN EXAM FLAGS (verbatim from her notes; treat these as on the exam):")
        lines += [f'- "{c["quote"]}"' for c in cues]
    if signals.get("more"):
        lines.append("SHE ASKED FOR MORE OF: " + "; ".join(signals["more"]))
    if signals.get("less"):
        lines.append("SHE ASKED FOR LESS OF: " + "; ".join(signals["less"]))
    if not lines:
        return ""
    lines.append("Cover every flagged subject that appears in her material, and put it early in the plan.")
    return "\n".join(lines)
