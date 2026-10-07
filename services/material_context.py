"""Turn the document analysis into context the proven quiz generator can use.

Why this exists (2026-10-05): the source-based rewrite in material_practice.py
replaced the old generator and produced list-regurgitation questions, because
its verifier rewarded verbatim copying. The old generator writes better
questions, but it never saw three things the student gave us:

- example questions she uploaded (style, command words, distractor pattern),
- explicit instructions in her prompt or practice profile,
- "MUST know / on the exam / excluded" markers inside her documents.

document_understanding.py already extracts all three as `examples`, `signals`
and `goals`, each tied to a verbatim quote. This module renders them as a
brief that is prepended to the generator's content context, so the question
writer follows them without changing how it writes.

Everything here is pure and tested. Nothing is ever invented: every line
comes from the analysis records or the student's own words.
"""

MAX_EXAMPLES = 4
MAX_SIGNALS = 12
MAX_GOALS = 25
MAX_INSTRUCTION_CHARS = 1500

# Analysis example formats -> old generator question types. true_false has no
# generator of its own; a two-option MCQ is what the renderer shows anyway.
FORMAT_TO_TYPE = {'mcq': 'mcq', 'sata': 'sata', 'case': 'casestudy', 'true_false': 'mcq'}

SIGNAL_LABELS = {
    'emphasis': 'Marked as important',
    'exam_instruction': 'Exam instruction',
    'exclusion': 'Excluded from the exam',
    'format_instruction': 'Format instruction',
}


def _clip(text, limit):
    text = ' '.join(str(text or '').split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _quote(record):
    return _clip((record.get('evidence') or {}).get('quote', ''), 220)


def example_formats(analyses):
    """Question types the student's own example questions use, most common first."""
    counts = {}
    for analysis in analyses or []:
        for example in analysis.get('examples', []):
            kind = FORMAT_TO_TYPE.get(example.get('format'))
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
    return [kind for kind, _ in sorted(counts.items(), key=lambda item: -item[1])]


def excluded_subjects(analyses):
    """Subjects the documents say are NOT on the exam. Only from explicit markers."""
    subjects = []
    for analysis in analyses or []:
        for signal in analysis.get('signals', []):
            if signal.get('kind') == 'exclusion' and signal.get('subject'):
                subjects.append(_clip(signal['subject'], 120))
    return list(dict.fromkeys(subjects))


def _render_example(example, index):
    lines = [f"Example {index}:"]
    stem = example.get('stem') or (example.get('evidence') or {}).get('quote', '')
    if stem:
        lines.append(f"  Q: {_clip(stem, 400)}")
    options = example.get('options') or []
    for position, option in enumerate(options[:7]):
        lines.append(f"  {chr(65 + position)}) {_clip(option, 160)}")
    indices = example.get('correctIndices') or []
    if indices and options:
        letters = ', '.join(chr(65 + i) for i in indices if 0 <= i < len(options))
        if letters:
            lines.append(f"  Answer: {letters}")
    if example.get('wording'):
        lines.append(f"  Style: {_clip(example['wording'], 200)}")
    return '\n'.join(lines)


def material_brief(analyses, profile=None, user_prompt=None, *, match_examples=True):
    """Render instructions, emphasis markers, objectives and examples as one brief.

    Returns '' when there is nothing specific to say, so callers can prepend
    it unconditionally.
    """
    profile = profile or {}
    analyses = analyses or []
    sections = []

    instructions = []
    for text in (profile.get('generationInstructions'), user_prompt, profile.get('emphasis')):
        text = _clip(text, MAX_INSTRUCTION_CHARS)
        if text and text not in instructions:
            instructions.append(text)
    if instructions:
        sections.append("STUDENT'S OWN INSTRUCTIONS (follow these exactly; they outrank everything below):\n"
                        + '\n'.join(f"- {text}" for text in instructions))

    signals = [s for a in analyses for s in a.get('signals', []) if s.get('kind') in SIGNAL_LABELS]
    excluded = excluded_subjects(analyses)
    marked = [s for s in signals if s.get('kind') != 'exclusion'][:MAX_SIGNALS]
    if marked:
        lines = []
        for signal in marked:
            subject = _clip(signal.get('subject', ''), 120)
            quote = _quote(signal)
            line = f"- {SIGNAL_LABELS[signal['kind']]}: {subject}" if subject else f"- {SIGNAL_LABELS[signal['kind']]}"
            if quote:
                line += f' — the document says: "{quote}"'
            lines.append(line)
        sections.append("WHAT THE DOCUMENTS MARK AS IMPORTANT (weight questions toward these):\n" + '\n'.join(lines))
    if excluded:
        sections.append("NOT ON THE EXAM (do not ask about these):\n" + '\n'.join(f"- {s}" for s in excluded))

    goals = [g for a in analyses for g in a.get('goals', [])]
    if goals:
        # Grouped under the document's own headings so the generator spreads
        # questions across chapters instead of mining one subtopic.
        seen, lines, headings = set(), [], {}
        for goal in goals:
            key = (goal.get('topic'), goal.get('outcome'))
            if key in seen or not goal.get('outcome'):
                continue
            seen.add(key)
            heading = goal.get('mainTopic') or goal.get('topic') or ''
            headings.setdefault(heading, []).append(
                f"  - {_clip(goal.get('topic', ''), 80)}: {_clip(goal['outcome'], 160)}")
        count = 0
        for heading, items in headings.items():
            lines.append(f"- {_clip(heading, 100)}")
            for item in items:
                if count >= MAX_GOALS:
                    break
                lines.append(item)
                count += 1
            if count >= MAX_GOALS:
                break
        sections.append("LEARNING OBJECTIVES FROM THE DOCUMENTS, BY MAIN TOPIC (cover these, not incidental details):\n" + '\n'.join(lines))

    examples = [e for a in analyses for e in a.get('examples', [])]
    if match_examples and examples:
        rendered = [_render_example(e, i + 1) for i, e in enumerate(examples[:MAX_EXAMPLES])]
        sections.append("THE STUDENT'S OWN EXAMPLE QUESTIONS (match their style: command words, stem length, "
                        "option length and distractor pattern. Never reuse an example's question or answer):\n"
                        + '\n'.join(rendered))

    if not sections:
        return ''
    return "MATERIAL BRIEF (built from the student's uploads and words):\n\n" + '\n\n'.join(sections)
