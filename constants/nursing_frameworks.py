# -*- coding: utf-8 -*-
"""
Nursing Frameworks — the closed set of classification skills we can test.
=========================================================================

WHY THIS EXISTS
---------------

Study plans are built topic-first: a document about vital signs produces quizzes
about vital signs. That covers "do you know facts about this subject" and misses
an entire genre of nursing-school exam item — the ones that test a *skill applied
across every subject*:

    "The nurse and patient agree the patient will ambulate 50 feet by Friday.
     Which step of the nursing process is the nurse in?"

The clinical content there is trivial; the skill is classifying an action into a
framework. A topic-first planner cannot produce that item, because the item is
not about a topic.

This was surfaced by a real student (uid egQ5oJwFxaRSqCHFhn7l77PpRv43, exam
27 Aug 2026). She uploaded `The Nursing Process-Hill (1).pdf`, the app correctly
extracted "The Nursing Process" as topic #1 — and then none of her 40 study nodes
covered it. Her exam asked about it roughly five times. Her post-exam debrief asked
us, in as many words, to "generate scenario questions that describe a nurse's
actions and ask which step of the nursing process is being carried out, covering
all steps including planning."

WHY A CLOSED SET, AND WHY IT IS A TABLE NOT AN ENUM
---------------------------------------------------

Asking a model to name "the framework in this document" open-endedly returns
plausible-sounding labels with nothing behind them. To generate an item you need
the *categories* — they are the answer options — and to mark it you need them to
be a finite list. So a framework is only useful here if we already know its
categories, which means the set has to be closed and curated.

A closed set also gives free, deterministic detection: each entry carries the
vocabulary that proves it is present in the student's own material. Nothing is
tested that the document does not teach, which is the same rule the rest of the
app follows — everything comes from her uploads.

`detect_frameworks()` deliberately returns an EMPTY LIST rather than a guess.
Most documents teach no framework, and that is the honest answer.

TWO KINDS OF TERM, AND WHY THEY ARE SEPARATE
---------------------------------------------

`categories` are display strings — the answer options a student picks between.
`detect_terms` are the strings we search the document for. They are NOT the same
list, because several categories are words that appear in almost every nursing
document. "Assessment" alone proves nothing (it matched 39.9% of all study nodes
in a Sep 2026 scan); "nursing diagnosis" and "evaluation of outcomes" are
distinctive. Detection uses the distinctive vocabulary; generation uses the clean
category names.

DETECTION RULE
--------------

A framework is present if EITHER:
  - an `anchor` phrase appears (the document names the framework outright), or
  - at least `min_terms` distinct `detect_terms` appear (it teaches the categories
    without naming the framework).

The anchor path reports confidence "named", the term path "inferred". `min_terms`
is set high on purpose — a false positive means generating questions on a
framework the student's course never covered, which is worse than generating none.

USAGE
-----

    from constants.nursing_frameworks import detect_frameworks, get_framework

    found = detect_frameworks(document_text)
    # [{'id': 'nursing_process', 'confidence': 'named', 'matched': [...], ...}]

    fw = get_framework('nursing_process')
    fw['categories']   # answer options
    fw['item_shape']   # how to phrase the question
    fw['confusable']   # which pairs to weight the distractors toward

IMPORTANT: detection must run over the FULL document text, not the sampled
excerpt used by `extract_file_insights_from_text` in main.py. That function reads
three random 1,000-character windows; a framework is typically defined once, in
one place, and random windows miss it. This module is pure and cheap enough to run
over everything.
"""

from typing import Dict, List, Optional, Any
import re


# ============================================
# ITEM SHAPES
# ============================================
# How a question built on this framework is phrased. The three shapes need
# genuinely different stems, so the generator must not treat them alike.
#
#   classify   — given one action/finding/statement, name its category
#   prioritize — given several options, say which comes first (framework = the
#                ordering rule, not a label set)
#   assign     — given a task, say who may perform it

SHAPE_CLASSIFY = "classify"
SHAPE_PRIORITIZE = "prioritize"
SHAPE_ASSIGN = "assign"


# ============================================
# THE REGISTRY
# ============================================
# Ordered roughly by how commonly the framework carries exam items in a US
# nursing curriculum. Tier 1 entries are near-universal in Fundamentals, which
# is where most of our users are (Semester 1–3 = 1,122 of 1,965 accounts).

FRAMEWORKS: Dict[str, Dict[str, Any]] = {

    # ---------------------------------------------------------------
    # TIER 1 — Fundamentals staples
    # ---------------------------------------------------------------

    "nursing_process": {
        "name": "The Nursing Process",
        "item_shape": SHAPE_CLASSIFY,
        "categories": [
            "Assessment", "Diagnosis", "Planning", "Implementation", "Evaluation",
        ],
        # NCLEX-prep framing. The NCLEX itself does not ask students to name the
        # ADPIE step; since the Next Generation NCLEX (Apr 2023) clinical judgment
        # is measured with the NCSBN model below. The two map near-cleanly, so one
        # generated item can be relabelled for the NCLEX cohort.
        "categories_nclex": [
            "Recognize Cues", "Analyze Cues", "Prioritize Hypotheses",
            "Generate Solutions", "Take Actions", "Evaluate Outcomes",
        ],
        "anchors": ["nursing process", "adpie", "adopie"],
        "detect_terms": [
            "nursing diagnosis", "nursing diagnoses", "planning phase",
            "implementation phase", "evaluation phase", "assessment phase",
            "expected outcome", "goal setting",
        ],
        "min_terms": 4,
        # Planning is the step students lose points on: it sits between naming
        # the problem and doing the thing, and it is "only" goal-setting.
        "confusable": [
            ("Planning", "Diagnosis"),
            ("Planning", "Implementation"),
            ("Assessment", "Evaluation"),
        ],
        "stem_subject": "a nurse's action",
    },

    "data_type": {
        "name": "Subjective vs Objective Data",
        "item_shape": SHAPE_CLASSIFY,
        "categories": ["Subjective", "Objective"],
        "anchors": [
            "subjective and objective", "objective and subjective",
            "subjective data", "objective data",
        ],
        "detect_terms": [
            "subjective data", "objective data", "subjective finding",
            "objective finding", "what the patient reports", "observable and measurable",
        ],
        "min_terms": 2,
        "confusable": [("Subjective", "Objective")],
        "stem_subject": "a piece of assessment data",
    },

    "abc_priority": {
        "name": "ABC / ABCDE Priority",
        "item_shape": SHAPE_PRIORITIZE,
        "categories": ["Airway", "Breathing", "Circulation", "Disability", "Exposure"],
        "anchors": [
            "abcde", "airway, breathing, circulation", "abcs of",
            "airway breathing circulation",
        ],
        "detect_terms": [
            "patent airway", "airway obstruction", "breathing", "circulation",
            "disability", "exposure", "primary survey",
        ],
        "min_terms": 5,
        "confusable": [("Airway", "Breathing"), ("Breathing", "Circulation")],
        "stem_subject": "several patients or findings competing for attention",
    },

    "maslow": {
        "name": "Maslow's Hierarchy of Needs",
        "item_shape": SHAPE_PRIORITIZE,
        "categories": [
            "Physiological", "Safety and Security", "Love and Belonging",
            "Esteem", "Self-actualization",
        ],
        "anchors": ["maslow", "hierarchy of needs"],
        "detect_terms": [
            "physiological need", "safety and security", "love and belonging",
            "self-actualization", "self actualization", "esteem need",
        ],
        "min_terms": 3,
        "confusable": [
            ("Physiological", "Safety and Security"),
            ("Love and Belonging", "Esteem"),
        ],
        "stem_subject": "several patient needs competing for priority",
    },

    "therapeutic_communication": {
        "name": "Therapeutic vs Non-therapeutic Communication",
        "item_shape": SHAPE_CLASSIFY,
        "categories": ["Therapeutic", "Non-therapeutic"],
        "anchors": [
            "therapeutic communication", "nontherapeutic", "non-therapeutic",
            "therapeutic response",
        ],
        "detect_terms": [
            "active listening", "open-ended question", "false reassurance",
            "giving advice", "changing the subject", "reflecting", "silence",
            "clarifying",
        ],
        "min_terms": 4,
        "confusable": [("Therapeutic", "Non-therapeutic")],
        "stem_subject": "a nurse's verbal response to a patient",
    },

    "prevention_levels": {
        "name": "Levels of Prevention",
        "item_shape": SHAPE_CLASSIFY,
        "categories": ["Primary", "Secondary", "Tertiary"],
        "anchors": [
            "levels of prevention", "primary, secondary, and tertiary",
            "primary prevention", "tertiary prevention",
        ],
        "detect_terms": [
            "primary prevention", "secondary prevention", "tertiary prevention",
            "health promotion", "early detection", "rehabilitation",
        ],
        "min_terms": 3,
        # Screening (secondary) is routinely mistaken for prevention (primary).
        "confusable": [("Primary", "Secondary"), ("Secondary", "Tertiary")],
        "stem_subject": "a nursing intervention or public-health activity",
    },

    # ---------------------------------------------------------------
    # TIER 2 — very common, slightly more specialised
    # ---------------------------------------------------------------

    "isolation_precautions": {
        "name": "Transmission-Based Precautions",
        "item_shape": SHAPE_CLASSIFY,
        "categories": ["Standard", "Contact", "Droplet", "Airborne"],
        "anchors": [
            "transmission-based precautions", "transmission based precautions",
            "isolation precautions", "standard precautions",
        ],
        "detect_terms": [
            "contact precautions", "droplet precautions", "airborne precautions",
            "standard precautions", "negative pressure", "n95",
            "personal protective equipment",
        ],
        "min_terms": 3,
        "confusable": [("Droplet", "Airborne"), ("Standard", "Contact")],
        "stem_subject": "a patient with a specific infection or presentation",
    },

    "delegation_scope": {
        "name": "Delegation and Scope of Practice",
        "item_shape": SHAPE_ASSIGN,
        "categories": [
            "Registered Nurse (RN)",
            "Licensed Practical/Vocational Nurse (LPN/LVN)",
            "Unlicensed Assistive Personnel (UAP)",
        ],
        "anchors": [
            "scope of practice", "delegation", "five rights of delegation",
            "rights of delegation",
        ],
        "detect_terms": [
            "unlicensed assistive personnel", "assistive personnel", "delegate",
            "lpn", "lvn", "scope of practice", "nursing assistant",
        ],
        "min_terms": 3,
        # The line that actually gets tested: assessment, teaching, evaluation and
        # unstable patients stay with the RN.
        "confusable": [
            ("Registered Nurse (RN)", "Licensed Practical/Vocational Nurse (LPN/LVN)"),
            ("Licensed Practical/Vocational Nurse (LPN/LVN)",
             "Unlicensed Assistive Personnel (UAP)"),
        ],
        "stem_subject": "a task that needs assigning",
    },

    "acid_base": {
        "name": "Acid–Base Imbalance",
        "item_shape": SHAPE_CLASSIFY,
        "categories": [
            "Respiratory Acidosis", "Respiratory Alkalosis",
            "Metabolic Acidosis", "Metabolic Alkalosis",
        ],
        "anchors": [
            "acid-base", "acid base balance", "arterial blood gas",
            "respiratory acidosis", "metabolic acidosis",
        ],
        "detect_terms": [
            "respiratory acidosis", "respiratory alkalosis", "metabolic acidosis",
            "metabolic alkalosis", "paco2", "hco3", "bicarbonate", "compensation",
        ],
        "min_terms": 4,
        "confusable": [
            ("Respiratory Acidosis", "Metabolic Acidosis"),
            ("Respiratory Alkalosis", "Metabolic Alkalosis"),
        ],
        "stem_subject": "a set of arterial blood gas values",
    },

    "sbar": {
        "name": "SBAR Handoff Communication",
        "item_shape": SHAPE_CLASSIFY,
        "categories": ["Situation", "Background", "Assessment", "Recommendation"],
        "anchors": ["sbar", "isbar", "situation background assessment recommendation"],
        "detect_terms": [
            "sbar", "handoff", "hand-off", "situation", "background",
            "recommendation", "shift report",
        ],
        "min_terms": 4,
        "confusable": [("Situation", "Background"), ("Assessment", "Recommendation")],
        "stem_subject": "a line from a nurse's handoff report",
    },

    "erikson": {
        "name": "Erikson's Stages of Psychosocial Development",
        "item_shape": SHAPE_CLASSIFY,
        "categories": [
            "Trust vs. Mistrust", "Autonomy vs. Shame and Doubt",
            "Initiative vs. Guilt", "Industry vs. Inferiority",
            "Identity vs. Role Confusion", "Intimacy vs. Isolation",
            "Generativity vs. Stagnation", "Integrity vs. Despair",
        ],
        "anchors": ["erikson", "psychosocial development"],
        "detect_terms": [
            "trust versus mistrust", "trust vs mistrust", "autonomy versus shame",
            "initiative versus guilt", "industry versus inferiority",
            "identity versus role confusion", "intimacy versus isolation",
            "generativity versus stagnation", "integrity versus despair",
        ],
        "min_terms": 3,
        "confusable": [
            ("Initiative vs. Guilt", "Industry vs. Inferiority"),
            ("Identity vs. Role Confusion", "Intimacy vs. Isolation"),
        ],
        "stem_subject": "a patient of a stated age and their behaviour",
    },

    "grief_stages": {
        "name": "Kübler-Ross Stages of Grief",
        "item_shape": SHAPE_CLASSIFY,
        "categories": ["Denial", "Anger", "Bargaining", "Depression", "Acceptance"],
        "anchors": ["kubler-ross", "kübler-ross", "kubler ross", "stages of grief"],
        "detect_terms": [
            "denial", "anger", "bargaining", "acceptance", "anticipatory grief",
        ],
        "min_terms": 4,
        "confusable": [("Denial", "Bargaining"), ("Anger", "Depression")],
        "stem_subject": "something a grieving patient or family member says",
    },
}


# Compiled once at import. Word-boundary matching so "n95" does not fire inside
# a longer token and "delegate" does not match "delegated" only by luck — \b
# handles the common inflections we care about at the edges.
def _compile(term: str) -> "re.Pattern[str]":
    return re.compile(r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\w{0,3}\b",
                      re.IGNORECASE)


_ANCHOR_PATTERNS: Dict[str, List["re.Pattern[str]"]] = {
    fid: [_compile(a) for a in fw["anchors"]] for fid, fw in FRAMEWORKS.items()
}
_TERM_PATTERNS: Dict[str, List["re.Pattern[str]"]] = {
    fid: [_compile(t) for t in fw["detect_terms"]] for fid, fw in FRAMEWORKS.items()
}


# ============================================
# PUBLIC API
# ============================================

def framework_ids() -> List[str]:
    """The closed set, in registry order."""
    return list(FRAMEWORKS.keys())


def get_framework(framework_id: str, *, nclex_framing: bool = False) -> Optional[Dict[str, Any]]:
    """
    One framework, or None for an unknown id.

    `nclex_framing=True` swaps the nursing process's ADPIE categories for the
    NCSBN clinical-judgment steps. Only that entry differs; everything else
    returns unchanged, so callers can pass the flag unconditionally.
    """
    fw = FRAMEWORKS.get(framework_id)
    if not fw:
        return None

    out = dict(fw)
    out["id"] = framework_id
    if nclex_framing and fw.get("categories_nclex"):
        out["categories"] = list(fw["categories_nclex"])
        out["confusable"] = []  # the ADPIE confusion pairs do not map across
    return out


def detect_frameworks(text: str, *, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Which of the closed set does this document actually teach?

    Runs over the FULL text — see the module docstring on why the sampled
    excerpt is not good enough. Pure and cheap: no model call, no I/O.

    Returns a list ordered strongest-first, or [] when the document teaches
    none of them. An empty list is the expected result for most documents and
    must not be worked around by loosening thresholds.

    Each result:
        {
          'id':         'nursing_process',
          'name':       'The Nursing Process',
          'confidence': 'named' | 'inferred',
          'matched':    ['nursing process', 'nursing diagnosis', ...],
          'evidence':   '...surrounding snippet...',
        }
    """
    if not text or not text.strip():
        return []

    results: List[Dict[str, Any]] = []

    for fid, fw in FRAMEWORKS.items():
        anchor_hits = [p.pattern for p in _ANCHOR_PATTERNS[fid] if p.search(text)]
        term_matches = [p for p in _TERM_PATTERNS[fid] if p.search(text)]
        term_count = len(term_matches)

        if anchor_hits:
            confidence = "named"
        elif term_count >= fw["min_terms"]:
            confidence = "inferred"
        else:
            continue

        # Report the literal strings, not the compiled patterns — callers log
        # these and a regex source is unreadable in a log line.
        matched = [a for a in fw["anchors"] if _compile(a).search(text)]
        matched += [t for t in fw["detect_terms"] if _compile(t).search(text)]

        results.append({
            "id": fid,
            "name": fw["name"],
            "confidence": confidence,
            "matched": matched[:8],
            "term_count": term_count,
            "evidence": _snippet(text, matched[0] if matched else fw["name"]),
        })

    # Named beats inferred; within a tier, more corroborating vocabulary wins.
    results.sort(key=lambda r: (r["confidence"] != "named", -r["term_count"]))
    return results[:max_results]


def build_model_choices(*, nclex_framing: bool = False) -> str:
    """
    The closed set rendered for a prompt, so the model picks an id or says none.

    Kept here rather than inline in the prompt so the list cannot drift from the
    registry that has to mark the resulting answers.
    """
    lines = []
    for fid in framework_ids():
        fw = get_framework(fid, nclex_framing=nclex_framing)
        cats = " | ".join(fw["categories"])
        lines.append(f'- {fid}: {fw["name"]} — categories: {cats}')
    lines.append('- none: the document teaches no framework from this list')
    return "\n".join(lines)


def _snippet(text: str, term: str, width: int = 90) -> str:
    """A short window around the first hit, for logs and eyeballing detections."""
    m = _compile(term).search(text)
    if not m:
        return ""
    start = max(0, m.start() - width // 2)
    end = min(len(text), m.end() + width // 2)
    return " ".join(text[start:end].split())
