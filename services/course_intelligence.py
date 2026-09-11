"""
Course Intelligence Service
===========================

What this is
------------
The student tells us four things about her class — school, course, professor,
exam description — and uploads her material. This service investigates that
specific academic environment and returns a structured brief the planner and
the UI both read.

It is the engine behind the "how does it know all this about my course?"
moment, and every step it streams corresponds to real work: a web search that
actually ran, an LLM pass that actually happened, or arithmetic over her own
files. Nothing here exists to fill time.

The three rules
---------------
1. HER MATERIALS ARE THE PRIMARY SOURCE. Public research adds context; it
   never outranks the document she uploaded. Topic priority is anchored on
   material emphasis, which is computed from her files in Python and cannot
   be moved by anything a web page said.

2. VERIFIED, PUBLIC AND INFERRED NEVER MIX. Every section carries a
   `confidence` of "verified" (her upload, or an official page we cited),
   "public" (a credible public source we cited) or "inference" (a pattern we
   noticed). The frontend renders the three differently and the copy for
   inference is always hedged. A section with no support is OMITTED, not
   softened — see `_discard_uncited`.

3. NOTHING ABOUT THE PERSON. The instructor pass returns role, department,
   specialty and publications, and the schema has no field capable of holding
   a personality claim, a difficulty rating or a teaching-style judgement.
   That is deliberate: prompts drift, schemas do not.

Shape of the run
----------------
    course_context_received     instant, echoes what she told us
    materials_analyzed          pure Python over the upload's insights
    course_research             ─┐
    professor_research           ├─ three real web searches, run concurrently
    academic_resources_research ─┤
    exam_analysis               ─┘  one LLM pass, no web, runs alongside
    concept_mapping             LLM: correlate exam ↔ materials ↔ objectives
    study_strategy              deterministic scoring + ordering
    complete                    the assembled report

The four middle steps run concurrently and their progress events are emitted
as each one genuinely lands, which is why the timeline finishes in roughly the
time of its slowest search rather than the sum of four.

Cost and speed
--------------
Three Haiku 4.5 calls with `web_search_20260209` (max_uses 2 each) plus one
Sonnet pass for the concept map. A live run lands around 20s end to end, with
the three searches finishing inside 10s and the concept map taking the rest.
The first cut of this used Sonnet for everything and took 48.6s, which is the
measurement the model split above exists to answer.

Two caches keep the cost down in the case that matters most — a whole class
uploading the same course:

    course + resources   keyed on (school, course), 7 days
    instructor           keyed on (school, professor), 30 days

A professor's faculty page does not change during a semester, and neither does
a course code, so these TTLs are long on purpose. Set
COURSE_RESEARCH_ENABLED=0 to switch the web passes off entirely; the run then
completes from her materials alone, which is a supported path rather than a
degraded one.
"""

import asyncio
import hashlib
import json
import os
import re
import time
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic

_client: Optional[AsyncAnthropic] = None

# MODEL SPLIT, and the measurement behind it
# ------------------------------------------
# A live run with all three searches on took 48.6s end to end: ~28s for the
# instructor pass, ~34s for the course pass, and the resources pass hit the
# timeout. That is too long for a screen a student is watching, and most of it
# was the research model reasoning rather than the searches themselves.
#
# The two jobs are not the same difficulty. The research passes ask "is this
# page about THIS course at THIS school" — retrieval plus a yes/no on
# identity, which Haiku answers as well as Sonnet and several times faster.
# The concept map asks how a topic in her material lines up with a sentence
# she wrote about her exam, which is judgement, and stays on Sonnet.
RESEARCH_MODEL = "claude-haiku-4-5-20251001"
ANALYSIS_MODEL = "claude-sonnet-4-6"

# Per-pass ceiling. A single slow search must not hold the whole timeline —
# the student is watching this, and a section that times out is reported as
# "nothing reliable found", which the UI already handles. 30s rather than 40:
# a search still thinking at 30s has not found her course, and capping the
# tail is worth more than the rare late find.
RESEARCH_TIMEOUT_S = 30
ANALYSIS_TIMEOUT_S = 45

COURSE_CACHE_TTL_S = 7 * 24 * 60 * 60
INSTRUCTOR_CACHE_TTL_S = 30 * 24 * 60 * 60


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def research_enabled() -> bool:
    return os.getenv("COURSE_RESEARCH_ENABLED", "1").strip().lower() not in ("0", "false", "no")


# ──────────────────────────────────────────────────────────────────────
# CONFIDENCE
# ──────────────────────────────────────────────────────────────────────
# Mirrored in src/Components/CourseIntelligence/courseIntelligenceModel.js
# (CONFIDENCE). The frontend colours and captions each band, so a new value
# here without one there renders as an unstyled unknown.

CONFIDENCE_VERIFIED = "verified"    # her upload, or an official page we cited
CONFIDENCE_PUBLIC = "public"        # a credible public source we cited
CONFIDENCE_INFERENCE = "inference"  # a pattern we noticed; always hedged in copy


# ──────────────────────────────────────────────────────────────────────
# PRIORITY SCORING — ⚠️ CROSS-REPO CONTRACT
# ──────────────────────────────────────────────────────────────────────
# Mirrored in src/Components/CourseIntelligence/courseIntelligenceModel.js as
# PRIORITY_WEIGHTS. The frontend re-derives scores when it re-ranks locally
# (a topic dropped, an exam date changed), so drift means the report and the
# plan disagree about what matters — the student is told to start on diuretics
# and handed cardiac output.
#
# Why these weights:
#   exam_relevance dominates because she is studying for one specific exam and
#   said so in her own words. material_emphasis is second and is the only
#   signal computed from her files rather than from a model, which is why it
#   is never allowed to be zero-weighted. complexity is last and small: a hard
#   topic needs more runway, but difficulty is not importance.
PRIORITY_WEIGHTS = {
    "exam_relevance": 0.34,
    "material_emphasis": 0.24,
    "objective_alignment": 0.16,
    "dependency": 0.12,
    "clinical_importance": 0.10,
    "complexity": 0.04,
}

# Among the top N by score, the recommended starting point is the one that
# unlocks the most other topics. Starting on the highest-scoring topic is not
# the same as starting where the plan flows best, and three is a wide enough
# window to prefer a foundation without demoting what her exam actually covers.
START_CANDIDATE_WINDOW = 3


# ──────────────────────────────────────────────────────────────────────
# TOOL SCHEMAS
# ──────────────────────────────────────────────────────────────────────

EMIT_COURSE_PROFILE = {
    "name": "emit_course_profile",
    "description": (
        "Emit what public sources say about this specific course at this "
        "specific school. Call exactly once, after searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": (
                    "True only if a search result was actually about THIS "
                    "course at THIS school. A generic page about the subject "
                    "at another university is not a find - return false."
                ),
            },
            "official_name": {"type": ["string", "null"], "description": "Course title as the institution writes it."},
            "department": {"type": ["string", "null"]},
            "level": {"type": ["string", "null"], "description": "e.g. 'Undergraduate, year 2'. Null if unclear."},
            "description": {
                "type": ["string", "null"],
                "description": "2-4 sentences from the official description. Do not invent one.",
            },
            "learning_objectives": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 12,
                "description": "Objectives as published. Empty if none were published.",
            },
            "syllabus_topics": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 20,
                "description": "Topics the published course outline lists.",
            },
            "official_source": {
                "type": "boolean",
                "description": "True if at least one citation is the university's own domain.",
            },
            "honesty_note": {
                "type": ["string", "null"],
                "description": (
                    "One sentence, shown to the student, when the search did "
                    "not find this course. Null when found is true."
                ),
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "snippet": {"type": "string", "description": "Under 40 words."},
                    },
                    "required": ["url", "title", "snippet"],
                },
                "maxItems": 6,
            },
        },
        "required": ["found", "citations"],
    },
}


EMIT_INSTRUCTOR_PROFILE = {
    "name": "emit_instructor_profile",
    "description": (
        "Emit verifiable professional information about this instructor from "
        "public academic sources. Call exactly once, after searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": (
                    "True only if you found a page about THIS person at THIS "
                    "institution. A same-name match elsewhere is not a find."
                ),
            },
            "display_name": {"type": ["string", "null"]},
            "title": {"type": ["string", "null"], "description": "Academic title, e.g. 'Associate Professor'."},
            "department": {"type": ["string", "null"]},
            "clinical_specialties": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
                "description": (
                    "Clinical or professional practice areas stated on a "
                    "public profile, e.g. 'critical care', 'maternal health'."
                ),
            },
            "research_areas": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
                "description": "Stated research or scholarly interests.",
            },
            "notable_publications": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
                "description": "Title (year) of published academic work.",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "snippet": {"type": "string", "description": "Under 40 words."},
                    },
                    "required": ["url", "title", "snippet"],
                },
                "maxItems": 5,
            },
        },
        "required": ["found", "citations"],
    },
    # NOTE FOR ANYONE EXTENDING THIS SCHEMA
    # There is no field here for teaching style, strictness, exam difficulty,
    # student ratings or "what they like to test". That is the point. The
    # student is being shown professional context, not a profile of a person,
    # and the schema is the only enforcement that survives a prompt edit.
}


EMIT_PUBLIC_RESOURCES = {
    "name": "emit_public_resources",
    "description": (
        "Emit legitimate, publicly available study resources relevant to this "
        "course's subject matter. Call exactly once, after searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "resources": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "course_page",
                                "syllabus",
                                "learning_objectives",
                                "study_guide",
                                "practice_questions",
                                "reference",
                                "open_courseware",
                            ],
                        },
                        "why_useful": {"type": "string", "description": "One short sentence."},
                        "official": {
                            "type": "boolean",
                            "description": "True if hosted by a university, ministry, or professional body.",
                        },
                    },
                    "required": ["title", "url", "kind", "why_useful", "official"],
                },
            },
            "published_objectives": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
                "description": "Learning objectives found in any public syllabus for this subject.",
            },
            "honesty_note": {"type": ["string", "null"]},
        },
        "required": ["found", "resources"],
    },
}


EMIT_EXAM_ANALYSIS = {
    "name": "emit_exam_analysis",
    "description": "Emit a structured reading of the student's own exam description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "exam_type": {
                "type": ["string", "null"],
                "description": "e.g. 'Midterm examination'. Null if she did not say.",
            },
            "coverage": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 12,
                "description": (
                    "Subject areas she named, in her own words, lightly "
                    "normalised. Only what the description actually states."
                ),
            },
            "formats": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
                "description": "Question formats she named, e.g. 'Multiple-choice questions'.",
            },
            "stated_emphasis": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
                "description": "Anything she flagged as weighted, prioritised, or 'focus on'.",
            },
            "unstated": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
                "description": (
                    "Useful things her description does NOT specify (e.g. "
                    "number of questions, whether it is cumulative). Honest "
                    "about the limits of what we were told."
                ),
            },
        },
        "required": ["coverage", "formats"],
    },
}


EMIT_CONCEPT_MAP = {
    "name": "emit_concept_map",
    "description": (
        "Score every topic from the student's uploaded materials against her "
        "exam and the course objectives. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Must be one of the topic labels given to you, copied exactly.",
                        },
                        "exam_relevance": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": (
                                "100 = named explicitly in her exam description. "
                                "60-80 = clearly part of a named area. 0-20 = not "
                                "mentioned and not implied. Be strict: if the "
                                "description does not reach this topic, score it low."
                            ),
                        },
                        "objective_alignment": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "How directly the published course objectives cover it. 0 if there are no objectives.",
                        },
                        "dependency": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "How much understanding this unlocks other topics in the list.",
                        },
                        "clinical_importance": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Weight of nursing judgement, patient safety, and decision-making it carries.",
                        },
                        "complexity": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "How much time and prerequisite knowledge it needs.",
                        },
                        "evidence": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": (
                                            "One short sentence a student could "
                                            "verify, e.g. 'Named in your exam "
                                            "description' or 'Appears in 3 of "
                                            "your 4 files'. Never a claim about "
                                            "the professor."
                                        ),
                                    },
                                    "source": {
                                        "type": "string",
                                        "enum": ["exam_description", "materials", "objectives", "inference"],
                                    },
                                },
                                "required": ["text", "source"],
                            },
                        },
                    },
                    "required": [
                        "topic",
                        "exam_relevance",
                        "objective_alignment",
                        "dependency",
                        "clinical_importance",
                        "complexity",
                        "evidence",
                    ],
                },
            },
            "connections": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "relation": {"type": "string", "description": "Under 10 words."},
                    },
                    "required": ["from", "to", "relation"],
                },
                "description": "Real conceptual links between her topics, for the plan's ordering.",
            },
            "strategy_note": {
                "type": "string",
                "description": (
                    "Two sentences, addressed to the student, explaining what "
                    "the plan will lead with and why. Hedged where it rests on "
                    "inference. Never asserts what will be on the exam."
                ),
            },
        },
        "required": ["topics", "strategy_note"],
    },
}


WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 2,
    # Required for the Haiku research passes: without it the API rejects the
    # request outright, because web_search defaults to a caller mode Haiku
    # does not implement ("does not support programmatic tool calling").
    # Direct calling is what we want anyway — the model searches, we read the
    # tool result it emits, and no code path calls the tool on its behalf.
    "allowed_callers": ["direct"],
}


# ──────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────────────

COURSE_SYSTEM = """You research university courses for a nursing study app.

A student has named her school and her course. Find what the institution and
other credible public sources actually publish about THAT course.

Search strategy (you have 2 searches - spend them well):
- "<school> <course code>" and "<school> <course name> syllabus" are usually
  the highest-yield pair.
- Prefer the university's own domain, department pages, and published course
  outlines over aggregator sites.

Hard rules:
- If you cannot find THIS course at THIS school, set found=false and say so in
  honesty_note. A course with the same name at a different university is NOT a
  match and must not be reported as one.
- Never invent a URL, an objective, or a course description. Every citation
  must be a page you actually retrieved.
- Do not claim access to private course pages, learning management systems, or
  materials behind a login.
"""

INSTRUCTOR_SYSTEM = """You look up publicly available professional information
about a named university instructor, for a nursing study app.

What to gather: their academic title, department, stated clinical or
professional specialties, stated research areas, and published academic work.
Faculty pages, department directories, professional bodies, and publication
records are the sources you want.

Hard rules, and they are the whole job:
- ONLY verifiable professional facts stated on a public page.
- NOTHING about their personality, teaching style, strictness, grading, exam
  difficulty, or what they "like to ask". Student-rating sites are not a
  source. If a page contains opinions about the person, ignore that page.
- If you cannot confirm this is the right person at the right institution, set
  found=false. A name match alone is not confirmation.
- Never invent a title, a specialty, or a URL.
"""

RESOURCES_SYSTEM = """You find legitimate, publicly available study resources
for a specific nursing course subject.

Good: university course pages, published syllabi and learning objectives,
open courseware, professional-body guidance, reputable open study guides,
official sample questions published by an institution or board.

Never: paywalled content presented as free, leaked or shared exam files, sites
offering another student's coursework, or anything that presents unofficial
material as this professor's actual exam. If you find such a page, exclude it.

Set found=false rather than padding the list with generic quiz sites.
Every URL must be one you actually retrieved.
"""

EXAM_SYSTEM = """You read a student's own description of her upcoming exam and
structure it. You have no other source and you must not add to what she wrote.

Extract only what is present. If she did not state the format, formats is
empty - do not guess "probably multiple choice". Put the things she left
unspecified into `unstated` so the app can be honest about them.
"""

CONCEPT_SYSTEM = """You are prioritising a nursing student's study topics.

You are given: the topics extracted from HER uploaded course materials, how
heavily each one is covered in those files, her own exam description, and any
published course objectives we found.

Score each topic on the given axes. Two rules matter more than the rest:

1. HER EXAM DESCRIPTION IS THE STRONGEST SIGNAL, and it is literal. A topic
   she named scores high on exam_relevance. A topic she did not name and that
   is not clearly inside an area she named scores low, however important the
   subject is in general.

2. EVIDENCE MUST BE CHECKABLE. Each evidence line is something the student
   could verify by looking at her own material or her own description. Write
   "Appears across your uploaded materials", not "Professors usually test
   this". Never write anything about her instructor.

Use only the topic labels you were given, copied exactly. Do not add topics,
do not rename them, do not merge them.
"""


# ──────────────────────────────────────────────────────────────────────
# CACHE (Firestore, same pattern as exam_research)
# ──────────────────────────────────────────────────────────────────────

def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _cache_key(prefix: str, *parts: Optional[str]) -> str:
    payload = "|".join(_norm(p) for p in parts).encode("utf-8")
    return f"{prefix}_" + hashlib.sha1(payload).hexdigest()[:24]


async def _cache_get(key: str, ttl_s: int) -> Optional[dict]:
    try:
        from firebase_admin import firestore
    except Exception:
        return None

    def _read():
        try:
            db = firestore.client()
            doc = db.collection("course_intelligence_cache").document(key).get()
            if not doc.exists:
                return None
            data = doc.to_dict() or {}
            saved_at = data.get("saved_at_epoch")
            if not saved_at or (time.time() - saved_at) > ttl_s:
                return None
            return data.get("payload")
        except Exception as e:
            print(f"course_intelligence cache read failed: {e}")
            return None

    return await asyncio.to_thread(_read)


async def _cache_set(key: str, payload: dict) -> None:
    try:
        from firebase_admin import firestore
    except Exception:
        return

    def _write():
        try:
            db = firestore.client()
            db.collection("course_intelligence_cache").document(key).set(
                {"payload": payload, "saved_at_epoch": time.time()}
            )
        except Exception as e:
            print(f"course_intelligence cache write failed: {e}")

    await asyncio.to_thread(_write)


# ──────────────────────────────────────────────────────────────────────
# CLAUDE HELPERS
# ──────────────────────────────────────────────────────────────────────

def _http_citations(raw) -> list:
    """Keep only citations with a real http(s) URL. A 'source' we cannot link
    to is not a source, and the UI renders every one of these as a link."""
    out = []
    seen = set()
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        url = (c.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "url": url,
                "title": (c.get("title") or url)[:200],
                "snippet": (c.get("snippet") or "")[:280],
            }
        )
    return out


async def _call_tool(
    *,
    system: str,
    user_prompt: str,
    tool: dict,
    web: bool,
    model: str,
    max_tokens: int = 3000,
) -> Optional[dict]:
    """One Claude call that must answer by calling `tool`. Returns the tool
    input, or None if the model declined to call it (which we treat as "no
    answer", never as an empty answer)."""
    client = _get_client()
    tools = [tool]
    if web:
        tools = [WEB_SEARCH_TOOL, tool]

    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        tools=tools,
        messages=[{"role": "user", "content": user_prompt}],
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool["name"]:
            return dict(block.input or {})

    print(
        f"course_intelligence: {tool['name']} not called. "
        f"stop_reason={getattr(response, 'stop_reason', '?')}"
    )
    return None


# Academic domains, in the shapes they actually occur. Deliberately narrow:
# a false positive here promotes a random blog to "Verified" in the student's
# report, which is the one thing this file cannot let happen.
_ACADEMIC_SUFFIXES = (".edu", ".ac.uk", ".edu.au", ".ac.nz", ".edu.sg", ".ac.za", ".edu.gh")

# Words that carry no identity, so they cannot be the thing that matches a
# school to a domain. "University of X" must match on X, never on "university".
_SCHOOL_STOPWORDS = {
    "university", "universite", "université", "college", "school", "institute",
    "of", "the", "and", "for", "at", "campus", "state", "national",
}


def _looks_official(citations: list, school: Optional[str]) -> bool:
    """
    Is at least one of these citations plausibly the institution's own page?

    WHY THIS IS NOT THE MODEL'S DECISION
    ------------------------------------
    `official_source` is what promotes the course section from "Public source"
    to "Verified" in the report, and verified is the badge that buys the
    student's trust in everything else on the card. Asked to self-report it,
    the research model marked a third-party LMS export (lms.courselearn.net)
    as official on a real run — defensible as a document, wrong as a domain,
    and exactly the kind of small over-claim that is invisible until a student
    checks one link.

    So the claim is checked here, against the URL, which cannot be persuaded:
    an academic TLD, or a host carrying a distinctive word from the school's
    own name. Anything else stays "public", which is still shown, still cited,
    and still useful — just not badged as coming from her university.
    """
    if not citations:
        return False

    tokens = {
        t for t in re.split(r"[^a-z0-9]+", (school or "").lower())
        if len(t) > 2 and t not in _SCHOOL_STOPWORDS
    }

    for c in citations:
        host = ""
        try:
            host = re.sub(r"^https?://", "", c.get("url", "")).split("/")[0].lower()
            host = host.split(":")[0]
        except Exception:
            continue
        if not host:
            continue
        if host.endswith(_ACADEMIC_SUFFIXES) or ".ac." in host:
            return True
        # Strip the TLD before matching so "nursing.example.com" cannot match a
        # school called "Example" via the registrar's domain alone... it can,
        # and that is the intended behaviour: mcgill.ca IS McGill's domain.
        if any(t in host for t in tokens):
            return True
    return False


def _discard_uncited(payload: Optional[dict], confidence: str) -> Optional[dict]:
    """A research section that claims a find but cites nothing did not find
    anything — it remembered something. Drop it.

    This is the single most important function in the file: it is what stops
    the report from becoming confident fiction, and it runs before any caller
    gets to look at the result."""
    if not payload:
        return None
    if not payload.get("found"):
        return None
    citations = _http_citations(payload.get("citations"))
    if not citations:
        print(f"course_intelligence: discarding uncited '{confidence}' section")
        return None
    payload["citations"] = citations
    payload["confidence"] = confidence
    return payload


# ──────────────────────────────────────────────────────────────────────
# MATERIALS — pure Python over her own files
# ──────────────────────────────────────────────────────────────────────

def summarize_materials(material_insights: dict) -> dict:
    """
    Everything here is counted, not judged: which topics appear, in how many
    files, with how many extracted key points. This is the one section that is
    always `verified`, because its source is the document she uploaded.

    `emphasis` is the 0-100 signal the priority score leans on. It is
    deliberately computed here rather than asked of a model: a number derived
    from her files cannot be talked out of by a web page.
    """
    per_topic = {}
    filenames = []
    doc_types = set()
    frameworks = set()
    total_concepts = 0

    for filename, insights in (material_insights or {}).items():
        if not insights:
            continue
        filenames.append(filename)
        dt = insights.get("document_type")
        if dt:
            doc_types.add(dt)
        for fw in insights.get("frameworks") or []:
            # Upload detection and Firestore use {id, name, confidence}; older
            # insights may contain plain labels. Summaries expose labels only.
            label = (fw.get("name") or fw.get("id")) if isinstance(fw, dict) else fw
            if isinstance(label, str) and label.strip():
                frameworks.add(label.strip())

        topics = [t for t in (insights.get("topics") or []) if isinstance(t, str) and t.strip()]
        concepts = [c for c in (insights.get("concepts") or []) if isinstance(c, str) and c.strip()]
        total_concepts += len(concepts)

        for topic in topics:
            label = topic.strip()
            slot = per_topic.setdefault(label, {"topic": label, "files": set(), "concepts": []})
            slot["files"].add(filename)
            # Concepts are extracted per file, not per topic, so a file's
            # concepts are attributed to each topic that file covers. That
            # over-counts a many-topic file, which is why the count is
            # normalised below rather than reported raw.
            slot["concepts"].extend(concepts)

    file_count = max(len(filenames), 1)
    rows = []
    for slot in per_topic.values():
        file_share = len(slot["files"]) / file_count
        concept_depth = min(len(set(slot["concepts"])), 12) / 12.0
        # Coverage breadth (how many of her files touch it) counts double the
        # depth of any one file: a topic in every deck is the spine of the
        # course, a topic with many bullets in one deck is one long slide.
        emphasis = round(100 * min(1.0, (0.66 * file_share) + (0.34 * concept_depth)))
        rows.append(
            {
                "topic": slot["topic"],
                "file_count": len(slot["files"]),
                "files": sorted(slot["files"]),
                "key_points": sorted(set(slot["concepts"]))[:8],
                "emphasis": emphasis,
            }
        )

    rows.sort(key=lambda r: (-r["emphasis"], r["topic"].lower()))

    return {
        "confidence": CONFIDENCE_VERIFIED,
        "file_count": len(filenames),
        "filenames": filenames,
        "topic_count": len(rows),
        "concept_count": total_concepts,
        "document_types": sorted(doc_types),
        "frameworks": sorted(frameworks),
        "topics": rows,
    }


# ──────────────────────────────────────────────────────────────────────
# RESEARCH PASSES
# ──────────────────────────────────────────────────────────────────────

def _course_label(ctx: dict) -> str:
    code = (ctx.get("courseCode") or "").strip()
    name = (ctx.get("courseName") or "").strip()
    if code and name:
        return f"{code} - {name}"
    return code or name or ""


async def research_course(ctx: dict, language: str) -> Optional[dict]:
    school = (ctx.get("school") or "").strip()
    course = _course_label(ctx)
    if not school or not course:
        return None

    key = _cache_key("course", school, course)
    cached = await _cache_get(key, COURSE_CACHE_TTL_S)
    if cached:
        cached = dict(cached)
        cached["cached"] = True
        return cached

    prompt = (
        f"School: {school}\n"
        f"Course: {course}\n"
        f"Response language: {language}\n\n"
        f"Search for this specific course at this specific school, then call "
        f"emit_course_profile once."
    )
    payload = await _call_tool(
        system=COURSE_SYSTEM,
        user_prompt=prompt,
        tool=EMIT_COURSE_PROFILE,
        web=True,
        model=RESEARCH_MODEL,
    )
    # An official university page is a stronger claim than a public mention,
    # and the report labels the two differently. The model's own
    # `official_source` is treated as a proposal, not a verdict: it only holds
    # if a citation's DOMAIN backs it up. See _looks_official.
    claimed_official = bool((payload or {}).get("official_source"))
    verified_official = claimed_official and _looks_official(
        _http_citations((payload or {}).get("citations")), school
    )
    if claimed_official and not verified_official:
        print(f"course_intelligence: demoting unofficial-domain course claim for {school!r}")
    if payload is not None:
        payload["official_source"] = verified_official
    confidence = CONFIDENCE_VERIFIED if verified_official else CONFIDENCE_PUBLIC
    payload = _discard_uncited(payload, confidence)
    if payload:
        payload["cached"] = False
        asyncio.create_task(_cache_set(key, payload))
    return payload


async def research_instructor(ctx: dict, language: str) -> Optional[dict]:
    school = (ctx.get("school") or "").strip()
    professor = (ctx.get("professor") or "").strip()
    if not professor:
        return None

    key = _cache_key("instructor", school, professor)
    cached = await _cache_get(key, INSTRUCTOR_CACHE_TTL_S)
    if cached:
        cached = dict(cached)
        cached["cached"] = True
        return cached

    prompt = (
        f"Instructor: {professor}\n"
        f"Institution: {school or 'unknown'}\n"
        f"Course they teach: {_course_label(ctx) or 'nursing'}\n"
        f"Response language: {language}\n\n"
        f"Find their public academic profile, then call emit_instructor_profile "
        f"once. Professional facts only."
    )
    payload = await _call_tool(
        system=INSTRUCTOR_SYSTEM,
        user_prompt=prompt,
        tool=EMIT_INSTRUCTOR_PROFILE,
        web=True,
        model=RESEARCH_MODEL,
        max_tokens=2000,
    )
    payload = _discard_uncited(payload, CONFIDENCE_PUBLIC)
    if payload:
        # Belt and braces on top of the schema: a profile with no professional
        # substance is not context, it is a name we found on a page.
        has_substance = any(
            payload.get(f)
            for f in ("title", "clinical_specialties", "research_areas", "notable_publications")
        )
        if not has_substance:
            return None
        payload["cached"] = False
        asyncio.create_task(_cache_set(key, payload))
    return payload


async def research_resources(ctx: dict, material_topics: list, language: str) -> Optional[dict]:
    course = _course_label(ctx)
    subject = course or ", ".join(material_topics[:3])
    if not subject:
        return None

    key = _cache_key("resources", ctx.get("school"), course, ", ".join(material_topics[:5]))
    cached = await _cache_get(key, COURSE_CACHE_TTL_S)
    if cached:
        cached = dict(cached)
        cached["cached"] = True
        return cached

    prompt = (
        f"Course: {subject}\n"
        f"School: {(ctx.get('school') or 'unspecified')}\n"
        f"Topics from the student's own materials: {', '.join(material_topics[:8]) or 'unspecified'}\n"
        f"Response language: {language}\n\n"
        f"Find publicly available, legitimate study resources for this subject, "
        f"then call emit_public_resources once."
    )
    payload = await _call_tool(
        system=RESOURCES_SYSTEM,
        user_prompt=prompt,
        tool=EMIT_PUBLIC_RESOURCES,
        web=True,
        model=RESEARCH_MODEL,
        max_tokens=2500,
    )
    if not payload or not payload.get("found"):
        return None

    # One URL, one row. The model happily returns the same page twice under
    # two different `kind` values, and a report that lists the same link as
    # both "open courseware" and "reference" looks like it padded the list.
    resources = []
    seen_urls = set()
    for r in payload.get("resources") or []:
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen_urls:
            continue
        seen_urls.add(url)
        resources.append(r)
        if len(resources) == 6:
            break
    if not resources:
        return None
    payload["resources"] = resources
    payload["confidence"] = CONFIDENCE_PUBLIC
    payload["citations"] = [
        {"url": r["url"], "title": r.get("title") or r["url"], "snippet": r.get("why_useful") or ""}
        for r in resources
    ]
    payload["cached"] = False
    asyncio.create_task(_cache_set(key, payload))
    return payload


async def analyze_exam(ctx: dict, language: str) -> Optional[dict]:
    description = (ctx.get("examDescription") or "").strip()
    if not description:
        return None

    prompt = (
        f"The student wrote this about her exam:\n\n\"{description}\"\n\n"
        f"Response language: {language}\n\n"
        f"Structure it with emit_exam_analysis. Add nothing she did not write."
    )
    payload = await _call_tool(
        system=EXAM_SYSTEM,
        user_prompt=prompt,
        tool=EMIT_EXAM_ANALYSIS,
        web=False,
        model=ANALYSIS_MODEL,
        max_tokens=1500,
    )
    if not payload:
        return None
    # Her own words about her own exam: verified, and the only section
    # allowed to be verified without a citation.
    payload["confidence"] = CONFIDENCE_VERIFIED
    payload["source_text"] = description[:600]
    return payload


async def map_concepts(
    *,
    materials: dict,
    exam: Optional[dict],
    objectives: list,
    language: str,
) -> Optional[dict]:
    topics = materials.get("topics") or []
    if not topics:
        return None

    topic_lines = "\n".join(
        f"- {r['topic']} (in {r['file_count']} of {materials.get('file_count', 1)} files; "
        f"key points: {', '.join(r['key_points'][:5]) or 'none extracted'})"
        for r in topics[:24]
    )
    exam_block = "none provided"
    if exam:
        exam_block = json.dumps(
            {
                "exam_type": exam.get("exam_type"),
                "coverage": exam.get("coverage"),
                "formats": exam.get("formats"),
                "stated_emphasis": exam.get("stated_emphasis"),
                "student_words": exam.get("source_text"),
            },
            ensure_ascii=False,
        )

    prompt = (
        f"TOPICS FROM HER UPLOADED MATERIALS:\n{topic_lines}\n\n"
        f"HER EXAM DESCRIPTION (structured):\n{exam_block}\n\n"
        f"PUBLISHED COURSE OBJECTIVES:\n"
        f"{chr(10).join('- ' + o for o in objectives[:12]) or 'none found'}\n\n"
        f"Response language: {language}\n\n"
        f"Score every topic with emit_concept_map."
    )
    return await _call_tool(
        system=CONCEPT_SYSTEM,
        user_prompt=prompt,
        tool=EMIT_CONCEPT_MAP,
        web=False,
        model=ANALYSIS_MODEL,
        max_tokens=6000,
    )


# ──────────────────────────────────────────────────────────────────────
# STRATEGY — deterministic, auditable, no model in the loop
# ──────────────────────────────────────────────────────────────────────

def build_strategy(materials: dict, concept_map: Optional[dict], exam: Optional[dict]) -> dict:
    """
    Turn the signals into an ordered list, in Python, so the ordering can be
    explained line by line and reproduces exactly on a re-run. The model
    scores; it does not rank.

    A topic the model failed to score is not dropped — it keeps its material
    emphasis and scores zero on everything else, which lands it below the
    scored topics without erasing it from her plan.
    """
    rows = []
    scored = {}
    for entry in (concept_map or {}).get("topics") or []:
        if isinstance(entry, dict) and entry.get("topic"):
            scored[_norm(entry["topic"])] = entry

    for mat in materials.get("topics") or []:
        label = mat["topic"]
        s = scored.get(_norm(label)) or {}

        signals = {
            "exam_relevance": int(s.get("exam_relevance") or 0),
            "material_emphasis": int(mat.get("emphasis") or 0),
            "objective_alignment": int(s.get("objective_alignment") or 0),
            "dependency": int(s.get("dependency") or 0),
            "clinical_importance": int(s.get("clinical_importance") or 0),
            "complexity": int(s.get("complexity") or 0),
        }
        score = round(sum(PRIORITY_WEIGHTS[k] * v for k, v in signals.items()))

        evidence = []
        for ev in (s.get("evidence") or [])[:3]:
            if isinstance(ev, dict) and ev.get("text"):
                evidence.append(
                    {
                        "text": ev["text"][:160],
                        "source": ev.get("source") or "inference",
                        # Anything the model reasoned rather than read is an
                        # inference and is captioned as one in the UI.
                        "confidence": CONFIDENCE_VERIFIED
                        if ev.get("source") in ("exam_description", "materials")
                        else CONFIDENCE_INFERENCE,
                    }
                )
        if not evidence:
            # Never leave a topic unexplained. This line is countable from her
            # own upload, so it is verified rather than inferred.
            evidence.append(
                {
                    "text": f"Covered in {mat['file_count']} of your {materials.get('file_count', 1)} files",
                    "source": "materials",
                    "confidence": CONFIDENCE_VERIFIED,
                }
            )

        rows.append(
            {
                "topic": label,
                "score": score,
                "signals": signals,
                "evidence": evidence,
                "key_points": mat.get("key_points", [])[:4],
                "file_count": mat.get("file_count", 0),
                "scored": bool(s),
            }
        )

    rows.sort(key=lambda r: (-r["score"], -r["signals"]["material_emphasis"], r["topic"].lower()))

    # Recommended start: highest dependency inside the top window, because the
    # best first topic is the one the rest of the plan stands on.
    start = None
    if rows:
        window = rows[:START_CANDIDATE_WINDOW]
        start = max(window, key=lambda r: (r["signals"]["dependency"], r["score"]))

    reasons = []
    if start:
        if start["signals"]["exam_relevance"] >= 60:
            reasons.append({"key": "exam", "confidence": CONFIDENCE_VERIFIED})
        if start["signals"]["dependency"] >= 50:
            reasons.append({"key": "unlocks", "confidence": CONFIDENCE_INFERENCE})
        if start["signals"]["clinical_importance"] >= 60:
            reasons.append({"key": "clinical", "confidence": CONFIDENCE_INFERENCE})
        if start["signals"]["material_emphasis"] >= 60:
            reasons.append({"key": "emphasis", "confidence": CONFIDENCE_VERIFIED})
        if not reasons:
            reasons.append({"key": "emphasis", "confidence": CONFIDENCE_VERIFIED})

    return {
        "ordered_topics": [r["topic"] for r in rows],
        "priority_topics": rows,
        "recommended_start": (
            {"topic": start["topic"], "reasons": reasons, "score": start["score"]} if start else None
        ),
        "connections": (concept_map or {}).get("connections") or [],
        "note": (concept_map or {}).get("strategy_note") or "",
        "confidence": CONFIDENCE_INFERENCE,
        "exam_aligned": bool(exam),
    }


# ──────────────────────────────────────────────────────────────────────
# THE RUN
# ──────────────────────────────────────────────────────────────────────

# Step → the progress percentage the bar sits at once that step is DONE.
# The frontend has the same table (courseIntelligenceModel.STEPS) and uses it
# for layout; the number on the wire is what the bar actually reads, so the
# two only need to agree in spirit, not to the point.
_STEP_PROGRESS = {
    "course_context_received": 6,
    "materials_analyzed": 18,
    "course_research": 42,
    "professor_research": 54,
    "academic_resources_research": 64,
    "exam_analysis": 72,
    "concept_mapping": 88,
    "study_strategy": 96,
}


def _event(step: str, state: str, detail=None, message: str = "") -> dict:
    return {
        "status": "course_intelligence_progress",
        "step": step,
        "state": state,  # "running" | "done" | "empty" | "failed"
        "progress": _STEP_PROGRESS.get(step, 0) if state != "running" else max(
            0, _STEP_PROGRESS.get(step, 0) - 8
        ),
        "message": message,
        "detail": detail or {},
    }


def normalize_context(raw: Optional[dict]) -> dict:
    raw = raw or {}
    out = {}
    for field in ("school", "courseCode", "courseName", "professor", "examDescription"):
        value = raw.get(field)
        out[field] = re.sub(r"\s+", " ", str(value)).strip()[:400] if value else ""
    # examDate rides along untouched — the planner reads it, this service does
    # not, and reformatting a date here is how two surfaces start disagreeing
    # about which day the exam is.
    out["examDate"] = raw.get("examDate") or None
    return out


async def stream_course_intelligence(
    course_context: Optional[dict],
    material_insights: Optional[dict],
    language: str = "english",
) -> AsyncIterator[dict]:
    """
    Yields progress dicts, then exactly one of:
        {"status": "course_intelligence_ready", "report": {...}}
        {"status": "error", "message": "..."}

    Never raises for a failed section. A search that finds nothing, times out,
    or is switched off produces a report without that section — which is a
    normal outcome, not an error, because her materials were always the point.
    """
    ctx = normalize_context(course_context)
    started = time.time()

    yield _event("course_context_received", "done", detail={"context": ctx})

    materials = summarize_materials(material_insights or {})
    yield _event(
        "materials_analyzed",
        "done" if materials["topic_count"] else "empty",
        detail={
            "topic_count": materials["topic_count"],
            "file_count": materials["file_count"],
            "concept_count": materials["concept_count"],
            "top_topics": [r["topic"] for r in materials["topics"][:4]],
        },
    )

    material_topics = [r["topic"] for r in materials["topics"]]

    # ── The concurrent middle ─────────────────────────────────────────
    web_on = research_enabled()
    jobs = {}
    if web_on and (ctx["school"] and (ctx["courseCode"] or ctx["courseName"])):
        jobs["course_research"] = research_course(ctx, language)
    if web_on and ctx["professor"]:
        jobs["professor_research"] = research_instructor(ctx, language)
    if web_on and (ctx["courseCode"] or ctx["courseName"] or material_topics):
        jobs["academic_resources_research"] = research_resources(ctx, material_topics, language)
    if ctx["examDescription"]:
        jobs["exam_analysis"] = analyze_exam(ctx, language)

    # Steps with nothing to do are reported immediately as empty rather than
    # silently skipped: a timeline that hides a step it never ran is the kind
    # of small lie this whole feature cannot afford.
    for step in ("course_research", "professor_research", "academic_resources_research", "exam_analysis"):
        if step not in jobs:
            yield _event(step, "empty", detail={"reason": "no_input" if not web_on or step == "exam_analysis" else "disabled"})

    for step in jobs:
        yield _event(step, "running")

    results = {}
    if jobs:
        tasks = {
            asyncio.create_task(asyncio.wait_for(coro, timeout=RESEARCH_TIMEOUT_S)): step
            for step, coro in jobs.items()
        }
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    step = tasks[task]
                    try:
                        value = task.result()
                    except asyncio.TimeoutError:
                        print(f"course_intelligence: {step} timed out")
                        value = None
                    except Exception as e:
                        print(f"course_intelligence: {step} failed: {type(e).__name__}: {e}")
                        value = None
                    results[step] = value
                    yield _event(
                        step,
                        "done" if value else "empty",
                        detail=_detail_for(step, value),
                    )
        finally:
            # Students can finish the early diagnostic before these searches.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    course = results.get("course_research")
    instructor = results.get("professor_research")
    resources = results.get("academic_resources_research")
    exam = results.get("exam_analysis")

    # ── Concept mapping ───────────────────────────────────────────────
    # Objectives come from the two passes that publish them. The instructor
    # pass contributes nothing here by design — see its schema note.
    objectives = list((course or {}).get("learning_objectives") or [])
    objectives += list((resources or {}).get("published_objectives") or [])

    yield _event("concept_mapping", "running")
    concept_map = None
    if materials["topic_count"]:
        try:
            concept_map = await asyncio.wait_for(
                map_concepts(materials=materials, exam=exam, objectives=objectives, language=language),
                timeout=ANALYSIS_TIMEOUT_S,
            )
        except Exception as e:
            print(f"course_intelligence: concept mapping failed: {type(e).__name__}: {e}")
    yield _event(
        "concept_mapping",
        "done" if concept_map else "empty",
        detail={"scored": len((concept_map or {}).get("topics") or [])},
    )

    # ── Strategy ──────────────────────────────────────────────────────
    yield _event("study_strategy", "running")
    strategy = build_strategy(materials, concept_map, exam)
    yield _event(
        "study_strategy",
        "done" if strategy["priority_topics"] else "empty",
        detail={
            "topic_count": len(strategy["priority_topics"]),
            "start": (strategy.get("recommended_start") or {}).get("topic"),
        },
    )

    report = {
        "generated_at": time.time(),
        "elapsed_ms": int((time.time() - started) * 1000),
        "course_information": {
            "confidence": CONFIDENCE_VERIFIED,  # she told us this herself
            **ctx,
            "public_profile": course,
        },
        "uploaded_material_insights": materials,
        "verified_public_information": course if (course or {}).get("confidence") == CONFIDENCE_VERIFIED else None,
        "relevant_public_context": {
            "instructor": instructor,
            "resources": resources,
            "course": course if (course or {}).get("confidence") == CONFIDENCE_PUBLIC else None,
        },
        "exam_analysis": exam,
        "priority_topics": strategy["priority_topics"],
        "inferences": {
            "confidence": CONFIDENCE_INFERENCE,
            "connections": strategy["connections"],
            "note": strategy["note"],
        },
        "study_strategy": strategy,
        "research_ran": bool(course or instructor or resources),
        "research_enabled": web_on,
    }

    yield {"status": "course_intelligence_ready", "report": report}


def _detail_for(step: str, value: Optional[dict]) -> dict:
    """The one true fact each finished step gets to put on screen. Kept small
    on purpose: the timeline shows momentum, the report shows findings."""
    if not value:
        return {}
    if step == "course_research":
        return {
            "official_name": value.get("official_name"),
            "department": value.get("department"),
            "objective_count": len(value.get("learning_objectives") or []),
            "official": bool(value.get("official_source")),
            "citation_count": len(value.get("citations") or []),
            "sources": _http_citations(value.get("citations"))[:3],
        }
    if step == "professor_research":
        return {
            "title": value.get("title"),
            "department": value.get("department"),
            "specialty_count": len(value.get("clinical_specialties") or []),
            "citation_count": len(value.get("citations") or []),
            "sources": _http_citations(value.get("citations"))[:3],
        }
    if step == "academic_resources_research":
        return {
            "resource_count": len(value.get("resources") or []),
            "sources": _http_citations(value.get("resources"))[:3],
            "official_count": sum(1 for r in value.get("resources") or [] if r.get("official")),
        }
    if step == "exam_analysis":
        return {
            "exam_type": value.get("exam_type"),
            "coverage_count": len(value.get("coverage") or []),
            "formats": value.get("formats") or [],
        }
    return {}


# ──────────────────────────────────────────────────────────────────────
# PLANNER HAND-OFF
# ──────────────────────────────────────────────────────────────────────

def planner_topics(report: Optional[dict], fallback: list, limit: int = 5) -> list:
    """
    The topic list /study/start builds the path from, in priority order.

    Falls back to the caller's own list when there is no report, so a client
    that never ran the intelligence pass gets exactly the behaviour it had
    before this file existed.
    """
    ordered = ((report or {}).get("study_strategy") or {}).get("ordered_topics") or []
    ordered = [t for t in ordered if isinstance(t, str) and t.strip()]
    if not ordered:
        return fallback
    return ordered[:limit]


def planner_context_block(report: Optional[dict]) -> str:
    """
    A compact briefing pasted into the path prompt so the generated nodes talk
    about HER exam. Cited facts only — no instructor information reaches the
    planner, because nothing in a study plan should be shaped by who is
    teaching the class.
    """
    if not report:
        return ""

    lines = []
    ctx = report.get("course_information") or {}
    label = " ".join(x for x in [ctx.get("courseCode"), ctx.get("courseName")] if x).strip()
    if label:
        lines.append(f"COURSE: {label}" + (f" at {ctx['school']}" if ctx.get("school") else ""))

    exam = report.get("exam_analysis") or {}
    if exam:
        if exam.get("exam_type"):
            lines.append(f"EXAM: {exam['exam_type']}")
        if exam.get("coverage"):
            lines.append("EXAM COVERS: " + "; ".join(exam["coverage"][:10]))
        if exam.get("formats"):
            lines.append("EXAM FORMAT: " + "; ".join(exam["formats"][:4]))

    course = (report.get("course_information") or {}).get("public_profile") or {}
    objectives = course.get("learning_objectives") or []
    if objectives:
        lines.append("PUBLISHED COURSE OBJECTIVES:\n" + "\n".join(f"- {o}" for o in objectives[:8]))

    priority = report.get("priority_topics") or []
    if priority:
        lines.append(
            "TOPIC PRIORITY (highest first, with why):\n"
            + "\n".join(
                f"- {p['topic']} (score {p['score']}): "
                + "; ".join(e["text"] for e in (p.get("evidence") or [])[:2])
                for p in priority[:8]
            )
        )

    if not lines:
        return ""
    return (
        "COURSE INTELLIGENCE BRIEF\n"
        "Build the path to serve this exam. Lead with the highest-priority "
        "topics. Do not claim anything about the instructor.\n\n"
        + "\n\n".join(lines)
    )
