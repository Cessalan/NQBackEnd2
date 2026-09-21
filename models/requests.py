from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Message(BaseModel):
    role: str
    content: str

class Document(BaseModel):
    filename: str
    source: str

class StatelessChatRequest(BaseModel):
    language: str = "fr"
    chat_id: str
    input: str
    chat_history: List[Message]
    documents: List[Document]

class QuizRequest(BaseModel):
    chat_id: str
    filename: Optional[str] = None
    quiz_type: str = "mcq"
    num_questions: int = 4
    language: str = "fr"

class ScratchQuizRequest(BaseModel):
    chat_id: str
    topic: str
    difficulty: str = "medium"
    num_questions: int = 4
    quiz_type: str = "mcq"
    language: str = "fr"

class SummaryRequest(BaseModel):
    chat_id: str
    filename: str
    language: str = "fr"

class DocumentsEmbedRequest(BaseModel):
    chatId: str  # Note: camelCase for frontend compatibility
    documents: List[Document]
    
    
class GenerateTitleRequest(BaseModel):
    message:str


class RewriteRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


class PlanRequest(BaseModel):
    topic: str
    chat_id: str
    num_sections: Optional[int] = 6
    
class SectionRequest(BaseModel):
    section_title: str
    topic: str
    chat_id: str
    context: str


# ============================================================================
# STUDY MODE REQUESTS
# These support the Duolingo-style study journey feature
# ============================================================================

class StudyPlanRequest(BaseModel):
    """
    Request to generate a personalized study path.

    The AI analyzes uploaded documents and creates a learning path
    with different node types: lessons, flashcards, quizzes, and audio.
    """
    chat_id: str                          # Chat ID where documents were uploaded
    upload_ids: Optional[List[str]] = []  # Optional: specific upload IDs to focus on
    language: str = "en"                  # Language for content generation
    userPreferences: Optional[dict] = {}  # Onboarding preferences (reviewFormat, userStage, etc)
    # What the pre-plan diagnostic learned: {topic_name: percent_correct}.
    #
    # OPTIONAL ON PURPOSE. Absent — a skipped diagnostic, or a client built
    # before this existed — must reproduce the old uniform plan exactly.
    # _weight_path_by_diagnostic returns its input untouched on a falsy value,
    # so the skip path is the same code path rather than a second one to keep
    # working.
    diagnostic: Optional[dict] = None
    quickCheckId: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')

    # What the student told us about her class: school, courseCode, courseName,
    # professor, examDescription, examDate. Collected by the course-context
    # form during upload.
    courseContext: Optional[dict] = None

    # The report produced by /study/course-intelligence. Optional for the same
    # reason `diagnostic` is: a client that never ran the intelligence pass, or
    # a run that found nothing, must produce exactly the plan this endpoint
    # produced before the feature existed. See course_intelligence.planner_topics.
    courseIntelligence: Optional[dict] = None


class CourseIntelligenceRequest(BaseModel):
    """
    Request to investigate a student's specific course before her plan is built.

    `courseContext` carries what she typed (school, courseCode, courseName,
    professor, examDescription, examDate). Every field is optional and the
    service degrades one section at a time: no professor means no instructor
    search, no school means no course search, and a run with neither still
    returns a full report built from her uploaded materials.
    """
    chat_id: str
    courseContext: Optional[dict] = None
    language: str = "en"
    materials_only: bool = False


class StudyItemRequest(BaseModel):
    """
    Request to generate content for a single study node.

    Content is generated on-demand when user clicks a node,
    not all at once (saves cost, feels more dynamic).
    """
    chat_id: str                          # Chat ID for context
    node_type: str                        # "lesson" | "flashcard" | "quiz" | "audio"
    node_label: str                       # Topic/label for this node (e.g., "Cardiac Medications")
    node_id: Optional[str] = None         # Resolve saved review focus for lesson generation
    context_tags: Optional[List[str]] = []  # Tags for better context
    asked_hashes: Optional[List[str]] = []  # Previously shown content hashes (anti-repeat)
    language: str = "en"                  # Language for content
    # Diagnostic mode: the auto-launched FIRST node of a plan. Generates a short
    # calibration quiz (3 questions) instead of the standard 12. Production data
    # showed quiz-first plans completing node 1 at 67% vs 89% for lesson-first —
    # opening with a full 12-question test drives anxious students off. The
    # frontend also suppresses scoring for these.
    is_diagnostic: bool = False
    num_questions: Optional[int] = None   # Override question count (None = default)
    # The plan node's 1-3 difficulty. Sets how much of a quiz node is APPLIED
    # (a named condition from her uploads + what the nurse monitors/does) rather
    # than plain recall — see distribute_quiz_modes in tools/sata_prompts.py.
    # Raw difficulty is sent, never a ratio: the mix table stays backend-only so
    # it does not become another mirrored cross-repo constant.
    # Defaults to 1 (recall-leaning) so callers without a node — e.g. the
    # FirstLessonPane quick check — keep the gentler opening they have today.
    difficulty: Optional[int] = 1


class StudyAudioRequest(BaseModel):
    """Request to generate audio for a study node"""
    chat_id: str
    topic: str
    intent: str = "teach"
    duration: int = 2  # Duration in minutes
    language: str = "en"


class StudyReviewPlanRequest(BaseModel):
    """
    Request to generate a Phase 2 review study path based on performance data.
    Frontend sends performance from Firestore since backend has no Firebase auth.
    """
    chat_id: str
    language: str = "en"
    performance: dict = {}                     # Full studyPerformance doc from Firestore
    original_topics: Optional[List[str]] = []  # Topics from phase 1 for context


class StudyExamRequest(BaseModel):
    """
    Generate a mixed-format exam for a study session.
    Supports MCQ, SATA, and Case Study question types.
    """
    chat_id: str
    topic: str                                     # Topic this exam covers
    question_types: List[str] = ["mcq", "sata", "casestudy", "matrix"]
    question_count: int = 10
    custom_instructions: Optional[str] = None      # Student's custom instructions
    language: str = "en"
    # Separate question complexity from the planner's applied/recall mix.
    question_difficulty: Literal["easy", "medium", "hard"] = "medium"
    # The plan node's difficulty sets the applied/recall mix.
    difficulty: Optional[int] = 2
    # Defaults to "knowledge" so the two OTHER callers of /study/generate-exam
    # keep today's behaviour: the adaptive drills in ExamDrillService and the
    # NCLEX practice screen. The priority drill in particular sends instructions
    # demanding "what the nurse does FIRST", which the applied template bans —
    # so applied has to be opt-in per caller, not flipped on at the endpoint.
    quiz_mode: Optional[str] = "knowledge"


class ExamDebriefMessage(BaseModel):
    """One line of the post-exam conversation. `role` is 'user' or 'assistant'."""
    role: str
    content: str


class ExamDebriefTurnRequest(BaseModel):
    """
    One turn of the post-exam debrief.

    The whole conversation is resent every turn — there is no server-side
    session. It is at most a handful of short messages, and a student who
    reloads mid-conversation would otherwise be talking to something with
    amnesia.
    """
    messages: List[ExamDebriefMessage]
    exam_name: Optional[str] = None
    exam_date: Optional[str] = None          # ISO date, display only
    days_after: Optional[int] = None          # How fresh the memory is
    study_context: Optional[dict] = None      # Topics / completion / avg score
    language: str = "en"


class StudyInterpretRequest(BaseModel):
    """
    Interpret a student's free-text request during a study session.
    Returns an echo message (what the system understood) and a node definition.
    The student confirms before the node is created.
    """
    chat_id: str
    user_text: str                                # What the student typed
    current_topic: str = ""                       # Topic of the node she just completed
    current_node_type: str = ""                   # Type of the node she just completed
    language: str = "en"
    missed_items: Optional[List[str]] = []        # Specific questions/cards she got wrong
    score_percent: Optional[int] = None           # Her score on the node she just completed


class StudyMindmapRequest(BaseModel):
    """Request to generate a concept map for a study node"""
    chat_id: str
    topic: str
    depth: str = "medium"  # shallow | medium | deep
    language: str = "en"


class NarrationRequest(BaseModel):
    """
    Rephrase already-written tutor lines so they sound spoken rather than
    templated.

    This is a PARAPHRASE job, never a generation job. Every claim in `lines`
    was decided by deterministic code that knows what the student actually
    got right — the model is only allowed to change the wording. That
    distinction is the entire safety model here: these sentences assert
    things about her performance ("you've got that one"), and a model free
    to invent them would eventually tell someone she is strong at something
    she just failed.

    `protected_terms` are the topic names that must survive the rewrite
    intact, so validation can reject a response that renamed her subject.
    """
    chat_id: str
    lines: List[str]
    language: str = "en"
    phase: Optional[str] = None          # steady|focus|final|examDay|past|undated
    days_to_exam: Optional[int] = None
    protected_terms: Optional[List[str]] = []


class DiagnosticQuizRequest(BaseModel):
    """
    Request to generate the pre-plan diagnostic.

    Establishes the baseline the whole experience is measured against: which
    topics get taught, in what order, and — later — what "you got stronger"
    is compared to.

    Note this endpoint is deliberately NOT quota-gated. Metering a student
    before she has been shown anything of value is the worst possible first
    experience, and it would be invisible to us in testing because we are all
    on Pro accounts.
    """
    chat_id: str
    upload_ids: Optional[List[str]] = []
    language: str = "en"
    # Was being sent by the frontend wrapper and silently dropped, because the
    # model never declared it.
    userPreferences: Optional[dict] = {}
    # Topics she told us were hardest, in onboarding Q2. Guaranteed at least
    # one question each, which turns a self-report into a measurement — and
    # when the map contradicts her ("you said pharmacology, but you're solid
    # there — it's fluid balance"), that contradiction is the moment the
    # product stops feeling like a quiz generator.
    hardestTopics: Optional[List[str]] = []
    # Course priorities are not a student self-report of difficulty.
    priorityTopics: Optional[List[str]] = []


class NodeDebriefItem(BaseModel):
    """One question the student answered, with how it went."""
    question: str
    correct: bool
    question_index: Optional[int] = Field(default=None, ge=0)
    options: List[str] = Field(default_factory=list, max_length=20)
    correct_answer: str = Field(default='', max_length=2000)
    question_type: str = "mcq"                    # mcq | sata | casestudy
    rationale: str = ""                           # Why the right answer is right


class NodeDebriefRequest(BaseModel):
    """
    Post-node debrief: what went right, what went wrong, what to work on.

    Fired straight after a scored node, which is the moment the student is
    most receptive — she has just felt the misses. The items carry their
    QUESTION TYPE so the debrief can name a format pattern ("all four you
    missed were select-all-that-apply"), which is the finding that changes
    how a student studies rather than just what.
    """
    chat_id: str
    topic: str = ""
    # quiz | exam | flashcard | lesson | audio | mindmap.
    # The last three are UNSCORED — they carry no items, so the debrief drops
    # the format-pattern machinery and writes from `covered` / `struggles`
    # instead. See node_debrief's scored/unscored split.
    node_type: str = "quiz"
    score_percent: int = 0
    items: List[NodeDebriefItem] = []
    reasoning_discussions: List[dict] = Field(default_factory=list, max_length=8)
    days_until_exam: Optional[int] = None         # Sharpens the "work on" line
    language: str = "en"
    # Accumulated per-format record for the WHOLE plan, e.g.
    # [{"type": "sata", "correct": 3, "total": 12}]. Lets the debrief say
    # "this keeps happening" instead of judging a single node in isolation —
    # the difference between feedback about a quiz and feedback about her.
    plan_formats: List[dict] = []

    # ── Unscored nodes (lesson, audio, mindmap) ──────────────────────────
    # A lesson produces no right/wrong, so there is nothing to diagnose. What
    # makes a note about one worth reading is CONNECTION: tying what she just
    # studied to what her record says she has been getting wrong. All three
    # of these come from data the client already holds, so none of it is the
    # model's invention — it only writes the sentence around them.
    covered: List[str] = []       # key points / concepts the node actually covered
    struggles: List[str] = []     # concept labels she is currently missing (ledger)
    resolved: List[str] = []      # concept labels she has demonstrably fixed
    skipped: bool = False         # she tapped Skip (audio) or left the map early
    next_label: str = ""          # the planned next node, so the note can point forward
    next_type: str = ""
