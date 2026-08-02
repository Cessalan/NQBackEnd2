# NurseQuizAI Orchestrator Rebuild — Python → C# (.NET) Implementation Plan

Planning pass only — no code yet. Based on a full read of `NQBackEnd2` (FastAPI) and
`ragfrontend` (React) as of 2026-08-01, branch `studysessionflow`.

---

## 1. Current-state analysis

### 1.1 What actually exists

| Piece | File(s) | Role |
|---|---|---|
| HTTP/WS host | `main.py` (5,618 lines) | 30+ REST routes + `/ws/{chat_id}` WebSocket, all handlers inline |
| "Orchestrator" | `services/orchestrator.py` (`NursingTutor`, 2,540 lines) | Regex fast-route → intent analyzer → forced tool_choice → giant if/elif tool dispatch inside one ~1,100-line async generator |
| Intent analyzer | `services/intent_analyzer.py` | Claude Sonnet 4.6, structured tool_use classification (intent, quality bar, honesty, needs_research). **This is the seed of the target `ExtractContext` tool** |
| Exam research | `services/exam_research.py` | Claude + web_search → cited brief, 24h Firestore cache |
| Tools | `tools/quiztools.py` (2,606), `flashcard_tools.py`, `audio_tools.py`, prompt libs (`sata_prompts`, `casestudy_prompts`, `unfolding_casestudy_prompts`) | LangChain `@tool` functions sharing session via a **module-level global** (`set_session_context`) |
| Question bank | `services/question_bank.py`, `quiz_with_bank.py` | Bank-first delivery, LLM top-up, background save-back, cosine-similarity dedupe |
| RAG | `services/vectorstore_manager.py` | LangChain FAISS per chat, persisted to Firebase Storage, OpenAI embeddings |
| Study mode | `/study/*` endpoints in `main.py` (SSE) | Plan generation, per-node on-demand content, diagnostic quiz, exam, mindmap, interpret-request |
| Session state | `models/session.py` (`PersistentSessionContext`) + `ACTIVE_SESSIONS` dict | In-memory, keyed by `chat_id`, rehydrated from Firestore per message (last 15 messages + quizzes) |
| Monetization | `services/usage_guard.py` (server re-check, validation only), `stripe_billing.py` (webhook, portal) | 30 question-units / rolling 3h; **charging happens client-side** |
| Misc services | studysheet (2 generators), mindmap, glossary, explain, quiz_rationale (deferred rationales), audio/TTS, recordings + Whisper STT, doc_extraction |
| Dead weight | Game mode (explicitly abandoned, kept in `main.py` + `WebSocketManager.js`), `exam_prep/` package (only `__pycache__` remains — sources deleted in the study-session redesign), disabled suggestion generation |

Model usage today is mixed: OpenAI `gpt-4.1` / `-mini` / `-nano` via LangChain for generation and routing,
Anthropic for intent analysis / research / study sheets, optional Gemini, Whisper, TTS.

### 1.2 Where it violates the target architecture

1. **No tool isolation.** Tools read/write a process-global session (`set_session_context`). The
   orchestrator generator inlines each tool's internal streaming loop (quiz, flashcards, mindmap,
   audio, summary) — the Orchestrator *is* the Item Writer and the Content Agent. Sub-agent
   internals (per-question generation chunks) leak all the way to the frontend.
2. **No state machine.** There is no session mode. Every message re-runs regex + LLM routing.
   Mode-like behavior is smeared across pattern tables (`QUIZ_PATTERNS`, continuation words in
   EN/FR, post-quiz-complaint guard) — all of which are symptoms of missing state.
3. **No Assessor.** Grading is done **client-side** (frontend computes `isCorrect` and writes it
   into the quiz doc); the backend only parses `"A)"` → `correctIndex`. There is no mastery map or
   misconception log — "weak areas" are re-derived per message by parsing the last quiz JSON.
4. **Session identity is wrong.** State is keyed by `chat_id`, not user; lives in process memory
   (`ACTIVE_SESSIONS`), so it silently resets on Cloud Run scale-out/restart. Race conditions are
   patched with retry loops (vectorstore load retries in game mode / upload).
5. **Two parallel products, duplicated pipelines.** Chat tutor (WS, NDJSON events) and study
   journey (`/study/*`, SSE) each have their own plan/quiz/flashcard/audio glue over the same
   underlying streamers, with slightly different output shapes (e.g. quiz `correctIndex`
   transformation exists twice: `orchestrator.transform_question_for_frontend` and inline in
   `/study/start`).
6. **No auth on generation endpoints.** Requests carry only `chat_id`; `usage_guard` resolves the
   owner via Firestore and *fails open*.

### 1.3 Salvageable vs. discard

**Salvage (port, don't rewrite):**
- All prompt engineering: quiz/SATA/case-study/unfolding-case prompt libraries, intent-analyzer
  system prompt (incl. the answer_question-vs-generate_quiz doctrine and frustration guard),
  empathetic-message rules, study-sheet prompts, exam-research prompts. This is years of tuned
  product behavior — move as `.md`/`.txt` prompt resources, not code.
- Firestore schema: `chats/{chatId}` + `messages` subcollection, `users/{uid}.usage`,
  question-bank collection, study session docs, research cache. Frontend reads/writes these
  directly; the C# backend must not break them.
- Wire contracts (§2) — the whole point of the migration is keeping these stable.
- Business rules: 15-item cap, quota window, deferred-rationale flow (`correct_blurb` +
  `/quiz_rationale` on demand), question dedupe, honesty preamble + `web_sources_found` flow,
  quiz-repeat guard (as a deterministic rule in the Orchestrator, exactly as today).
- Stripe webhook logic, usage_guard policy (but move charging server-side — §7).

**Discard:**
- LangChain plumbing, regex fast-router and continuation-word tables (replaced by
  `ExtractContext` + mode machine + cheap-model router), game mode, `exam_prep` orphans,
  the dual studysheet generators (keep one), disabled suggestion code.

---

## 2. Frontend contract inventory (what the C# API must not break)

Three transports, all consumed by `ragfrontend`:

### 2.1 WebSocket `/ws/{chat_id}` (chat tutor — `WebSocketManager.js`, `ChatInterface.js`)

Client → server message types: `chat_message {input, language, chat_history, documents}`,
`ping`, `cancel_stream`, `micro_rationale_request` (+ game types, dead).

Server → client envelope: `{type, ...}` where `type` ∈
`status | stream_chunk {data} | stream_complete | stream_cancelled | error {code?, message} | pong`.
`error.code === "quota_exceeded"` opens the upgrade modal — must be preserved.

`stream_chunk.data` vocabulary the UI actually handles (from `ChatInterface.js` /
`WebSocketManager.js`):

- Text: `answer_chunk`
- Quiz: `quiz_generating {current,total}`, `quiz_question {question, index, total_so_far}`,
  `quiz_complete {quiz_data, total_generated}`, `empathetic_message_start/chunk/complete`
- Flashcards: `flashcard_generating`, `flashcard_ready {flashcard, index}`, `flashcard_complete {flashcard_data}`
- Study sheet: `study_sheet_trigger {topic}`, `study_sheet_start/chunk/complete/error`
- Mindmap: `mindmap_generating`, `mindmap_complete {mindmap_data}`
- Audio: `audio_options`, `audio_generating`, `audio_script_ready`, `audio_tts_progress`,
  `audio_ready/complete/error`
- Research: `web_research_started`, `web_sources_found {citations,...}`, `web_research_failed`
- `suggested_prompts`, `error {message}`, `complete`

Question shape rendered by `StudyQuizCard`/`ChatQuiz`:
`{question, options[], correctIndex, rationale? (legacy HTML), correctBlurb?, topic, metadata?}`.
Flashcard: `{front, back, topic}`.

### 2.2 NDJSON POST streams

- `/chat/upload-files` (multipart) → newline-JSON events: `batch_start`, `loading_existing_documents`,
  `file_start/complete/error`, `embedding_start/progress/complete`, `insight_batch`,
  `firebase_start/complete`, `upload_summary`, `post_upload_message`, `all_complete`, heartbeats
  every 10s (frontend has a 45s stall watchdog — timing contract is load-bearing).
- `/chat/generate-summary` → same reader.

### 2.3 REST + SSE (`FastAPICalls.js`)

Called and live: `/study/start` (SSE: `session_ready → plan_generating → plan_ready {plan} →
first_node_generating → question_ready|flashcard_ready* → first_node_ready|first_node_skipped →
complete`), `/study/plan`, `/study/plan-review`, `/study/diagnostic-quiz`, `/study/generate-item`,
`/study/generate-item-stream`, `/study/generate-exam`, `/study/generate-audio`,
`/study/generate-mindmap`, `/study/interpret-request`, `/plan`, `/generate-section`, `/search`,
`/glossary`, `/explain`, `/quiz_rationale`, `/chat/generate-title`, `/chat/rewrite`,
`/speech-to-text`, `/recordings/*`, `/files/proxy/*`, `/warm_up`, `/billing/create-portal-session`,
`/billing/webhook` (Stripe), `/admin/import-questions`, `/admin/question-bank-stats`.

Note: `StudySessionService.js` (the split-panel tutor state) is Firestore-only — session/node state
lives client-side in Firestore; the backend is stateless content generation for it today.

### 2.4 Frontend calls with **no backend route** (found mismatches)

| Frontend caller | Endpoint | Status |
|---|---|---|
| `ask_llm_stream` (FastAPICalls.js:81) | `POST /chat/stream` | Legacy HTTP chat path; still imported by ChatInterface but superseded by `ask_llm_websocket`. Would 404. **Delete.** |
| `embed_documents` (:208) | `POST /chat/embed` | No route. Dead. **Delete.** |
| `generate_quiz` (:577) / `generate_flashcards` (:610) | `/chat/generate-quiz`, `/chat/generate-flashcards` | No routes; not imported anywhere. **Delete.** |
| `generate_scenario` (:642) | `POST /chat/generate-scenario` | No route, but **still invoked** by `ChatInterface.handleScenario` (line ~3885). Either the UI entry point is unreachable or this 404s in prod. **Resolve before migration** (remove or reimplement). |
| `submit_study_answer` (:1810) | `POST /study/submit-answer` | No route; not imported. Interesting: this is exactly the future Assessor endpoint. **Delete now, resurrect as Assessor contract in Phase 5.** |
| `config/billing.js` | `/billing/create-checkout-session` | Commented out (payment-link flow used instead). Confirm intended state. |

---

## 3. Proposed C# solution structure

Single deployable (Cloud Run container), five projects — enough separation to enforce the
isolation principle via project references, without microservice overhead:

```
NurseQuiz.sln
├── src/
│   ├── NurseQuiz.Api/                      ← ASP.NET Core host (the only web project)
│   │   ├── Endpoints/                      (minimal APIs, grouped: Chat, Study, Upload, Billing, Utility, Admin)
│   │   ├── Websockets/ChatSocketHandler.cs (envelope protocol, cancellation, idle timeout, quota gate)
│   │   ├── Streaming/                      (NdjsonWriter, SseWriter — one serializer for all transports)
│   │   └── Program.cs                      (DI composition root)
│   │
│   ├── NurseQuiz.Contracts/                ← wire DTOs, frozen to §2 (no logic, no deps)
│   │   ├── Events/                         (StreamEvent hierarchy: QuizQuestionEvent, FlashcardReadyEvent, …)
│   │   ├── Items/                          (QuizQuestion, Flashcard, CaseStudy, MindmapData, StudyPlanNode)
│   │   └── Requests/                       (ports of models/requests.py)
│   │
│   ├── NurseQuiz.Orchestration/            ← the Orchestrator agent + spine
│   │   ├── SessionState/                   (SessionState, MasteryMap, MisconceptionLog, ContentScope, IntentContext)
│   │   ├── Modes/                          (SessionMode enum, ModeTransitionTable, IModeGate)
│   │   ├── Orchestrator.cs                 (agent loop; yields IAsyncEnumerable<StreamEvent>)
│   │   ├── Tools/                          (ExtractContext, SetSessionMode, GetMasteryMap, SummarizeGaps,
│   │   │                                    BuildStudyPlan, ScheduleReview — orchestrator-owned tools)
│   │   └── Guards/                         (QuizRepeatGuard, QuotaGuard, SafetyRouter — deterministic, pre-LLM)
│   │
│   ├── NurseQuiz.Agents/                   ← the three sub-agents, each in its own namespace,
│   │   │                                     referencing Contracts + Abstractions only (NOT Orchestration)
│   │   ├── Abstractions/                   (IAgent, ITool, AgentResult<T>, AgentToolAdapter)
│   │   ├── Content/                        (ContentAgent: IngestDocument, ExtractExamMetadata,
│   │   │                                    ReconcileSources, RetrieveChunks, MapToBlueprint)
│   │   ├── ItemWriter/                     (ItemWriterAgent: GenerateQuestion/CaseStudy/Flashcards,
│   │   │                                    ValidateItem critic loop, QuestionBank integration)
│   │   ├── Assessor/                       (AssessorAgent: GradeStructured, GradeFreeText,
│   │   │                                    LogMisconception, UpdateMastery)
│   │   └── Prompts/                        (*.md prompt resources ported from Python, per-agent, per-locale)
│   │
│   └── NurseQuiz.Infrastructure/           ← all I/O
│       ├── Firestore/                      (SessionStateStore, ChatContextReader, QuestionBankStore,
│       │                                    ResearchCache, UsageStore — Google.Cloud.Firestore)
│       ├── Storage/                        (chunk/embedding persistence — Google.Cloud.Storage.V1)
│       ├── Retrieval/                      (IVectorSearch: embedding client + in-process cosine search; §5.4)
│       ├── Llm/                            (IChatCompletion abstraction; AnthropicClient primary,
│       │                                    OpenAiClient secondary; tool-schema generation)
│       ├── Billing/                        (Stripe webhook/portal)
│       └── Media/                          (TTS, Whisper STT, file proxy)
│
└── tests/
    ├── NurseQuiz.Contracts.Tests/          (golden-transcript replay: recorded Python streams must deserialize
    │                                        and re-serialize byte-compatibly)
    ├── NurseQuiz.Orchestration.Tests/      (mode machine, guards, tool filtering — pure unit tests, fake LLM)
    └── NurseQuiz.Agents.Tests/             (ValidateItem critic on fixture items, grading rubrics)
```

Dependency rule (enforced by project refs): `Api → Orchestration → Agents.Abstractions ← Agents`,
everything → `Contracts`, only `Api` + concrete agents → `Infrastructure`. Sub-agents cannot see
`SessionState` — they receive a scoped `AgentRequest` and return an `AgentResult`, period.

---

## 4. Interface design & DI

### 4.1 Core abstractions

```csharp
// A tool the LLM can call. Schema is generated from TArgs, not handwritten.
public interface ITool
{
    string Name { get; }
    string Description { get; }
    JsonNode ParameterSchema { get; }                 // via JsonSchemaExporter (§5.2)
    IReadOnlySet<SessionMode> AllowedModes { get; }   // mode-based tool filtering
    IAsyncEnumerable<StreamEvent> InvokeAsync(ToolCall call, ToolContext ctx, CancellationToken ct);
}

// A sub-agent: internally runs its own LLM loop with its own system prompt + tool subset.
public interface IAgent<TRequest, TResult>
{
    Task<AgentResult<TResult>> RunAsync(TRequest request, CancellationToken ct);
}

// AgentResult is the "single clean JSON result or structured failure" from the brief:
public sealed record AgentResult<T>(bool Success, T? Value, AgentFailure? Failure, AgentUsage Usage);
```

**Sub-agent-as-tool** is one adapter class, `AgentToolAdapter<TReq, TRes>`, which maps a tool call
from the Orchestrator's LLM onto `IAgent.RunAsync` and yields (a) optional progress events
(`quiz_generating`, `question_ready` — progress is part of the *product contract*, so the adapter
forwards a bounded, typed progress channel, never the sub-agent's reasoning) and (b) the final
artifact as the tool result appended to the Orchestrator's message list. The ItemWriter's
critic/retry loop stays entirely inside `ItemWriterAgent`.

### 4.2 Session state (the spine)

```csharp
public sealed class SessionState        // persisted @ Firestore: sessions/{chatId}
{
    public string SessionId, UserId;    // UserId resolved once from chats/{chatId}.userId (or auth token)
    public string Locale;               // replaces per-message language detection: detect once, store
    public ExamTrack Track;             // course_exam | nclex_rn | ngn | oiiq | general
    public SessionMode Mode;            // intake | triage | planning | targeted_drill | simulation | debrief | spaced_review
    public IntentContext Context;       // last ExtractContext output (urgency, scope, goal, confidence)
    public ContentScope Scope;          // uploads, pasted specs, bank
    public MasteryMap Mastery;          // per-topic {seen, correct, confidence, lastSeen}
    public List<Misconception> Misconceptions;
    public string? StudyPlanId; public List<string> FlashcardDeckIds;
    public int Version;                 // optimistic concurrency
}

public interface ISessionStateStore
{
    Task<SessionState> GetOrCreateAsync(string chatId, CancellationToken ct);
    Task SaveAsync(SessionState state, CancellationToken ct);   // Firestore transaction on Version
}
```

Backed by `FirestoreSessionStateStore` + `IMemoryCache` read-through (60s TTL). This replaces
`ACTIVE_SESSIONS` and survives multi-instance Cloud Run: memory is a cache, Firestore is truth.
Conversation history keeps living where it lives today (`chats/{chatId}/messages`, written by the
frontend) and is read via `ChatContextReader` exactly like `get_chat_context_from_db` — no schema
change in Phase 1–3.

### 4.3 Mode state machine

```csharp
public static class ModeTransitions
{
    // Explicit allowed-transition table, validated server-side on every SetSessionMode:
    // INTAKE   → TRIAGE | PLANNING | TARGETED_DRILL
    // TRIAGE   → PLANNING | TARGETED_DRILL
    // PLANNING → TARGETED_DRILL | SIMULATION
    // TARGETED_DRILL → SIMULATION | DEBRIEF | SPACED_REVIEW
    // SIMULATION → DEBRIEF
    // DEBRIEF  → TARGETED_DRILL | SPACED_REVIEW | PLANNING
    // SPACED_REVIEW → TARGETED_DRILL | SIMULATION
}
```

`SetSessionMode` is an orchestrator tool; an illegal transition returns a structured tool error
(the model self-corrects; the server never trusts model judgment for legality). The tool list
handed to the Orchestrator's LLM each turn is `tools.Where(t => t.AllowedModes.Contains(state.Mode))`
— this is how per-turn tool count stays small.

Mapping today's product onto modes (so existing behavior doesn't regress): free chat quiz/flashcard
requests = `TARGETED_DRILL`; `/study/start` plan flow = `INTAKE → PLANNING`; diagnostic quiz =
`TRIAGE`; `/study/generate-exam` = `SIMULATION`; post-quiz "practice weak areas" = `DEBRIEF →
TARGETED_DRILL`. `SPACED_REVIEW` is net-new (SM-2 `ScheduleReview`).

### 4.4 Intent & urgency detection (`ExtractContext`)

Direct port of `intent_analyzer.py` to a **forced tool_use** call on Claude with the strict schema
from the brief (`urgency`, `time_available_minutes`, `scope`, `stated_weaknesses`,
`confidence_level`, `goal`, `session_mode`), merged with the analyzer's existing fields
(`honesty_required`, `needs_research`, `document_action`, `tool_args_overrides`). Runs on:
first message, any message the cheap pre-classifier flags as goal-stating, and signal triggers
(bad-answer streak from the Assessor). The existing skip heuristics (`should_skip_analysis`)
carry over as a cheap gate. The crisis/safety branch routes to a standalone `SafetyRouter`
*before* the tutoring loop, per the brief.

### 4.5 DI wiring (Program.cs sketch)

```csharp
builder.Services
    .AddSingleton<FirestoreDb>(...)                     // one client, credentials from env
    .AddSingleton<ISessionStateStore, FirestoreSessionStateStore>()
    .AddSingleton<IChatCompletion, AnthropicChatCompletion>()   // keyed services for OpenAI where retained
    .AddSingleton<IVectorSearch, InProcessCosineSearch>()
    .AddSingleton<IQuestionBank, FirestoreQuestionBank>()
    .AddScoped<ContentAgent>().AddScoped<ItemWriterAgent>().AddScoped<AssessorAgent>()
    .AddScoped<Orchestrator>()
    // tools discovered by scanning for ITool; sub-agents wrapped:
    .AddScoped<ITool, ExtractContextTool>() /* … */
    .AddScoped<ITool>(sp => AgentToolAdapter.For(sp.GetRequiredService<ItemWriterAgent>(), progress: QuizProgressMap));
```

Scoped lifetime per WS message / HTTP request; `SessionState` loaded once per scope and saved once
at scope end (plus incremental saves after mastery updates).

Exam track profiles = config objects (`appsettings.json` section `ExamTracks:{nclex_rn,ngn,oiiq,course_exam}`
→ `IOptionsMonitor<ExamTrackProfile>`): format distribution, cognitive-level mix, blueprint id,
rationale style, language/terminology directives. Agents receive the resolved profile in their
request — no per-track code paths.

---

## 5. LLM integration in C#

### 5.1 SDK choice

- **Primary: official Anthropic C# SDK** (`Anthropic` on NuGet) for the Orchestrator loop,
  ExtractContext, exam research (server-side `web_search` tool), and ItemWriter/Assessor. It gives
  typed streaming events, tool_use blocks, and prompt caching (`cache_control`) — the analyzer's
  cost model depends on caching a stable system prefix, keep that.
- **Secondary: `OpenAI` official .NET SDK** only for what stays on OpenAI initially (gpt-4.1
  generation prompts port as-is; embeddings; Whisper; TTS). Long-term consolidation is a product
  decision (§8), so the code depends on our own `IChatCompletion`/`IEmbeddingClient` interfaces,
  and each agent's model is config, not code.
- Deliberately **not** Semantic Kernel/LangChain-for-.NET: the whole point is an explicit,
  auditable loop; frameworks re-introduce the magic being removed.

### 5.2 JSON schema generation from C# types

.NET 9's `JsonSchemaExporter.GetJsonSchemaAsNode(JsonSerializerOptions, typeof(TArgs))` generates
tool parameter schemas from the same records used to deserialize tool calls — one source of truth,
no drift. Descriptions come from `[Description]` attributes. Tool dispatch:
`JsonSerializer.Deserialize<TArgs>(toolUse.Input)` → typed handler; serializer errors become
structured tool_result errors so the model can retry.

### 5.3 Streaming pipeline

Everything internal is `IAsyncEnumerable<StreamEvent>` (a polymorphic record hierarchy serialized
with a `status` discriminator matching §2 exactly). The Api layer owns transport framing:
`ChatSocketHandler` wraps events in `{type:"stream_chunk", data:…}`; `SseWriter` emits
`data: {...}\n\n`; `NdjsonWriter` emits line-JSON — so the same Orchestrator serves all three
transports and the Python event vocabulary is preserved verbatim. Cancellation: WS `cancel_stream`
→ `CancellationTokenSource` per in-flight message (proper token propagation replaces the Python
poll-a-flag approach). Heartbeats: a `System.Threading.PeriodicTimer` merge on upload/study streams
keeps the 10s heartbeat contract with the frontend's 45s watchdog.

### 5.4 Retrieval (FAISS has no .NET reader — decision required)

FAISS index files in Firebase Storage cannot be read from C#. Recommended replacement:
**store chunks + raw embedding vectors** (per chat partition) in Storage as a simple binary/JSON
blob and do **in-process cosine top-k** in C#. Per-chat corpora are small (a handful of documents,
hundreds–low-thousands of chunks), so brute-force is microseconds and removes a dependency; this is
also exactly the isolation the Content Agent needs (session-scoped partition). Migration path:
new uploads write the new format; old chats lazily re-embed on first post-migration retrieval
(embedding cost is trivial at this scale). Alternative if corpora grow: Firestore vector search
(`FindNearest`) — same interface (`IVectorSearch`), swap the implementation.

---

## 6. Migration sequencing (strangler, no big bang)

**Phase 0 — Contract freeze (before any C#).**
Record golden transcripts of live streams (chat quiz, flashcards, upload, /study/start, error and
quota paths) from the Python backend into test fixtures. Write `NurseQuiz.Contracts` from §2 and a
replay test: every recorded line must round-trip through the C# DTOs. Also resolve the dead
frontend callers (§2.4) now — they only add noise to the contract.

**Phase 1 — C# host in front, utility endpoints ported.**
Stand up `NurseQuiz.Api` with YARP reverse-proxying everything to Python. Port the stateless,
low-risk endpoints natively: `/glossary`, `/explain`, `/quiz_rationale`, `/chat/generate-title`,
`/chat/rewrite`, `/warm_up`, `/billing/*`, `/admin/*`, `/files/proxy`. One deployable URL for the
frontend from day one; each route flips from proxy → native individually and can flip back.
*Exit criterion: prod traffic through C# host, zero frontend changes.*

**Phase 2 — Session spine + Orchestrator skeleton on the WebSocket.**
Implement `SessionState`/store, mode machine, quota guard, `ExtractContext` (analyzer port),
`SafetyRouter`, and the WS handler. The Orchestrator handles `respond_directly`/conversation turns
natively (plain streaming answer); for content-generation tools it initially **delegates to the
Python service via internal HTTP as if Python were the sub-agents** (the sub-agent-as-tool
abstraction makes this a legitimate intermediate state, not a hack). Deterministic guards
(quiz-repeat, quota) move here.
*Exit criterion: Python no longer owns routing or session state; it only generates content.*

**Phase 3 — Item Writer agent.**
Port quiz/SATA/case-study/flashcard/study-sheet generation with the two-pass
Generate→ValidateItem→regenerate loop (net-new quality gate), question-bank read/write, dedupe,
deferred-rationale (`correctBlurb`) behavior, empathetic message, and exam-research grounding.
`/study/generate-item(-stream)`, `/study/generate-exam`, `/study/diagnostic-quiz` move over here
too, emitting the same SSE events. Retire the Python quiz paths.
*Exit criterion: all item generation in C#; bank hit-rates and golden transcripts match.*

**Phase 4 — Content agent + upload pipeline.**
Port `/chat/upload-files` (PDF/OCR extraction, chunking, embeddings → new storage format §5.4,
insights extraction, NDJSON progress + 10s heartbeats), `/search`, summaries, mindmap,
`ExtractExamMetadata` + `ReconcileSources` (pasted-teacher-text mining is net-new but slots into
the same ingestion path), `/study/plan` + `/study/start` plan generation, audio/STT/recordings.
*Exit criterion: Python receives no traffic; decommission it (Phase 6 is just deleting the proxy).*

**Phase 5 — Assessor + mastery (the net-new capability, needs frontend work).**
Add `POST /assess/answer` (resurrecting the frontend's orphaned `submit_study_answer` shape:
`{chat_id, node_id, answer, question} → {isCorrect, rationale, xpEarned}`) and free-text rubric
grading. Frontend switches from local `isCorrect` computation to the Assessor per answer (batch
fallback at quiz end for offline). Assessor writes MasteryMap + MisconceptionLog; `SummarizeGaps`
and `BuildStudyPlan`/`ScheduleReview` (SM-2) light up `DEBRIEF`/`SPACED_REVIEW` modes. Server-side
quota **charging** also moves here (single writer, closes the client-side-charging gap).

Ordering rationale: each phase ships independently, the riskiest pure-port (Item Writer) happens
while Python is still one env-var away as fallback, and the only phase requiring coordinated
frontend changes (5) is last and additive.

---

## 7. Contract/design mismatches to resolve before or during the build

1. **Dead frontend callers** (§2.4) — delete `/chat/stream`, `/chat/embed`, `/chat/generate-quiz`,
   `/chat/generate-flashcards`; fix or remove `handleScenario` (currently calls a nonexistent
   endpoint); confirm checkout-session strategy.
2. **Client-side grading & charging.** Grading moves to Assessor (Phase 5). Quota charging today is
   `consumeGeneration()` in the browser with a validate-only server guard — flip to server-side
   charge + client mirror.
3. **No auth on generation/WS endpoints.** Add Firebase ID-token verification middleware in
   `NurseQuiz.Api` (frontend already has the token). Resolve `UserId` from the token, not from
   `chats/{chatId}.userId` lookups. Do it in Phase 2 while the WS handler is being written; keep a
   grace mode during rollout.
4. **`correctIndex` parsed from `"A)"` strings** in two places — Item Writer must emit
   `correctIndex` natively; keep the string `answer` only for bank back-compat.
5. **Language detection per session** — store `Locale` in `SessionState` at creation (frontend
   already knows the UI language; pass it explicitly instead of LLM-detecting).
6. **Duplicate study-sheet generators and quiz-shape transforms** — one canonical implementation
   each in C#.
7. **`chat_id`-keyed in-memory sessions** — replaced by the Firestore-backed store (§4.2); removes
   the Cloud Run scale-out bug class and the vectorstore retry loops.

---

## 8. Open questions / assumptions made

1. **Where is the existing ASP.NET Core / Blazor backend?** The brief says to integrate with it,
   but neither working directory contains a C# project. Assumed for now: new solution, deployed as
   its own Cloud Run service replacing the Python container URL. If the Blazor app should host it,
   `NurseQuiz.Api` folds into that host and only `Program.cs` wiring changes — the project split
   was chosen to make that cheap. **Please point me at the repo.**
2. **Model consolidation.** Plan assumes: Claude for orchestration/analysis/validation (already
   true for the analyzer), keep gpt-4.1 prompts for item generation initially (lowest-risk port),
   revisit per-agent models via config afterward. OK?
3. **Retrieval replacement** (§5.4): in-process cosine over stored embeddings, lazy re-embed of old
   chats. Confirm acceptable vs. adopting Firestore vector search now.
4. **NGN multi-step case state & partial credit** — current `casestudy` items are single-shot with
   client-side ordering checks. Treating full NGN case-state (and OIIQ French track content) as
   *post-migration feature work* on the ExamTrack profile system, not part of the port. Confirm.
5. **Recordings/Whisper, TTS audio, mindmaps, `/plan`+`/generate-section` study-guide generator** —
   assumed to be kept features (all have live frontend callers) and are scheduled in Phase 4.
   Game mode assumed permanently dead (will not be ported).
6. **Firestore writes from frontend** (messages, quiz results, study session docs) — assumed to
   stay as-is through Phase 4 to avoid touching the frontend; Assessor phase revisits who writes
   quiz results.
7. **Blueprint/competency frameworks** (`MapToBlueprint`) — no blueprint data exists in the repo
   today beyond `constants/nclex_classification.py`; assumed that file seeds the NCLEX blueprint
   config and other tracks start empty.

---

## 9. Motivating failure case (acceptance test for the rebuild)

Real user prompt (verbatim intent): *"Make me 100 NCLEX-style application-level MCQs from my
resources and this exam blueprint (Modules 1&2: 23q, 3&4: 20q, 5&6: 27q, 7&8: 20q, dosage calc:
10q), no answers shown, interactive, final score after I submit."* The app failed her completely.

Why it fails today, layer by layer:
1. `generate_quiz_stream` hard-caps at 15 (`quiztools.py:1148`) and the system prompt counteroffers 15.
2. The blueprint is flattened into a single `topic` string — per-section counts have nowhere to go.
3. Chat quiz UI gives instant per-question feedback; no deferred-feedback exam mode (though
   `reviewMode`/`QuizResults`/`QuizResultsAnalytics` already contain most of the UI machinery).
4. `/study/generate-exam` exists but is Study-Mode-only, capped 5–20, unreachable from chat.
5. **No graceful degradation**: nothing detects "request exceeds capabilities → propose an
   alternative plan." She got a silent failure instead of a negotiation. Quota (30 units/3h free)
   compounds it.

Target-architecture handling (this prompt must pass end-to-end after Phase 5):
- `ExtractExamMetadata` parses the pasted blueprint → `ExamSpec {sections[{topics, count}], format,
  cognitiveLevel, deferredFeedback}`.
- Orchestrator enters `SIMULATION`; exam assembly is **deterministic** (distribute counts — no LLM
  judgment), sections generated by the ItemWriter in parallel validated batches with bank reuse;
  user starts section 1 while later sections generate.
- Frontend exam player runs deferred-feedback; the **Assessor** grades at submit → final score +
  per-module breakdown → MasteryMap → `DEBRIEF` offers targeted drills on the weakest module.
- If the request still exceeds a limit (cost, quota, tier), the Orchestrator must respond with a
  concrete counter-plan (sectioned delivery, Pro gate), never a silent clamp.

Interim fix on the Python stack (optional, pre-rebuild): add an `exam_request` intent +
blueprint extraction to the analyzer; stream a counter-plan via the existing honesty-preamble
path; per-section batched generation over `stream_quiz_with_bank` with an `exam_section_ready`
event; `deferredFeedback` prop on ChatQuiz. Cheapest first step is the counter-plan alone —
prompt-only change, turns the hard failure into a guided flow.
