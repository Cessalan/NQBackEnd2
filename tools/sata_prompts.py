"""
============================================
SATA (Select All That Apply) Question Generation
============================================

This module provides the prompt template and generation function for SATA questions.
It's designed to be called from the main quiz generation flow when a SATA question
is needed as part of a mixed-type quiz.

The architecture is simple:
- Each question has a "questionType" field ("mcq", "sata", "ordering", etc.)
- Frontend detects the type and renders the appropriate component
- Scoring is handled per-type by the frontend

Usage in quiztools.py:
    from tools.sata_prompts import generate_sata_question

    # When generating a mixed quiz:
    for question_type in question_types_list:
        if question_type == 'sata':
            question = await generate_sata_question(topic, difficulty, ...)
        else:
            question = await _generate_single_question(...)  # existing MCQ

@author NurseQuiz Team
@version 1.0.0
"""
import os

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from core.quiz_model import quiz_chat_model
from langchain_core.output_parsers import StrOutputParser
import json
import math
import random

# ============================================
# SATA PROMPT TEMPLATE
# ============================================

SATA_PROMPT_TEMPLATE = """
You are a {language}-speaking nursing quiz generator creating NCLEX-style SATA (Select All That Apply) questions.

Generate **EXACTLY ONE high-quality SATA question** about: {topic}

Difficulty: {difficulty}
Question number: {question_num}

CRITICAL - DO NOT repeat these questions:
{questions_to_avoid}

Context:
{content}

📋 SATA QUESTION REQUIREMENTS:

1. **Question Format:**
   - End the question with "(Select all that apply)" or the French equivalent
   - Present a realistic clinical nursing scenario
   - Include specific patient details (age, vital signs, symptoms, lab values when relevant)

2. **Options (MUST have exactly 5-6 options):**
   - Provide 5-6 plausible options (A through E or F)
   - EXACTLY {num_correct} options should be correct
   - Correct options should be clearly the best evidence-based practices
   - Incorrect options should be plausible but clinically inappropriate or less optimal
   - Mix up the order - don't cluster correct answers together

3. **Answer Array:**
   - List ALL correct options in the "answer" field as an array
   - Example: ["A) First correct option", "C) Third correct option", "E) Fifth correct option"]

4. **Justification Format:**
   - Explain why EACH option is correct or incorrect
   - Use <b>...</b> for option-label headers like "A, C, and D are correct" (visual emphasis only, NOT clickable)
   - Use <strong>...</strong> ONLY for medical terminology — drugs, conditions, signs, labs, anatomy, procedures (e.g. <strong>hypoglycemia</strong>, <strong>diaphoresis</strong>, <strong>tachycardia</strong>). Each <strong> term in the rendered UI becomes a tappable popover, so wrap the noun phrase only — never an option label, generic word, or full sentence.
   - Aim for 2–5 <strong> medical terms across the full justification.
   - Be concise but clinically accurate

🎯 TOPIC ASSIGNMENT:
- Assign a SPECIFIC topic/subject to this question
- The topic should be 2-4 words maximum in {language}
- Be specific (e.g., "Signs of Hypoglycemia" not "Diabetes")

📤 Return ONLY valid JSON (no markdown wrapper):
{{
    "question": "A 58-year-old patient with Type 2 diabetes is admitted with blood glucose of 45 mg/dL. The nurse should anticipate which of the following signs and symptoms? (Select all that apply)",
    "questionType": "sata",
    "options": [
        "A) Tremors and shakiness",
        "B) Bradycardia",
        "C) Diaphoresis",
        "D) Confusion and irritability",
        "E) Hypertension",
        "F) Pallor"
    ],
    "answer": ["A) Tremors and shakiness", "C) Diaphoresis", "D) Confusion and irritability", "F) Pallor"],
    "justification": "<b>A, C, D, and F are correct.</b> <strong>Hypoglycemia</strong> triggers the <strong>sympathetic nervous system</strong>, causing tremors, <strong>diaphoresis</strong>, and <strong>pallor</strong> due to peripheral vasoconstriction. <strong>Neuroglycopenic symptoms</strong> include confusion and irritability as the brain is deprived of glucose.<br><br><b>B (Bradycardia) is incorrect</b> because <strong>hypoglycemia</strong> causes <strong>tachycardia</strong>, not <strong>bradycardia</strong>, due to <strong>catecholamine</strong> release.<br><br><b>E (Hypertension) is incorrect</b> because while some blood pressure elevation may occur, it is not a classic or reliable sign of <strong>hypoglycemia</strong>.",
    "topic": "Signs of Hypoglycemia",
    "scoringType": "partial",
    "metadata": {{
        "sourceLanguage": "{language}",
        "questionType": "sata",
        "category": "nursing",
        "difficulty": "{difficulty}",
        "numCorrectOptions": {num_correct},
        "sourceDocument": "conversational_generation"
    }}
}}

Critical Rules:
1. The "questionType" field MUST be "sata"
2. The "answer" field MUST be an ARRAY of correct options (not a single string)
3. Include EXACTLY {num_correct} correct answers in the array
4. The "scoringType" should be "partial" for NCLEX-style scoring
5. Write everything in {language}
6. Each correct answer in the array must exactly match an option from the options list
"""

# ============================================
# KNOWLEDGE MODE SATA PROMPT TEMPLATE
# ============================================

SATA_KNOWLEDGE_PROMPT_TEMPLATE = """
You are a {language}-speaking nursing quiz generator creating KNOWLEDGE TEST SATA (Select All That Apply) questions.

Generate **EXACTLY ONE factual SATA question** about: {topic}

Difficulty: {difficulty}
Question number: {question_num}

CRITICAL - DO NOT repeat these questions:
{questions_to_avoid}

Context:
{content}

IMPORTANT - This is a KNOWLEDGE TEST, NOT an NCLEX exam:
- Ask DIRECT questions testing factual recall
- DO NOT use patient scenarios or clinical situations
- DO NOT start questions with "A nurse is caring for..." or "A patient presents with..."
- Test facts, classifications, definitions, lists, and categories
- Questions should be "Which of the following are TRUE?" style

GOOD question examples (USE THESE STYLES):
- "Which of the following are symptoms of hypoglycemia? (Select all that apply)"
- "Which medications are classified as loop diuretics? (Select all that apply)"
- "Which of the following are normal laboratory values? (Select all that apply)"
- "Which cranial nerves are involved in eye movement? (Select all that apply)"

BAD question examples (DO NOT USE - these are NCLEX-style):
- "A patient presents with low blood sugar. The nurse should anticipate..."
- "Which nursing interventions are appropriate for a patient with..."

SATA KNOWLEDGE QUESTION REQUIREMENTS:

1. **Question Format:**
   - End the question with "(Select all that apply)" or the French equivalent
   - Ask a DIRECT factual question, NOT a clinical scenario
   - Focus on facts, definitions, lists, or classifications

2. **Options (MUST have exactly 5-6 options):**
   - Provide 5-6 plausible options (A through E or F)
   - EXACTLY {num_correct} options should be factually correct
   - Incorrect options should be common misconceptions or factually wrong
   - Mix up the order - don't cluster correct answers together

3. **Answer Array:**
   - List ALL correct options in the "answer" field as an array
   - Example: ["A) First correct option", "C) Third correct option"]

4. **Justification Format:**
   - Explain why EACH option is correct or incorrect with factual reasoning
   - Use <b>...</b> for option-label headers like "A, C, and D are correct" (visual emphasis only, NOT clickable)
   - Use <strong>...</strong> ONLY for medical terminology — drugs, conditions, signs, labs, anatomy, procedures (e.g. <strong>hypoglycemia</strong>, <strong>diaphoresis</strong>). Each <strong> term in the rendered UI becomes a tappable popover, so wrap the noun phrase only — never an option label, generic word, or full sentence.
   - Aim for 2–5 <strong> medical terms across the full justification.
   - Be concise and factual

TOPIC ASSIGNMENT:
- Assign a SPECIFIC topic/subject to this question
- The topic should be 2-4 words maximum in {language}
- Be specific (e.g., "Loop Diuretics" not "Medications")

Return ONLY valid JSON (no markdown wrapper):
{{
    "question": "Which of the following are symptoms of hypoglycemia? (Select all that apply)",
    "questionType": "sata",
    "quizMode": "knowledge",
    "options": [
        "A) Tremors",
        "B) Bradycardia",
        "C) Diaphoresis",
        "D) Confusion",
        "E) Hypertension",
        "F) Pallor"
    ],
    "answer": ["A) Tremors", "C) Diaphoresis", "D) Confusion", "F) Pallor"],
    "justification": "<b>A, C, D, and F are correct.</b> Tremors, <strong>diaphoresis</strong>, confusion, and <strong>pallor</strong> are classic symptoms of <strong>hypoglycemia</strong> due to <strong>sympathetic nervous system</strong> activation and <strong>neuroglycopenia</strong>.<br><br><b>B (Bradycardia) is incorrect</b> because <strong>hypoglycemia</strong> causes <strong>tachycardia</strong>, not <strong>bradycardia</strong>.<br><br><b>E (Hypertension) is incorrect</b> because <strong>hypertension</strong> is not a typical symptom of <strong>hypoglycemia</strong>.",
    "topic": "Hypoglycemia Symptoms",
    "scoringType": "partial",
    "metadata": {{
        "sourceLanguage": "{language}",
        "questionType": "sata",
        "quizMode": "knowledge",
        "category": "nursing",
        "difficulty": "{difficulty}",
        "numCorrectOptions": {num_correct},
        "sourceDocument": "conversational_generation"
    }}
}}

Critical Rules:
1. The "questionType" field MUST be "sata"
2. The "quizMode" field MUST be "knowledge"
3. The "answer" field MUST be an ARRAY of correct options (not a single string)
4. Include EXACTLY {num_correct} correct answers in the array
5. Write everything in {language}
6. DO NOT include patient scenarios - keep it factual
"""


SATA_APPLIED_PROMPT_TEMPLATE = """
You are a {language}-speaking nursing quiz generator creating APPLIED SATA (Select All That Apply) questions.

Generate **EXACTLY ONE applied SATA question** about: {topic}

Difficulty: {difficulty}
Question number: {question_num}

CRITICAL - DO NOT repeat these questions:
{questions_to_avoid}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STUDENT'S DOCUMENT CONTENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{content}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT "APPLIED" MEANS — read this carefully, it is the whole point:

There are three rungs of nursing question. You are writing the MIDDLE one.
  1. KNOWLEDGE  - "Which of these are symptoms of hypoglycemia?"        <- NOT this
  2. APPLIED    - "A patient is started on furosemide. Which should
                   the nurse monitor? (Select all that apply)"           <- THIS
  3. NCLEX      - "K+ is 2.8 with cramps. Which action FIRST?"           <- NOT this

An APPLIED question puts a real patient with a NAMED condition, medication, device or
situation in the stem, then asks what the nurse MONITORS, DOES, TEACHES, ASSESSES or
DOCUMENTS. Each option is judged on its own merits — true or false for THIS patient.

🚨 GROUNDING RULE (non-negotiable):
The condition / medication / device / situation named in your stem MUST appear in the
document content above. You may use your own nursing knowledge for the clinical
REASONING about it — what is monitored, which actions apply — but you may NOT invent
the subject itself. If the content does not name a condition you can build on, pick a
different aspect of the same content rather than inventing one.

🚨 NO PRIORITY RANKING. These words are BANNED from the question stem:
FIRST, PRIORITY, MOST IMPORTANT, IMMEDIATE, INITIAL, BEST.
Those belong to rung 3. Every correct option here is simply correct — the student is
not ordering or ranking anything.

GOOD question examples (USE THESE STYLES):
- "A patient with systemic lupus erythematosus is seen for routine follow-up. Which should the nurse monitor? (Select all that apply)"
- "A patient is discharged home with a walker. Which instructions should the nurse include? (Select all that apply)"
- "A patient is receiving furosemide. Which findings should the nurse report? (Select all that apply)"

BAD question examples (DO NOT USE):
- "Which of the following are symptoms of hypoglycemia?" (rung 1 — no patient)
- "Which action should the nurse take FIRST?" (rung 3 — ranking)
- "A patient with sarcoidosis..." when sarcoidosis is nowhere in the content (invented subject)

SATA APPLIED QUESTION REQUIREMENTS:

1. **Question Format:**
   - Open with the patient and the named condition/medication/device from the content
   - End the question with "(Select all that apply)" or the {language} equivalent
   - Ask what the nurse monitors, does, teaches, assesses or documents

2. **Options (MUST have exactly 5-6 options):**
   - Provide 5-6 plausible nursing actions or findings (A through E or F)
   - EXACTLY {num_correct} options should be correct FOR THIS PATIENT
   - Incorrect options must be real nursing actions that are simply wrong for this
     condition — never nonsense, never actions invented to be obviously wrong
   - Mix up the order - don't cluster correct answers together

3. **Answer Array:**
   - List ALL correct options in the "answer" field as an array

4. **Justification Format:**
   - Explain why EACH option applies or does not apply to THIS patient
   - Use <b>...</b> for option-label headers like "A, C, and D are correct" (visual emphasis only, NOT clickable)
   - Use <strong>...</strong> ONLY for medical terminology — drugs, conditions, signs, labs, anatomy, procedures. Each <strong> term in the rendered UI becomes a tappable popover, so wrap the noun phrase only — never an option label, generic word, or full sentence.
   - Aim for 2–5 <strong> medical terms across the full justification.

TOPIC ASSIGNMENT:
- Assign a SPECIFIC topic/subject to this question
- The topic should be 2-4 words maximum in {language}
- Be specific (e.g., "Lupus Monitoring" not "Autoimmune")

Return ONLY valid JSON (no markdown wrapper).
The "_reasoning" field MUST come first — fill it out before writing anything else.
{{
    "_reasoning": {{
        "grounding": "The exact phrase from the document content naming the condition, medication or device this question is built on. If you cannot fill this honestly, you have invented the subject — start over with different content.",
        "applied_check": "Why this is rung 2: what the patient has, and what the nurse is being asked to monitor or do",
        "no_ranking_check": "Confirm the stem contains none of FIRST/PRIORITY/MOST IMPORTANT/IMMEDIATE/INITIAL/BEST"
    }},
    "question": "A patient with <condition from the content> ... Which should the nurse monitor? (Select all that apply)",
    "questionType": "sata",
    "quizMode": "applied",
    "options": [
        "A) First option",
        "B) Second option",
        "C) Third option",
        "D) Fourth option",
        "E) Fifth option",
        "F) Sixth option"
    ],
    "answer": ["A) First option", "C) Third option"],
    "justification": "<b>A and C are correct.</b> ... <br><br><b>B is incorrect</b> because ...",
    "topic": "Specific Topic Name",
    "scoringType": "partial",
    "metadata": {{
        "sourceLanguage": "{language}",
        "questionType": "sata",
        "quizMode": "applied",
        "category": "nursing",
        "difficulty": "{difficulty}",
        "numCorrectOptions": {num_correct},
        "sourceDocument": "conversational_generation"
    }}
}}

Critical Rules:
1. The "questionType" field MUST be "sata"
2. The "quizMode" field MUST be "applied"
3. The "answer" field MUST be an ARRAY of correct options (not a single string)
4. Include EXACTLY {num_correct} correct answers in the array
5. Write everything in {language}
6. The stem MUST contain a patient and a condition drawn from the content
7. The stem MUST NOT ask the student to rank, order, or pick what comes FIRST
"""


# ============================================
# SATA QUESTION GENERATOR
# ============================================

async def generate_sata_question(
    topic: str,
    difficulty: str,
    question_num: int,
    language: str,
    content_context: str = "",
    questions_to_avoid: list = None,
    quiz_mode: str = "knowledge"
) -> dict:
    """
    Generate a single SATA (Select All That Apply) question using LLM.

    Args:
        topic: Subject area for the question (e.g., "Hypoglycemia", "Cardiac Care")
        difficulty: Question difficulty ("easy", "medium", "hard")
        question_num: Question number in the quiz sequence
        language: Language for the question ("english" or "french")
        content_context: Optional document content to base questions on
        questions_to_avoid: List of previous questions to avoid duplication
        quiz_mode: "knowledge" for factual recall questions (default),
                   "nclex" for clinical judgment questions

    Returns:
        dict: Complete SATA question object with all required fields

    Example:
        question = await generate_sata_question(
            topic="Diabetes Management",
            difficulty="medium",
            question_num=1,
            language="english",
            quiz_mode="knowledge"  # For factual questions
        )

        # Returns:
        # {
        #     "question": "Which nursing interventions are appropriate for...",
        #     "questionType": "sata",
        #     "options": ["A) ...", "B) ...", ...],
        #     "answer": ["A) ...", "C) ..."],  # Array of correct answers
        #     "justification": "...",
        #     "topic": "Diabetes Interventions",
        #     "scoringType": "partial"
        # }
    """

    # Defensive defaults
    if questions_to_avoid is None:
        questions_to_avoid = []

    # Build question deduplication text
    if questions_to_avoid:
        avoid_text = "\n".join([f"- {q}" for q in questions_to_avoid])
    else:
        avoid_text = "None - this is the first question"

    # Randomly determine number of correct answers (2-4 for good SATA variety)
    # Easier questions have fewer correct answers, harder have more
    if difficulty == "easy":
        num_correct = random.choice([2, 2, 3])  # Weighted toward 2
    elif difficulty == "hard":
        num_correct = random.choice([3, 4, 4])  # Weighted toward 4
    else:  # medium
        num_correct = random.choice([2, 3, 3, 4])  # Balanced

    # Default content context if not provided
    if not content_context:
        content_context = f"""You are generating SATA questions about: {topic}

        Create clinically relevant scenarios that test the student's ability to:
        - Recognize multiple correct nursing actions
        - Differentiate between appropriate and inappropriate interventions
        - Apply critical thinking to select ALL correct options
        """

    print(f"\n{'='*60}")
    print(f"🎯 Generating SATA Question {question_num}")
    print(f"📚 Topic: {topic}")
    print(f"⚡ Difficulty: {difficulty}")
    print(f"✓ Target correct answers: {num_correct}")
    print(f"🌐 Language: {language}")
    print(f"🎮 Quiz mode: {quiz_mode}")
    print(f"{'='*60}\n")

    # Select template based on quiz mode. Three rungs: factual recall,
    # applied (patient + named condition, no ranking), NCLEX clinical judgement.
    if quiz_mode == "knowledge":
        template_to_use = SATA_KNOWLEDGE_PROMPT_TEMPLATE
    elif quiz_mode == "applied":
        template_to_use = SATA_APPLIED_PROMPT_TEMPLATE
    else:
        template_to_use = SATA_PROMPT_TEMPLATE

    # Create prompt
    prompt = PromptTemplate(
        input_variables=[
            "content", "topic", "difficulty", "question_num",
            "language", "questions_to_avoid", "num_correct"
        ],
        template=template_to_use
    )

    # Use GPT-4o for high-quality NCLEX questions
    llm = quiz_chat_model()  # Luna; fast tier for Pro (core/quiz_model.py)
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "content": content_context,
            "topic": topic,
            "difficulty": difficulty,
            "question_num": question_num,
            "language": language,
            "questions_to_avoid": avoid_text,
            "num_correct": num_correct
        })

        # Clean and parse JSON response
        cleaned = result.strip().strip("```json").strip("```").strip()
        parsed_question = json.loads(cleaned)

        # Validate required SATA fields
        if not isinstance(parsed_question.get('answer'), list):
            print(f"⚠️ Warning: Answer is not an array, converting...")
            # Try to convert single answer to array
            single_answer = parsed_question.get('answer', '')
            if single_answer:
                parsed_question['answer'] = [single_answer]
            else:
                raise ValueError("No answer provided in question")

        # Ensure questionType is set
        parsed_question['questionType'] = 'sata'

        # Ensure scoringType is set
        if 'scoringType' not in parsed_question:
            parsed_question['scoringType'] = 'partial'

        # Validate topic exists
        if 'topic' not in parsed_question or not parsed_question['topic']:
            parsed_question['topic'] = topic

        # Log success
        num_options = len(parsed_question.get('options', []))
        num_answers = len(parsed_question.get('answer', []))
        print(f"✅ SATA Question {question_num} generated successfully")
        print(f"   Options: {num_options}, Correct answers: {num_answers}")
        print(f"   Topic: {parsed_question.get('topic')}")

        return parsed_question

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse SATA question {question_num}: {e}")
        if 'result' in locals():
            print(f"Raw output: {result[:500]}...")
        return None

    except Exception as e:
        print(f"❌ Error generating SATA question {question_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# MIXED QUIZ GENERATION HELPER
# ============================================

def distribute_question_types(
    total_questions: int,
    question_types: list
) -> list:
    """
    Distribute question types across a quiz.

    Supported types:
    - 'mcq' - Multiple Choice Question
    - 'sata' - Select All That Apply
    - 'casestudy' - Simple ordering/case study
    - 'unfoldingCase' - 6-item unfolding case study (NGN advanced)

    Args:
        total_questions: Total number of questions to generate
        question_types: List of types to include (e.g., ['mcq', 'sata', 'casestudy'])

    Returns:
        list: List of question types in order to generate

    Example:
        types = distribute_question_types(10, ['mcq', 'sata'])
        # Returns: ['mcq', 'mcq', 'sata', 'mcq', 'mcq', 'sata', 'mcq', 'mcq', 'sata', 'mcq']

        types = distribute_question_types(2, ['mcq', 'sata'])
        # Returns: ['mcq', 'sata'] (guaranteed one of each)

        types = distribute_question_types(3, ['mcq', 'sata', 'casestudy'])
        # Returns: ['mcq', 'sata', 'casestudy'] (one of each)

        # Unfolding case studies are special - they count as 1 quiz item but have 6 internal items
        types = distribute_question_types(2, ['mcq', 'unfoldingCase'])
        # Returns: ['mcq', 'unfoldingCase']
    """

    if not question_types:
        question_types = ['mcq']

    # Reserve a matrix slot while keeping the established mix for other types.
    # Never exceed the requested count, even when there are more types than slots.
    if 'matrix' in question_types:
        types = list(dict.fromkeys(question_types))
        if total_questions <= 0:
            return []
        if len(types) == 1:
            return ['matrix'] * total_questions
        if total_questions < len(types):
            return types[:total_questions]
        count = max(1, total_questions // 4)
        result = ['matrix'] * count + distribute_question_types(total_questions - count, [t for t in types if t != 'matrix'])
        random.shuffle(result)
        return result[:total_questions]

    # Normalize case for unfoldingCase (handle both unfoldingcase and unfoldingCase)
    normalized_types = []
    for qtype in question_types:
        if qtype.lower() == 'unfoldingcase':
            normalized_types.append('unfoldingCase')  # Use consistent casing
        else:
            normalized_types.append(qtype)
    question_types = normalized_types

    print(f"📊 distribute_question_types called: total={total_questions}, types={question_types}")

    # If only one type, return all of that type
    if len(question_types) == 1:
        result = [question_types[0]] * total_questions
        print(f"📊 Single type distribution: {result}")
        return result

    # Special case: if total_questions equals number of types requested,
    # ensure exactly one of each type
    if total_questions == len(question_types):
        result = list(question_types)  # One of each
        random.shuffle(result)
        print(f"📊 Exact match distribution (1 of each): {result}")
        return result

    # For mixed types with MCQ, SATA, Case Study, and/or Unfolding Case
    result = []

    # Count how many special types we have
    has_sata = 'sata' in question_types
    has_casestudy = 'casestudy' in question_types
    has_mcq = 'mcq' in question_types
    has_unfolding = 'unfoldingCase' in question_types

    # Special handling for small quizzes (2-4 questions)
    if total_questions <= 4:
        # Ensure at least 1 of each requested type
        for qtype in question_types:
            result.append(qtype)

        # Fill remaining slots with MCQ (or first type if no MCQ)
        while len(result) < total_questions:
            fill_type = 'mcq' if has_mcq else question_types[0]
            result.append(fill_type)

        random.shuffle(result)
        print(f"📊 Small quiz distribution: {result}")
        return result

    # Handle unfolding case study specially - it's a large complex item
    # Typically only 1 unfolding case per quiz makes sense
    if has_unfolding:
        # Add 1 unfolding case
        result.append('unfoldingCase')
        remaining = total_questions - 1

        # Distribute remaining among other types
        other_types = [t for t in question_types if t != 'unfoldingCase']
        if other_types and remaining > 0:
            # Recursively distribute remaining questions
            remaining_distribution = distribute_question_types(remaining, other_types)
            result.extend(remaining_distribution)

        random.shuffle(result)
        print(f"📊 Unfolding case + others distribution: {result}")
        return result

    # For larger quizzes: distribute proportionally
    # MCQ: ~60%, SATA: ~25%, Case Study: ~15% (when all three present)
    if has_mcq and has_sata and has_casestudy:
        casestudy_count = max(1, int(total_questions * 0.15))
        sata_count = max(1, int(total_questions * 0.25))
        mcq_count = total_questions - sata_count - casestudy_count

        result = ['mcq'] * mcq_count + ['sata'] * sata_count + ['casestudy'] * casestudy_count
        random.shuffle(result)
        print(f"📊 Mixed distribution: {mcq_count} MCQ, {sata_count} SATA, {casestudy_count} Case Study = {result}")

    elif has_mcq and has_sata:
        # MCQ + SATA only
        sata_count = max(1, min(total_questions // 3, int(total_questions * 0.4)))
        mcq_count = total_questions - sata_count

        result = ['mcq'] * mcq_count + ['sata'] * sata_count
        random.shuffle(result)
        print(f"📊 MCQ+SATA distribution: {mcq_count} MCQ, {sata_count} SATA = {result}")

    elif has_mcq and has_casestudy:
        # MCQ + Case Study only
        casestudy_count = max(1, min(total_questions // 4, int(total_questions * 0.25)))
        mcq_count = total_questions - casestudy_count

        result = ['mcq'] * mcq_count + ['casestudy'] * casestudy_count
        random.shuffle(result)
        print(f"📊 MCQ+CaseStudy distribution: {mcq_count} MCQ, {casestudy_count} Case Study = {result}")

    elif has_sata and has_casestudy:
        # SATA + Case Study only
        casestudy_count = max(1, total_questions // 3)
        sata_count = total_questions - casestudy_count

        result = ['sata'] * sata_count + ['casestudy'] * casestudy_count
        random.shuffle(result)
        print(f"📊 SATA+CaseStudy distribution: {sata_count} SATA, {casestudy_count} Case Study = {result}")

    else:
        # Even distribution for other type combinations
        type_count = len(question_types)
        for i in range(total_questions):
            result.append(question_types[i % type_count])
        random.shuffle(result)
        print(f"📊 Other type distribution: {result}")

    return result


# ============================================
# QUIZ MODE DISTRIBUTION
# ============================================

# What fraction of a node's questions should be APPLIED (patient + condition,
# "what does the nurse monitor/do") rather than KNOWLEDGE (plain recall).
#
# Keyed on the planner's node difficulty, which already ramps across a plan:
# a topic runs quick-check(1) -> lesson(1) -> quiz(2) -> mini-test(2), so the
# student is taught before she is tested in applied form. Difficulty 1 keeps a
# single applied question so the rung is never completely absent.
#
# This table is deliberately BACKEND-ONLY. The frontend sends the raw node
# difficulty and never a ratio, so this does not become another mirrored
# cross-repo constant (see "Cross-repo contracts" in the frontend CLAUDE.md).
APPLIED_FRACTION = {1: 0.2, 2: 0.5, 3: 0.8}
APPLIED_FRACTION_DEFAULT = 0.2


def distribute_quiz_modes(total_questions: int, node_difficulty=None) -> list:
    """
    Decide, per question, whether it is a KNOWLEDGE or an APPLIED item.

    Mirrors distribute_question_types above: returns one entry per question, in
    generation order, INTERLEAVED rather than blocked. All the recall questions
    first and all the applied ones last would read as a difficulty cliff halfway
    through the node.

    Args:
        total_questions: How many questions the node will generate.
        node_difficulty: The planner's 1-3 difficulty on the node. Anything
                         missing or unrecognised falls back to the difficulty-1
                         fraction — the recall-leaning end, never the reverse.

    Returns:
        list[str]: "knowledge" / "applied", length == total_questions.

    Example:
        distribute_quiz_modes(5, 1)  -> 1 applied  of 5
        distribute_quiz_modes(5, 2)  -> 3 applied  of 5
        distribute_quiz_modes(5, 3)  -> 4 applied  of 5
    """
    if total_questions <= 0:
        return []

    # A real number is CLAMPED into the table's range, not dropped. Production
    # plans contain difficulty-4 nodes; looking those up and missing would hand
    # the hardest node in the plan the most recall-heavy mix — exactly backwards.
    # Only a genuinely unusable value (None, "", a list) takes the default.
    try:
        difficulty_key = max(1, min(3, int(node_difficulty)))
    except (TypeError, ValueError):
        difficulty_key = None

    fraction = APPLIED_FRACTION.get(difficulty_key, APPLIED_FRACTION_DEFAULT)

    # Explicit half-up rounding. Python's round() is banker's rounding, so
    # round(0.5 * 5) would give 2 where this gives 3.
    applied_count = int(math.floor(fraction * total_questions + 0.5))
    applied_count = max(0, min(total_questions, applied_count))
    knowledge_count = total_questions - applied_count

    if applied_count == 0:
        return ["knowledge"] * total_questions
    if knowledge_count == 0:
        return ["applied"] * total_questions

    # Interleave by spreading the minority mode across evenly-spaced slots.
    minority = "applied" if applied_count <= knowledge_count else "knowledge"
    minority_count = min(applied_count, knowledge_count)
    majority = "knowledge" if minority == "applied" else "applied"

    result = [majority] * total_questions
    step = total_questions / float(minority_count)
    for i in range(minority_count):
        slot = int(math.floor(i * step + step / 2.0))
        result[min(slot, total_questions - 1)] = minority

    # Spacing collisions can cost an item; top up so the counts stay exact.
    while result.count("applied") < applied_count:
        result[result.index("knowledge")] = "applied"
    while result.count("applied") > applied_count:
        result[result.index("applied")] = "knowledge"

    # Plain ASCII deliberately: this module is imported by standalone test
    # scripts run from a cp1252 Windows console, where an emoji in a log line
    # raises UnicodeEncodeError and takes the caller down with it.
    print(f"[quiz_modes] difficulty={node_difficulty} -> {result}")
    return result


# ============================================
# QUESTION VALIDATION
# ============================================

def validate_sata_question(question: dict) -> tuple:
    """
    Validate a SATA question has all required fields and proper format.

    Args:
        question: The question dictionary to validate

    Returns:
        tuple: (is_valid: bool, errors: list)

    Example:
        is_valid, errors = validate_sata_question(question)
        if not is_valid:
            print(f"Validation errors: {errors}")
    """
    errors = []

    # Check required fields
    required_fields = ['question', 'questionType', 'options', 'answer', 'justification']
    for field in required_fields:
        if field not in question:
            errors.append(f"Missing required field: {field}")

    # Check questionType
    if question.get('questionType') != 'sata':
        errors.append(f"questionType must be 'sata', got: {question.get('questionType')}")

    # Check options is a list with 5-6 items
    options = question.get('options', [])
    if not isinstance(options, list):
        errors.append("options must be a list")
    elif len(options) < 5 or len(options) > 6:
        errors.append(f"options should have 5-6 items, got: {len(options)}")

    # Check answer is a list
    answer = question.get('answer', [])
    if not isinstance(answer, list):
        errors.append("answer must be a list for SATA questions")
    elif len(answer) < 2:
        errors.append(f"SATA questions should have at least 2 correct answers, got: {len(answer)}")
    elif len(answer) > 4:
        errors.append(f"SATA questions should have at most 4 correct answers, got: {len(answer)}")

    # Check that all answers exist in options
    if isinstance(options, list) and isinstance(answer, list):
        for ans in answer:
            if ans not in options:
                errors.append(f"Answer '{ans}' not found in options")

    # Check question ends with SATA indicator
    question_text = question.get('question', '')
    sata_indicators = [
        '(select all that apply)',
        '(sélectionnez toutes les réponses applicables)',
        '(cochez toutes les réponses)',
        'select all that apply',
        'sélectionnez tout'
    ]
    has_indicator = any(ind.lower() in question_text.lower() for ind in sata_indicators)
    if not has_indicator:
        errors.append("Question should end with 'Select all that apply' or equivalent")

    is_valid = len(errors) == 0
    return is_valid, errors
