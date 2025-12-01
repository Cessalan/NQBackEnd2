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

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json
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
   - Use <strong>bold</strong> for option labels
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
    "justification": "<strong>A, C, D, and F are correct.</strong> Hypoglycemia triggers the sympathetic nervous system, causing tremors, diaphoresis (sweating), and pallor due to peripheral vasoconstriction. Neuroglycopenic symptoms include confusion and irritability as the brain is deprived of glucose.<br><br><strong>B (Bradycardia) is incorrect</strong> because hypoglycemia causes tachycardia, not bradycardia, due to catecholamine release.<br><br><strong>E (Hypertension) is incorrect</strong> because while some blood pressure elevation may occur, it is not a classic or reliable sign of hypoglycemia.",
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

📌 Critical Rules:
1. The "questionType" field MUST be "sata"
2. The "answer" field MUST be an ARRAY of correct options (not a single string)
3. Include EXACTLY {num_correct} correct answers in the array
4. The "scoringType" should be "partial" for NCLEX-style scoring
5. Write everything in {language}
6. Each correct answer in the array must exactly match an option from the options list
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
    questions_to_avoid: list = None
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

    Returns:
        dict: Complete SATA question object with all required fields

    Example:
        question = await generate_sata_question(
            topic="Diabetes Management",
            difficulty="medium",
            question_num=1,
            language="english"
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
    print(f"{'='*60}\n")

    # Create prompt
    prompt = PromptTemplate(
        input_variables=[
            "content", "topic", "difficulty", "question_num",
            "language", "questions_to_avoid", "num_correct"
        ],
        template=SATA_PROMPT_TEMPLATE
    )

    # Use GPT-4o for high-quality NCLEX questions
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
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
