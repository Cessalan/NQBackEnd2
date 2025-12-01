"""
============================================
Unfolding Case Study Question Generation
============================================

WHAT IS AN UNFOLDING CASE STUDY?
--------------------------------
An unfolding case study is an advanced NCLEX NGN (Next Generation NCLEX) format where:
- A single patient scenario "unfolds" across multiple items (typically 6)
- Each item reveals new information about the patient
- The student sees clinical data in tabs (Health History, Assessment, Vital Signs, Labs)
- Questions can be different types (MCQ, SATA) within the same case

VISUAL LAYOUT:
--------------
+--------------------------------------------------+
| Unfolding Case Study              Item 1 of 6    |
+------------------------+-------------------------+
| [Tab: Health History]  |                         |
| [Tab: Assessment    ]  |  Question text here     |
| [Tab: Vital Signs   ]  |                         |
| [Tab: Lab Results   ]  |  [ ] Option A           |
|                        |  [ ] Option B           |
| Clinical data shows    |  [ ] Option C           |
| here based on active   |  [ ] Option D           |
| tab selection...       |                         |
+------------------------+-------------------------+

HOW IT DIFFERS FROM SIMPLE CASE STUDY:
--------------------------------------
- Simple Case Study: 1 scenario, 1 ordering question
- Unfolding Case Study: 1 scenario, 6 items, mixed question types, evolving data

DATA STRUCTURE:
---------------
{
    "questionType": "unfoldingCase",
    "scenario": {
        "patientInfo": "45-year-old female presents to ED...",
        "items": [
            {
                "itemNumber": 1,
                "clinicalData": {
                    "healthHistory": "Patient reports chest pain...",
                    "assessment": "Alert, anxious, diaphoretic...",
                    "vitalSigns": "BP 160/95, HR 110, RR 24...",
                    "labResults": "Troponin pending..."
                },
                "question": "Which actions should the nurse take?",
                "questionType": "sata",
                "options": ["A) ...", "B) ...", ...],
                "answer": ["A) ...", "C) ..."],
                "justification": "..."
            },
            // ... items 2-6
        ]
    }
}

USAGE:
------
    from tools.unfolding_casestudy_prompts import generate_unfolding_casestudy

    # Generate a complete 6-item unfolding case study
    case_study = await generate_unfolding_casestudy(
        topic="Acute Coronary Syndrome",
        difficulty="medium",
        language="english"
    )

@author NurseQuiz Team
@version 1.0.0
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json
import random
import logging

logger = logging.getLogger(__name__)


# ============================================
# MAIN PROMPT TEMPLATE
# ============================================
# This prompt generates ALL 6 items at once to ensure narrative consistency

UNFOLDING_CASE_PROMPT = """
You are a {language}-speaking nursing educator creating an NCLEX NGN-style Unfolding Case Study.

Create a complete 6-item unfolding case study about: {topic}
Difficulty: {difficulty}

CRITICAL RULES:
1. Generate EXACTLY 6 items that tell a cohesive patient story
2. Each item should reveal NEW clinical information as the case progresses
3. Mix question types: use SATA for items 1, 3, 5 and MCQ for items 2, 4, 6
4. The patient's condition should evolve logically (admission → assessment → intervention → evaluation)

CLINICAL DATA TABS (provide for EACH item):
- healthHistory: Patient's medical history, medications, allergies, social history
- assessment: Current nursing assessment findings (physical exam, patient statements)
- vitalSigns: Current vital signs with units (BP, HR, RR, Temp, SpO2, Pain level)
- labResults: Relevant laboratory values with reference ranges

ITEM PROGRESSION GUIDE:
- Item 1: Initial presentation - what brings patient in?
- Item 2: Assessment findings - what does the nurse observe?
- Item 3: Priority interventions - what should nurse do first?
- Item 4: Patient response - how is patient responding to interventions?
- Item 5: Complications or changes - what new issues arise?
- Item 6: Evaluation/Discharge - outcome assessment or teaching

DO NOT repeat these questions from previous quizzes:
{questions_to_avoid}

📤 Return ONLY valid JSON (no markdown wrapper):
{{
    "questionType": "unfoldingCase",
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "scenario": {{
        "patientInfo": "Brief 1-sentence patient introduction (age, gender, chief complaint)",
        "setting": "Emergency Department" or "Medical-Surgical Unit" or "ICU" etc.,
        "items": [
            {{
                "itemNumber": 1,
                "progressNote": "Brief note about what's happening at this point in care",
                "clinicalData": {{
                    "healthHistory": "Detailed health history text...",
                    "assessment": "Current assessment findings...",
                    "vitalSigns": "BP: 140/90 mmHg | HR: 88 bpm | RR: 18/min | Temp: 37.2°C | SpO2: 96% RA | Pain: 4/10",
                    "labResults": "Na: 138 mEq/L (136-145) | K: 4.2 mEq/L (3.5-5.0) | ..."
                }},
                "question": "Based on the assessment findings, which nursing actions are appropriate? (Select all that apply)",
                "questionType": "sata",
                "options": [
                    "A) First option text",
                    "B) Second option text",
                    "C) Third option text",
                    "D) Fourth option text",
                    "E) Fifth option text"
                ],
                "answer": ["A) First option text", "C) Third option text", "E) Fifth option text"],
                "justification": "<strong>Correct answers: A, C, E</strong><br>Explanation of why each is correct...<br><br><strong>Incorrect: B, D</strong><br>Explanation of why each is incorrect..."
            }},
            {{
                "itemNumber": 2,
                "progressNote": "30 minutes later...",
                "clinicalData": {{
                    "healthHistory": "Same or updated...",
                    "assessment": "New findings...",
                    "vitalSigns": "Updated vitals...",
                    "labResults": "New lab results available..."
                }},
                "question": "Which finding requires immediate nursing intervention?",
                "questionType": "mcq",
                "options": [
                    "A) Option A",
                    "B) Option B",
                    "C) Option C",
                    "D) Option D"
                ],
                "answer": "B) Option B",
                "justification": "Explanation..."
            }},
            {{
                "itemNumber": 3,
                "progressNote": "Description of current situation...",
                "clinicalData": {{
                    "healthHistory": "...",
                    "assessment": "...",
                    "vitalSigns": "...",
                    "labResults": "..."
                }},
                "question": "SATA question... (Select all that apply)",
                "questionType": "sata",
                "options": ["A)...", "B)...", "C)...", "D)...", "E)..."],
                "answer": ["A)...", "D)..."],
                "justification": "..."
            }},
            {{
                "itemNumber": 4,
                "progressNote": "...",
                "clinicalData": {{
                    "healthHistory": "...",
                    "assessment": "...",
                    "vitalSigns": "...",
                    "labResults": "..."
                }},
                "question": "MCQ question...",
                "questionType": "mcq",
                "options": ["A)...", "B)...", "C)...", "D)..."],
                "answer": "C)...",
                "justification": "..."
            }},
            {{
                "itemNumber": 5,
                "progressNote": "...",
                "clinicalData": {{
                    "healthHistory": "...",
                    "assessment": "...",
                    "vitalSigns": "...",
                    "labResults": "..."
                }},
                "question": "SATA question... (Select all that apply)",
                "questionType": "sata",
                "options": ["A)...", "B)...", "C)...", "D)...", "E)..."],
                "answer": ["B)...", "C)...", "E)..."],
                "justification": "..."
            }},
            {{
                "itemNumber": 6,
                "progressNote": "...",
                "clinicalData": {{
                    "healthHistory": "...",
                    "assessment": "...",
                    "vitalSigns": "...",
                    "labResults": "..."
                }},
                "question": "MCQ question about evaluation or discharge...",
                "questionType": "mcq",
                "options": ["A)...", "B)...", "C)...", "D)..."],
                "answer": "A)...",
                "justification": "..."
            }}
        ]
    }},
    "metadata": {{
        "sourceLanguage": "{language}",
        "questionType": "unfoldingCase",
        "category": "nursing",
        "difficulty": "{difficulty}",
        "totalItems": 6,
        "sourceDocument": "conversational_generation"
    }}
}}

CRITICAL FORMATTING RULES:
1. Each SATA question MUST end with "(Select all that apply)"
2. SATA answer MUST be an ARRAY of correct options
3. MCQ answer MUST be a SINGLE STRING (the correct option)
4. All options must start with letter and parenthesis: "A) ", "B) ", etc.
5. Clinical data should be detailed and realistic
6. Vital signs should include units and be formatted consistently
7. Lab results should include reference ranges
8. The story must be medically accurate and flow logically
9. Write everything in {language}
"""


# ============================================
# GENERATOR FUNCTION
# ============================================

async def generate_unfolding_casestudy(
    topic: str,
    difficulty: str,
    language: str = "english",
    questions_to_avoid: list = None
) -> dict:
    """
    Generate a complete 6-item unfolding case study.

    This function calls the LLM once to generate all 6 items together,
    ensuring narrative consistency throughout the case.

    Args:
        topic: Clinical topic (e.g., "Acute Coronary Syndrome", "Diabetic Ketoacidosis")
        difficulty: Question difficulty ("easy", "medium", "hard")
        language: Output language ("english" or "french")
        questions_to_avoid: List of previous questions to avoid duplication

    Returns:
        dict: Complete unfolding case study with all 6 items

    Example:
        case = await generate_unfolding_casestudy(
            topic="Heart Failure",
            difficulty="medium",
            language="english"
        )

        # Access items:
        for item in case["scenario"]["items"]:
            print(f"Item {item['itemNumber']}: {item['question'][:50]}...")

    Raises:
        ValueError: If LLM response cannot be parsed
        Exception: For other generation errors
    """

    # Default empty list if none provided
    if questions_to_avoid is None:
        questions_to_avoid = []

    # Build avoidance text for the prompt
    if questions_to_avoid:
        avoid_text = "\n".join([f"- {q}" for q in questions_to_avoid[-10:]])  # Last 10
    else:
        avoid_text = "None - this is a new case study"

    # Log the generation attempt
    print(f"\n{'='*60}")
    print(f"🏥 Generating Unfolding Case Study")
    print(f"📚 Topic: {topic}")
    print(f"⚡ Difficulty: {difficulty}")
    print(f"🌐 Language: {language}")
    print(f"📝 Generating 6 items with mixed MCQ/SATA")
    print(f"{'='*60}\n")

    # Create the prompt
    prompt = PromptTemplate(
        input_variables=["topic", "difficulty", "language", "questions_to_avoid"],
        template=UNFOLDING_CASE_PROMPT
    )

    # Use GPT-4o for complex multi-item generation
    # Temperature 0.7 balances creativity with consistency
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    try:
        # Call the LLM
        result = await chain.ainvoke({
            "topic": topic,
            "difficulty": difficulty,
            "language": language,
            "questions_to_avoid": avoid_text
        })

        # Clean the response (remove markdown code blocks if present)
        cleaned = result.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Parse the JSON
        case_study = json.loads(cleaned)

        # Validate the structure
        validation_result = validate_unfolding_casestudy(case_study)
        if not validation_result["valid"]:
            logger.warning(f"Validation warnings: {validation_result['warnings']}")
            # Try to fix common issues
            case_study = _fix_common_issues(case_study)

        # Ensure required fields
        case_study["questionType"] = "unfoldingCase"
        if "topic" not in case_study:
            case_study["topic"] = topic

        # Log success
        items = case_study.get("scenario", {}).get("items", [])
        print(f"✅ Unfolding Case Study generated successfully")
        print(f"   Patient: {case_study.get('scenario', {}).get('patientInfo', 'Unknown')[:50]}...")
        print(f"   Items: {len(items)}")

        return case_study

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse unfolding case study JSON: {e}")
        if 'result' in locals():
            logger.error(f"Raw output (first 500 chars): {result[:500]}")
        raise ValueError(f"Invalid JSON response from LLM: {e}")

    except Exception as e:
        logger.error(f"Error generating unfolding case study: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================
# VALIDATION FUNCTION
# ============================================

def validate_unfolding_casestudy(case_study: dict) -> dict:
    """
    Validate an unfolding case study has the correct structure.

    This function checks:
    - Required top-level fields exist
    - Scenario contains patient info and items
    - Each item has required fields
    - SATA answers are arrays, MCQ answers are strings
    - Clinical data tabs are present

    Args:
        case_study: The case study dictionary to validate

    Returns:
        dict: {"valid": bool, "warnings": list, "errors": list}

    Example:
        result = validate_unfolding_casestudy(case_study)
        if not result["valid"]:
            print(f"Errors: {result['errors']}")
    """
    warnings = []
    errors = []

    # Check top-level structure
    if "questionType" not in case_study:
        warnings.append("Missing questionType field")
    elif case_study["questionType"] != "unfoldingCase":
        errors.append(f"questionType must be 'unfoldingCase', got: {case_study['questionType']}")

    if "scenario" not in case_study:
        errors.append("Missing scenario field")
        return {"valid": False, "warnings": warnings, "errors": errors}

    scenario = case_study["scenario"]

    # Check scenario fields
    if "patientInfo" not in scenario:
        warnings.append("Missing patientInfo in scenario")

    if "items" not in scenario:
        errors.append("Missing items array in scenario")
        return {"valid": False, "warnings": warnings, "errors": errors}

    items = scenario["items"]

    # Check we have 6 items
    if len(items) != 6:
        warnings.append(f"Expected 6 items, got {len(items)}")

    # Validate each item
    for i, item in enumerate(items):
        item_num = i + 1

        # Required fields
        required_fields = ["itemNumber", "clinicalData", "question", "questionType", "options", "answer", "justification"]
        for field in required_fields:
            if field not in item:
                errors.append(f"Item {item_num}: Missing required field '{field}'")

        # Check clinical data tabs
        if "clinicalData" in item:
            clinical_data = item["clinicalData"]
            required_tabs = ["healthHistory", "assessment", "vitalSigns", "labResults"]
            for tab in required_tabs:
                if tab not in clinical_data:
                    warnings.append(f"Item {item_num}: Missing clinical data tab '{tab}'")

        # Validate question type specific fields
        if "questionType" in item and "answer" in item:
            qtype = item["questionType"]
            answer = item["answer"]

            if qtype == "sata":
                if not isinstance(answer, list):
                    errors.append(f"Item {item_num}: SATA answer must be an array, got {type(answer).__name__}")
                elif len(answer) < 2:
                    warnings.append(f"Item {item_num}: SATA should have at least 2 correct answers")
            elif qtype == "mcq":
                if isinstance(answer, list):
                    errors.append(f"Item {item_num}: MCQ answer must be a string, got array")

    is_valid = len(errors) == 0
    return {"valid": is_valid, "warnings": warnings, "errors": errors}


# ============================================
# HELPER FUNCTIONS
# ============================================

def _fix_common_issues(case_study: dict) -> dict:
    """
    Attempt to fix common issues in LLM-generated case studies.

    Common issues include:
    - MCQ answer as array instead of string
    - SATA answer as string instead of array
    - Missing questionType on items

    Args:
        case_study: The case study to fix

    Returns:
        dict: Fixed case study
    """
    if "scenario" not in case_study or "items" not in case_study["scenario"]:
        return case_study

    for item in case_study["scenario"]["items"]:
        qtype = item.get("questionType", "mcq")
        answer = item.get("answer")

        # Fix MCQ with array answer
        if qtype == "mcq" and isinstance(answer, list):
            item["answer"] = answer[0] if answer else ""
            logger.info(f"Fixed: Converted MCQ array answer to string for item {item.get('itemNumber')}")

        # Fix SATA with string answer
        if qtype == "sata" and isinstance(answer, str):
            item["answer"] = [answer]
            logger.info(f"Fixed: Converted SATA string answer to array for item {item.get('itemNumber')}")

    return case_study


def get_item_score(item: dict, user_answers: list) -> dict:
    """
    Calculate score for a single item in an unfolding case study.

    For MCQ: 1 point for correct, 0 for incorrect
    For SATA: Partial credit based on correct selections

    Args:
        item: The item dictionary with question and correct answer
        user_answers: List of user's selected options

    Returns:
        dict: {"score": float, "maxScore": float, "percentage": float, "correct": bool}

    Example:
        # MCQ example
        score = get_item_score(mcq_item, ["B) Correct option"])
        # Returns: {"score": 1, "maxScore": 1, "percentage": 100, "correct": True}

        # SATA example (2 of 3 correct)
        score = get_item_score(sata_item, ["A) Opt1", "C) Opt3"])
        # Returns: {"score": 0.67, "maxScore": 1, "percentage": 67, "correct": False}
    """
    qtype = item.get("questionType", "mcq")
    correct_answer = item.get("answer")

    if qtype == "mcq":
        # MCQ: Simple right/wrong
        is_correct = len(user_answers) == 1 and user_answers[0] == correct_answer
        return {
            "score": 1 if is_correct else 0,
            "maxScore": 1,
            "percentage": 100 if is_correct else 0,
            "correct": is_correct
        }

    elif qtype == "sata":
        # SATA: Partial credit scoring
        if not isinstance(correct_answer, list):
            correct_answer = [correct_answer]

        correct_set = set(correct_answer)
        user_set = set(user_answers)

        # Calculate correct and incorrect selections
        correct_selections = len(correct_set & user_set)
        incorrect_selections = len(user_set - correct_set)
        missed_selections = len(correct_set - user_set)

        # NCLEX-style scoring:
        # Full credit only if all correct selected and no incorrect
        # Partial credit for partial correct
        total_correct = len(correct_set)

        if incorrect_selections > 0:
            # Penalty for wrong selections
            score = max(0, correct_selections - incorrect_selections) / total_correct
        else:
            score = correct_selections / total_correct

        is_perfect = correct_set == user_set

        return {
            "score": round(score, 2),
            "maxScore": 1,
            "percentage": round(score * 100),
            "correct": is_perfect
        }

    else:
        # Unknown type, default to 0
        return {"score": 0, "maxScore": 1, "percentage": 0, "correct": False}


def calculate_case_score(case_study: dict, all_user_answers: list) -> dict:
    """
    Calculate total score for an entire unfolding case study.

    Args:
        case_study: The complete case study
        all_user_answers: List of answer lists, one per item
                         e.g., [["A)", "C)"], ["B)"], ...]

    Returns:
        dict: {
            "totalScore": float,
            "maxScore": int,
            "percentage": float,
            "itemScores": list of individual item scores,
            "passed": bool (>= 70%)
        }

    Example:
        result = calculate_case_score(case_study, user_answers)
        print(f"Score: {result['percentage']}% - {'Passed' if result['passed'] else 'Failed'}")
    """
    items = case_study.get("scenario", {}).get("items", [])
    item_scores = []
    total_score = 0

    for i, item in enumerate(items):
        user_answers = all_user_answers[i] if i < len(all_user_answers) else []
        item_score = get_item_score(item, user_answers)
        item_scores.append(item_score)
        total_score += item_score["score"]

    max_score = len(items)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0

    return {
        "totalScore": round(total_score, 2),
        "maxScore": max_score,
        "percentage": round(percentage, 1),
        "itemScores": item_scores,
        "passed": percentage >= 70
    }
