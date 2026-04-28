"""
============================================
CASE STUDY / BOWTIE QUESTION GENERATION
============================================

This module provides the prompt template and generation function for NGN-style
Case Study questions with drag-and-drop ordering. These questions present
clinical scenarios with tabbed data (Nurses' Notes, Vital Signs, Lab Results)
and require students to arrange nursing actions in the correct sequence.

The architecture:
- Each question has "questionType": "casestudy"
- Frontend renders tabs for clinical data and sortable items
- Scoring uses partial credit for each correctly positioned item

Usage in quiztools.py:
    from tools.casestudy_prompts import generate_casestudy_question

    # When generating a mixed quiz:
    for question_type in question_types_list:
        if question_type == 'casestudy':
            question = await generate_casestudy_question(topic, difficulty, ...)

@author NurseQuiz Team
@version 1.0.0
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json
import random
import uuid

# ============================================
# CASE STUDY PROMPT TEMPLATE
# ============================================

CASESTUDY_PROMPT_TEMPLATE = """
You are a {language}-speaking nursing quiz generator creating NCLEX NGN-style Case Study questions.

Generate **EXACTLY ONE high-quality Case Study question** about: {topic}

Difficulty: {difficulty}
Question number: {question_num}

CRITICAL - DO NOT repeat these questions:
{questions_to_avoid}

Context:
{content}

📋 CASE STUDY QUESTION REQUIREMENTS:

1. **Clinical Scenario (caseStudy object):**
   Create a realistic scenario. Include ONLY the tabs that are relevant to the topic:

   - **nursesNotes** (ALWAYS REQUIRED): Narrative from the nurse's perspective.
     For CLINICAL topics (patient care, medications, procedures): include patient demographics, chief complaint, medical history, current symptoms, observations, recent events.
     For NON-CLINICAL topics (conflict management, delegation, communication, ethics, leadership, professionalism, interdisciplinary collaboration, patient education): describe the workplace situation, the people involved, relevant context, and the challenge at hand — written as a scenario narrative.
     Use HTML formatting (<p>, <strong>, timestamps or section headers)

   - **vitalSigns** (OPTIONAL — ONLY if clinically relevant): Include ONLY when the scenario involves a patient with a medical condition where vital signs directly inform nursing decisions.
     Present as HTML table with current and previous readings (Temperature, HR, BP, RR, SpO2). Include at least 2 time points.
     Use <table><tr><th>/<td> tags.
     ⚠️ OMIT this tab entirely for topics like conflict resolution, delegation, communication, ethics, leadership, professionalism, or other non-clinical scenarios. Do NOT fabricate vital signs.

   - **labResults** (OPTIONAL — ONLY if clinically relevant): Include ONLY when lab values are meaningful to the clinical decision-making in the scenario.
     Present as HTML table with results and reference ranges. Flag abnormal values.
     Use <table><tr><th>/<td> tags.
     ⚠️ OMIT this tab entirely for non-clinical scenarios. Do NOT fabricate lab results.

   IMPORTANT: The caseStudy object must ONLY contain keys for tabs you actually include. Do NOT include keys with empty or placeholder values. If only nursesNotes is relevant, the caseStudy object should have only that one key.

2. **Question (ordering task):**
   - Ask students to prioritize or sequence nursing actions
   - Be specific about what needs to be ordered (e.g., "prioritize", "sequence", "order of priority")
   - Example: "Place the following nursing actions in order of priority."

3. **Options (4-6 items to order):**
   - Each option needs a unique "id" and "text"
   - Options should be initially presented in a SCRAMBLED order (not the correct order)
   - Each item should be a distinct, complete nursing action
   - Items should be plausible and relevant to the scenario

4. **Correct Order (correctOrder array):**
   - List items in the CORRECT sequence
   - Each item has "id" and "text" matching the options
   - Order should follow evidence-based nursing priorities (ABCs, Maslow's, etc.)

5. **Justification:**
   - Explain WHY this specific order is correct
   - Reference nursing priority frameworks (ABCs, Maslow's hierarchy)
   - Use <strong>bold</strong> for emphasis
   - Be concise but clinically accurate

🎯 TOPIC ASSIGNMENT:
- Assign a SPECIFIC topic/subject to this question
- The topic should be 2-4 words maximum in {language}
- Be specific (e.g., "Hypoglycemia Priority Actions" not "Diabetes")

📤 Return ONLY valid JSON (no markdown wrapper):
{{
    "question": "Place the following nursing actions in order of priority for this patient.",
    "questionType": "casestudy",
    "caseStudy": {{
        "nursesNotes": "<p><strong>1400:</strong> 58-year-old female admitted to medical-surgical unit with chief complaint of chest pain. Patient reports pain began 2 hours ago, described as 'pressure' radiating to left arm. History of Type 2 DM, HTN, and hyperlipidemia. Patient appears anxious and diaphoretic.</p><p><strong>Nurse Assessment:</strong> Patient is alert and oriented x4. Skin is cool and clammy. States pain is 8/10 on numeric scale.</p>",
        "vitalSigns": "<table><tr><th>Parameter</th><th>1200</th><th>1400 (Current)</th></tr><tr><td>Temperature</td><td>98.4°F</td><td>98.6°F</td></tr><tr><td>Heart Rate</td><td>78 bpm</td><td>102 bpm</td></tr><tr><td>Blood Pressure</td><td>138/88 mmHg</td><td>158/94 mmHg</td></tr><tr><td>Respiratory Rate</td><td>16/min</td><td>22/min</td></tr><tr><td>SpO2</td><td>98%</td><td>94% on RA</td></tr></table>",
        "labResults": "<table><tr><th>Test</th><th>Result</th><th>Reference Range</th></tr><tr><td>Troponin I</td><td><strong>0.8 ng/mL</strong></td><td>&lt;0.04 ng/mL</td></tr><tr><td>BNP</td><td>450 pg/mL</td><td>&lt;100 pg/mL</td></tr><tr><td>Blood Glucose</td><td>186 mg/dL</td><td>70-100 mg/dL</td></tr><tr><td>Potassium</td><td>4.2 mEq/L</td><td>3.5-5.0 mEq/L</td></tr></table>"
    }},
    "options": [
        {{"id": "item3", "text": "Notify the healthcare provider of findings"}},
        {{"id": "item1", "text": "Apply supplemental oxygen via nasal cannula"}},
        {{"id": "item4", "text": "Obtain 12-lead ECG"}},
        {{"id": "item2", "text": "Establish IV access"}},
        {{"id": "item5", "text": "Administer aspirin 325mg as ordered"}}
    ],
    "correctOrder": [
        {{"id": "item1", "text": "Apply supplemental oxygen via nasal cannula"}},
        {{"id": "item2", "text": "Establish IV access"}},
        {{"id": "item3", "text": "Notify the healthcare provider of findings"}},
        {{"id": "item4", "text": "Obtain 12-lead ECG"}},
        {{"id": "item5", "text": "Administer aspirin 325mg as ordered"}}
    ],
    "justification": "<strong>Priority Order Rationale:</strong><br><br><strong>1. Oxygen first</strong> - Airway and breathing take priority (ABCs). SpO2 of 94% on room air indicates hypoxemia requiring immediate intervention.<br><br><strong>2. IV Access</strong> - Essential for medication administration and emergency access. Must be established before calling provider to be ready for orders.<br><br><strong>3. Notify Provider</strong> - With elevated troponin and symptoms, this is a time-sensitive cardiac event requiring immediate medical attention.<br><br><strong>4. 12-lead ECG</strong> - Critical diagnostic tool for suspected MI, done urgently after notifying provider.<br><br><strong>5. Aspirin</strong> - Standard cardiac protocol, but given after other priorities are addressed.",
    "topic": "Cardiac Emergency Priorities",
    "scoringType": "partial",
    "metadata": {{
        "sourceLanguage": "{language}",
        "questionType": "casestudy",
        "category": "nursing",
        "difficulty": "{difficulty}",
        "numItems": {num_items},
        "sourceDocument": "conversational_generation"
    }}
}}

NOTE: The example above shows a CLINICAL scenario with all three tabs. For NON-CLINICAL topics (ethics, delegation, communication, leadership, conflict management), the caseStudy object should contain ONLY "nursesNotes":
{{
    "question": "Placez les actions infirmières suivantes dans l'ordre de priorité lors d'une rencontre interdisciplinaire avec un nouvel usager.",
    "questionType": "casestudy",
    "caseStudy": {{
        "nursesNotes": "<p><strong>Situation :</strong> Une infirmière de l'unité de médecine reçoit un nouvel usager transféré d'un autre établissement. L'équipe interdisciplinaire (médecin, travailleur social, physiothérapeute) se réunit pour planifier le plan d'intervention. L'infirmière doit coordonner la rencontre et s'assurer que toutes les préoccupations sont adressées.</p><p><strong>Contexte :</strong> L'usager est accompagné de sa famille qui exprime des inquiétudes. Le médecin est pressé et souhaite procéder rapidement.</p>"
    }},
    "options": [
        {{"id": "item2", "text": "Demander à chaque intervenant de se présenter et de préciser son champ d'expertise"}},
        {{"id": "item4", "text": "Valider si l'usager ou sa famille a des questions ou des préoccupations"}},
        {{"id": "item1", "text": "Présenter brièvement son rôle et expliquer le but de la rencontre à l'usager et sa famille"}},
        {{"id": "item3", "text": "Partager les informations pertinentes concernant la situation clinique de l'usager selon sa profession"}}
    ],
    "correctOrder": [
        {{"id": "item1", "text": "Présenter brièvement son rôle et expliquer le but de la rencontre à l'usager et sa famille"}},
        {{"id": "item2", "text": "Demander à chaque intervenant de se présenter et de préciser son champ d'expertise"}},
        {{"id": "item3", "text": "Partager les informations pertinentes concernant la situation clinique de l'usager selon sa profession"}},
        {{"id": "item4", "text": "Valider si l'usager ou sa famille a des questions ou des préoccupations"}}
    ],
    "justification": "<strong>Ordre de priorité :</strong><br><br><strong>1. Se présenter et expliquer le but</strong> — Établir un climat de confiance dès le départ.<br><strong>2. Présentation des intervenants</strong> — L'usager doit savoir qui participe.<br><strong>3. Partage d'informations</strong> — Chaque professionnel apporte son expertise.<br><strong>4. Valider les questions</strong> — S'assurer que l'usager et la famille se sentent entendus.",
    "topic": "Rencontre interdisciplinaire",
    "scoringType": "partial",
    "metadata": {{
        "sourceLanguage": "{language}",
        "questionType": "casestudy",
        "category": "nursing",
        "difficulty": "{difficulty}",
        "numItems": {num_items},
        "sourceDocument": "conversational_generation"
    }}
}}

📌 Critical Rules:
1. The "questionType" field MUST be "casestudy"
2. The "options" array must have items in SCRAMBLED order
3. The "correctOrder" array must have items in the CORRECT sequence
4. Both arrays must have matching ids and texts
5. Include {num_items} items to order
6. Write everything in {language}
7. Use realistic clinical values and scenarios
8. Follow evidence-based nursing prioritization (ABCs, Maslow's, safety first)
9. Only include vitalSigns and labResults when they contain clinically meaningful data relevant to the scenario. nursesNotes is ALWAYS required. Do NOT fabricate clinical data for non-clinical topics.
"""


# ============================================
# CASE STUDY QUESTION GENERATOR
# ============================================

async def generate_casestudy_question(
    topic: str,
    difficulty: str,
    question_num: int,
    language: str,
    content_context: str = "",
    questions_to_avoid: list = None
) -> dict:
    """
    Generate a single Case Study (NGN-style ordering) question using LLM.

    Args:
        topic: Subject area for the question (e.g., "Cardiac Care", "Diabetic Emergency")
        difficulty: Question difficulty ("easy", "medium", "hard")
        question_num: Question number in the quiz sequence
        language: Language for the question ("english" or "french")
        content_context: Optional document content to base questions on
        questions_to_avoid: List of previous questions to avoid duplication

    Returns:
        dict: Complete case study question object with all required fields

    Example:
        question = await generate_casestudy_question(
            topic="Cardiac Emergency",
            difficulty="medium",
            question_num=1,
            language="english"
        )

        # Returns:
        # {
        #     "question": "Place the following nursing actions in order of priority.",
        #     "questionType": "casestudy",
        #     "caseStudy": { nursesNotes, vitalSigns, labResults },
        #     "options": [...],  # Scrambled order
        #     "correctOrder": [...],  # Correct sequence
        #     "justification": "...",
        #     "topic": "Cardiac Emergency Priorities"
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

    # Determine number of items based on difficulty
    if difficulty == "easy":
        num_items = random.choice([4, 4, 5])  # Weighted toward 4
    elif difficulty == "hard":
        num_items = random.choice([5, 6, 6])  # Weighted toward 6
    else:  # medium
        num_items = random.choice([4, 5, 5])  # Balanced

    # Default content context if not provided
    if not content_context:
        content_context = f"""You are generating Case Study questions about: {topic}

        Create clinically relevant scenarios that test the student's ability to:
        - Analyze clinical data from multiple sources
        - Prioritize nursing actions based on patient assessment
        - Apply critical thinking to determine correct sequencing
        - Use evidence-based frameworks (ABCs, Maslow's hierarchy)
        """

    print(f"\n{'='*60}")
    print(f"🏥 Generating Case Study Question {question_num}")
    print(f"📚 Topic: {topic}")
    print(f"⚡ Difficulty: {difficulty}")
    print(f"📋 Items to order: {num_items}")
    print(f"🌐 Language: {language}")
    print(f"{'='*60}\n")

    # Create prompt
    prompt = PromptTemplate(
        input_variables=[
            "content", "topic", "difficulty", "question_num",
            "language", "questions_to_avoid", "num_items"
        ],
        template=CASESTUDY_PROMPT_TEMPLATE
    )

    # Use GPT-4o for high-quality clinical scenarios
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({
            "content": content_context,
            "topic": topic,
            "difficulty": difficulty,
            "question_num": question_num,
            "language": language,
            "questions_to_avoid": avoid_text,
            "num_items": num_items
        })

        # Clean and parse JSON response
        cleaned = result.strip().strip("```json").strip("```").strip()
        parsed_question = json.loads(cleaned)

        # Validate required fields
        if 'caseStudy' not in parsed_question:
            raise ValueError("Missing caseStudy field")

        # Clean up empty optional tabs from caseStudy
        case_study_obj = parsed_question.get('caseStudy', {})
        for optional_key in ['vitalSigns', 'labResults']:
            val = case_study_obj.get(optional_key)
            if val is not None and (not val or (isinstance(val, str) and not val.strip())):
                del case_study_obj[optional_key]

        if not isinstance(parsed_question.get('options'), list):
            raise ValueError("options must be a list")

        if not isinstance(parsed_question.get('correctOrder'), list):
            raise ValueError("correctOrder must be a list")

        # Ensure questionType is set
        parsed_question['questionType'] = 'casestudy'

        # Ensure scoringType is set
        if 'scoringType' not in parsed_question:
            parsed_question['scoringType'] = 'partial'

        # Validate topic exists
        if 'topic' not in parsed_question or not parsed_question['topic']:
            parsed_question['topic'] = topic

        # Ensure all items have unique IDs
        for i, item in enumerate(parsed_question.get('options', [])):
            if 'id' not in item:
                item['id'] = f"item{i+1}"

        # Log success
        num_options = len(parsed_question.get('options', []))
        num_correct = len(parsed_question.get('correctOrder', []))
        print(f"✅ Case Study Question {question_num} generated successfully")
        print(f"   Options: {num_options}, Correct order items: {num_correct}")
        print(f"   Topic: {parsed_question.get('topic')}")

        return parsed_question

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse Case Study question {question_num}: {e}")
        if 'result' in locals():
            print(f"Raw output: {result[:500]}...")
        return None

    except Exception as e:
        print(f"❌ Error generating Case Study question {question_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# QUESTION VALIDATION
# ============================================

def validate_casestudy_question(question: dict) -> tuple:
    """
    Validate a Case Study question has all required fields and proper format.

    Args:
        question: The question dictionary to validate

    Returns:
        tuple: (is_valid: bool, errors: list)

    Example:
        is_valid, errors = validate_casestudy_question(question)
        if not is_valid:
            print(f"Validation errors: {errors}")
    """
    errors = []

    # Check required fields
    required_fields = ['question', 'questionType', 'caseStudy', 'options', 'correctOrder', 'justification']
    for field in required_fields:
        if field not in question:
            errors.append(f"Missing required field: {field}")

    # Check questionType
    if question.get('questionType') != 'casestudy':
        errors.append(f"questionType must be 'casestudy', got: {question.get('questionType')}")

    # Check caseStudy tabs — nursesNotes is always required, others are optional
    case_study = question.get('caseStudy', {})
    if 'nursesNotes' not in case_study or not case_study['nursesNotes']:
        errors.append("caseStudy missing or empty required tab: nursesNotes")

    # Optional tabs: if present, they must not be empty
    for optional_tab in ['vitalSigns', 'labResults']:
        if optional_tab in case_study and not case_study[optional_tab]:
            errors.append(f"caseStudy tab '{optional_tab}' is present but empty — either populate it or remove the key")

    # Check options is a list with 4-6 items
    options = question.get('options', [])
    if not isinstance(options, list):
        errors.append("options must be a list")
    elif len(options) < 4 or len(options) > 6:
        errors.append(f"options should have 4-6 items, got: {len(options)}")

    # Check correctOrder is a list matching options length
    correct_order = question.get('correctOrder', [])
    if not isinstance(correct_order, list):
        errors.append("correctOrder must be a list")
    elif len(correct_order) != len(options):
        errors.append(f"correctOrder length ({len(correct_order)}) must match options length ({len(options)})")

    # Check that all option IDs exist in correctOrder
    if isinstance(options, list) and isinstance(correct_order, list):
        option_ids = {item.get('id') for item in options}
        correct_ids = {item.get('id') for item in correct_order}

        if option_ids != correct_ids:
            errors.append("Option IDs must match correctOrder IDs")

    # Check items have required fields (id, text)
    for i, item in enumerate(options):
        if not isinstance(item, dict):
            errors.append(f"Option {i} must be an object")
        elif 'id' not in item or 'text' not in item:
            errors.append(f"Option {i} missing 'id' or 'text' field")

    is_valid = len(errors) == 0
    return is_valid, errors
