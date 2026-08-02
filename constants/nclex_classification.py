"""
NCLEX Classification Constants & Topic Mapping
==============================================

This module provides the mapping between nursing topics and NCLEX classification.
Used by the Question Bank to categorize questions without needing LLM calls.

NCLEX-RN Test Plan Categories:
1. Safe and Effective Care Environment (SECE)
   - Management of Care
   - Safety and Infection Control
2. Health Promotion and Maintenance (HPM)
3. Psychosocial Integrity (PSYCH)
4. Physiological Integrity (PHYS)
   - Basic Care and Comfort
   - Pharmacological Therapies
   - Reduction of Risk Potential
   - Physiological Adaptation

Usage:
    from constants.nclex_classification import get_classification, build_query_key

    # Get NCLEX category for a topic
    cat, subcat, cognitive = get_classification("cardiac medications")
    # Returns: ("PHYS", "PHAR", "Analysis")

    # Build composite key for Firebase query
    key = build_query_key("en", "medium", "cardiac medications")
    # Returns: "en_medium_cardiac medications"
"""

from typing import Tuple, Dict


# ============================================
# NCLEX CATEGORY CODES (Short for storage)
# ============================================

NCLEX_CATEGORIES: Dict[str, str] = {
    "SECE": "Safe and Effective Care Environment",
    "HPM": "Health Promotion and Maintenance",
    "PSYCH": "Psychosocial Integrity",
    "PHYS": "Physiological Integrity"
}

NCLEX_SUBCATEGORIES: Dict[str, str] = {
    # Safe and Effective Care Environment
    "MC": "Management of Care",
    "SIC": "Safety and Infection Control",

    # Health Promotion (uses same code as category)
    "HPM": "Health Promotion and Maintenance",

    # Psychosocial (uses same code as category)
    "PSYCH": "Psychosocial Integrity",

    # Physiological Integrity
    "BC": "Basic Care and Comfort",
    "PHAR": "Pharmacological Therapies",
    "RR": "Reduction of Risk Potential",
    "PA": "Physiological Adaptation"
}

COGNITIVE_LEVELS = ["Knowledge", "Comprehension", "Application", "Analysis"]


# ============================================
# TOPIC TO NCLEX MAPPING
# ============================================
# Format: "topic": (category_code, subcategory_code, cognitive_level)
# Topics should be lowercase for consistent matching

TOPIC_MAP: Dict[str, Tuple[str, str, str]] = {
    # -----------------------------------------
    # PHARMACOLOGY (PHYS → PHAR)
    # -----------------------------------------
    "pharmacology": ("PHYS", "PHAR", "Application"),
    "medication administration": ("PHYS", "PHAR", "Application"),
    "drug calculations": ("PHYS", "PHAR", "Analysis"),
    "dosage calculations": ("PHYS", "PHAR", "Analysis"),
    "cardiac medications": ("PHYS", "PHAR", "Analysis"),
    "cardiovascular drugs": ("PHYS", "PHAR", "Analysis"),
    "antihypertensives": ("PHYS", "PHAR", "Application"),
    "antiarrhythmics": ("PHYS", "PHAR", "Analysis"),
    "anticoagulants": ("PHYS", "PHAR", "Analysis"),
    "antibiotics": ("PHYS", "PHAR", "Application"),
    "antimicrobials": ("PHYS", "PHAR", "Application"),
    "pain management": ("PHYS", "PHAR", "Analysis"),
    "analgesics": ("PHYS", "PHAR", "Application"),
    "opioids": ("PHYS", "PHAR", "Analysis"),
    "insulin": ("PHYS", "PHAR", "Application"),
    "diabetes medications": ("PHYS", "PHAR", "Application"),
    "diuretics": ("PHYS", "PHAR", "Application"),
    "psychotropic medications": ("PHYS", "PHAR", "Analysis"),
    "antidepressants": ("PHYS", "PHAR", "Application"),
    "antipsychotics": ("PHYS", "PHAR", "Analysis"),
    "sedatives": ("PHYS", "PHAR", "Application"),
    "anesthesia": ("PHYS", "PHAR", "Analysis"),
    "chemotherapy": ("PHYS", "PHAR", "Analysis"),
    "immunosuppressants": ("PHYS", "PHAR", "Analysis"),
    "steroids": ("PHYS", "PHAR", "Application"),
    "bronchodilators": ("PHYS", "PHAR", "Application"),
    "iv therapy": ("PHYS", "PHAR", "Application"),
    "blood products": ("PHYS", "PHAR", "Analysis"),

    # -----------------------------------------
    # SAFETY & INFECTION CONTROL (SECE → SIC)
    # -----------------------------------------
    "infection control": ("SECE", "SIC", "Application"),
    "standard precautions": ("SECE", "SIC", "Knowledge"),
    "isolation precautions": ("SECE", "SIC", "Application"),
    "transmission-based precautions": ("SECE", "SIC", "Application"),
    "hand hygiene": ("SECE", "SIC", "Knowledge"),
    "sterile technique": ("SECE", "SIC", "Application"),
    "aseptic technique": ("SECE", "SIC", "Application"),
    "fall prevention": ("SECE", "SIC", "Application"),
    "patient safety": ("SECE", "SIC", "Analysis"),
    "restraints": ("SECE", "SIC", "Application"),
    "fire safety": ("SECE", "SIC", "Knowledge"),
    "emergency procedures": ("SECE", "SIC", "Application"),
    "disaster preparedness": ("SECE", "SIC", "Analysis"),
    "hazardous materials": ("SECE", "SIC", "Application"),
    "radiation safety": ("SECE", "SIC", "Application"),
    "sharps safety": ("SECE", "SIC", "Knowledge"),
    "medication errors": ("SECE", "SIC", "Analysis"),
    "surgical safety": ("SECE", "SIC", "Application"),

    # -----------------------------------------
    # MANAGEMENT OF CARE (SECE → MC)
    # -----------------------------------------
    "delegation": ("SECE", "MC", "Analysis"),
    "prioritization": ("SECE", "MC", "Analysis"),
    "triage": ("SECE", "MC", "Analysis"),
    "ethical practice": ("SECE", "MC", "Analysis"),
    "legal issues": ("SECE", "MC", "Knowledge"),
    "nursing ethics": ("SECE", "MC", "Analysis"),
    "informed consent": ("SECE", "MC", "Application"),
    "patient rights": ("SECE", "MC", "Knowledge"),
    "advocacy": ("SECE", "MC", "Application"),
    "advance directives": ("SECE", "MC", "Application"),
    "confidentiality": ("SECE", "MC", "Application"),
    "hipaa": ("SECE", "MC", "Knowledge"),
    "documentation": ("SECE", "MC", "Application"),
    "care coordination": ("SECE", "MC", "Analysis"),
    "discharge planning": ("SECE", "MC", "Application"),
    "case management": ("SECE", "MC", "Analysis"),
    "quality improvement": ("SECE", "MC", "Analysis"),
    "supervision": ("SECE", "MC", "Analysis"),
    "scope of practice": ("SECE", "MC", "Knowledge"),
    "leadership": ("SECE", "MC", "Analysis"),
    "interprofessional collaboration": ("SECE", "MC", "Application"),

    # -----------------------------------------
    # HEALTH PROMOTION & MAINTENANCE (HPM)
    # -----------------------------------------
    "health promotion": ("HPM", "HPM", "Application"),
    "patient education": ("HPM", "HPM", "Application"),
    "health teaching": ("HPM", "HPM", "Application"),
    "health screening": ("HPM", "HPM", "Application"),
    "disease prevention": ("HPM", "HPM", "Application"),
    "immunizations": ("HPM", "HPM", "Application"),
    "vaccines": ("HPM", "HPM", "Application"),
    "growth and development": ("HPM", "HPM", "Knowledge"),
    "developmental stages": ("HPM", "HPM", "Knowledge"),
    "pediatric development": ("HPM", "HPM", "Knowledge"),
    "aging": ("HPM", "HPM", "Knowledge"),
    "lifestyle modifications": ("HPM", "HPM", "Application"),
    "prenatal care": ("HPM", "HPM", "Application"),
    "postpartum care": ("HPM", "HPM", "Application"),
    "newborn care": ("HPM", "HPM", "Application"),
    "breastfeeding": ("HPM", "HPM", "Application"),
    "family planning": ("HPM", "HPM", "Application"),
    "self-care": ("HPM", "HPM", "Application"),
    "wellness": ("HPM", "HPM", "Application"),

    # -----------------------------------------
    # PSYCHOSOCIAL INTEGRITY (PSYCH)
    # -----------------------------------------
    "mental health": ("PSYCH", "PSYCH", "Analysis"),
    "psychiatric nursing": ("PSYCH", "PSYCH", "Analysis"),
    "therapeutic communication": ("PSYCH", "PSYCH", "Application"),
    "crisis intervention": ("PSYCH", "PSYCH", "Analysis"),
    "grief and loss": ("PSYCH", "PSYCH", "Application"),
    "death and dying": ("PSYCH", "PSYCH", "Application"),
    "end of life care": ("PSYCH", "PSYCH", "Application"),
    "coping mechanisms": ("PSYCH", "PSYCH", "Application"),
    "stress management": ("PSYCH", "PSYCH", "Application"),
    "anxiety disorders": ("PSYCH", "PSYCH", "Analysis"),
    "depression": ("PSYCH", "PSYCH", "Analysis"),
    "bipolar disorder": ("PSYCH", "PSYCH", "Analysis"),
    "schizophrenia": ("PSYCH", "PSYCH", "Analysis"),
    "personality disorders": ("PSYCH", "PSYCH", "Analysis"),
    "substance abuse": ("PSYCH", "PSYCH", "Analysis"),
    "addiction": ("PSYCH", "PSYCH", "Analysis"),
    "eating disorders": ("PSYCH", "PSYCH", "Analysis"),
    "abuse and neglect": ("PSYCH", "PSYCH", "Analysis"),
    "domestic violence": ("PSYCH", "PSYCH", "Analysis"),
    "suicide prevention": ("PSYCH", "PSYCH", "Analysis"),
    "cultural considerations": ("PSYCH", "PSYCH", "Application"),
    "spiritual care": ("PSYCH", "PSYCH", "Application"),
    "family dynamics": ("PSYCH", "PSYCH", "Application"),
    "support systems": ("PSYCH", "PSYCH", "Application"),

    # -----------------------------------------
    # BASIC CARE & COMFORT (PHYS → BC)
    # -----------------------------------------
    "nutrition": ("PHYS", "BC", "Application"),
    "diet therapy": ("PHYS", "BC", "Application"),
    "enteral nutrition": ("PHYS", "BC", "Application"),
    "parenteral nutrition": ("PHYS", "BC", "Application"),
    "mobility": ("PHYS", "BC", "Application"),
    "positioning": ("PHYS", "BC", "Application"),
    "body mechanics": ("PHYS", "BC", "Application"),
    "transfers": ("PHYS", "BC", "Application"),
    "range of motion": ("PHYS", "BC", "Application"),
    "comfort measures": ("PHYS", "BC", "Application"),
    "sleep": ("PHYS", "BC", "Application"),
    "rest": ("PHYS", "BC", "Application"),
    "hygiene": ("PHYS", "BC", "Application"),
    "personal care": ("PHYS", "BC", "Application"),
    "elimination": ("PHYS", "BC", "Application"),
    "urinary care": ("PHYS", "BC", "Application"),
    "bowel care": ("PHYS", "BC", "Application"),
    "skin integrity": ("PHYS", "BC", "Application"),
    "wound care": ("PHYS", "BC", "Application"),
    "pressure ulcers": ("PHYS", "BC", "Application"),
    "palliative care": ("PHYS", "BC", "Application"),
    "hospice care": ("PHYS", "BC", "Application"),
    "non-pharmacological pain management": ("PHYS", "BC", "Application"),
    "alternative therapies": ("PHYS", "BC", "Application"),

    # -----------------------------------------
    # REDUCTION OF RISK POTENTIAL (PHYS → RR)
    # -----------------------------------------
    "diagnostic procedures": ("PHYS", "RR", "Application"),
    "lab values": ("PHYS", "RR", "Analysis"),
    "laboratory tests": ("PHYS", "RR", "Analysis"),
    "blood tests": ("PHYS", "RR", "Analysis"),
    "vital signs": ("PHYS", "RR", "Application"),
    "assessment": ("PHYS", "RR", "Application"),
    "physical assessment": ("PHYS", "RR", "Application"),
    "health assessment": ("PHYS", "RR", "Application"),
    "complications": ("PHYS", "RR", "Analysis"),
    "post-operative care": ("PHYS", "RR", "Application"),
    "pre-operative care": ("PHYS", "RR", "Application"),
    "surgical complications": ("PHYS", "RR", "Analysis"),
    "potential complications": ("PHYS", "RR", "Analysis"),
    "monitoring": ("PHYS", "RR", "Application"),
    "therapeutic procedures": ("PHYS", "RR", "Application"),
    "invasive procedures": ("PHYS", "RR", "Application"),
    "tubes and drains": ("PHYS", "RR", "Application"),
    "specimen collection": ("PHYS", "RR", "Application"),

    # -----------------------------------------
    # PHYSIOLOGICAL ADAPTATION (PHYS → PA)
    # -----------------------------------------
    "cardiac disorders": ("PHYS", "PA", "Analysis"),
    "heart failure": ("PHYS", "PA", "Analysis"),
    "myocardial infarction": ("PHYS", "PA", "Analysis"),
    "arrhythmias": ("PHYS", "PA", "Analysis"),
    "hypertension": ("PHYS", "PA", "Analysis"),
    "shock": ("PHYS", "PA", "Analysis"),
    "respiratory disorders": ("PHYS", "PA", "Analysis"),
    "pneumonia": ("PHYS", "PA", "Analysis"),
    "copd": ("PHYS", "PA", "Analysis"),
    "asthma": ("PHYS", "PA", "Analysis"),
    "respiratory failure": ("PHYS", "PA", "Analysis"),
    "mechanical ventilation": ("PHYS", "PA", "Analysis"),
    "oxygen therapy": ("PHYS", "PA", "Application"),
    "neurological disorders": ("PHYS", "PA", "Analysis"),
    "stroke": ("PHYS", "PA", "Analysis"),
    "seizures": ("PHYS", "PA", "Analysis"),
    "head injury": ("PHYS", "PA", "Analysis"),
    "increased intracranial pressure": ("PHYS", "PA", "Analysis"),
    "renal disorders": ("PHYS", "PA", "Analysis"),
    "kidney failure": ("PHYS", "PA", "Analysis"),
    "dialysis": ("PHYS", "PA", "Application"),
    "diabetes": ("PHYS", "PA", "Analysis"),
    "diabetic emergencies": ("PHYS", "PA", "Analysis"),
    "thyroid disorders": ("PHYS", "PA", "Analysis"),
    "endocrine disorders": ("PHYS", "PA", "Analysis"),
    "fluid and electrolytes": ("PHYS", "PA", "Analysis"),
    "acid-base balance": ("PHYS", "PA", "Analysis"),
    "gastrointestinal disorders": ("PHYS", "PA", "Analysis"),
    "liver disorders": ("PHYS", "PA", "Analysis"),
    "pancreatitis": ("PHYS", "PA", "Analysis"),
    "musculoskeletal disorders": ("PHYS", "PA", "Analysis"),
    "fractures": ("PHYS", "PA", "Application"),
    "arthritis": ("PHYS", "PA", "Application"),
    "hematological disorders": ("PHYS", "PA", "Analysis"),
    "anemia": ("PHYS", "PA", "Analysis"),
    "cancer": ("PHYS", "PA", "Analysis"),
    "oncology": ("PHYS", "PA", "Analysis"),
    "burns": ("PHYS", "PA", "Analysis"),
    "trauma": ("PHYS", "PA", "Analysis"),
    "emergency care": ("PHYS", "PA", "Analysis"),
    "critical care": ("PHYS", "PA", "Analysis"),
    "icu": ("PHYS", "PA", "Analysis"),
    "sepsis": ("PHYS", "PA", "Analysis"),
    "infectious diseases": ("PHYS", "PA", "Analysis"),
    "immune disorders": ("PHYS", "PA", "Analysis"),
    "hiv/aids": ("PHYS", "PA", "Analysis"),
    "autoimmune disorders": ("PHYS", "PA", "Analysis"),

    # -----------------------------------------
    # MATERNAL-CHILD NURSING (PHYS → PA / HPM)
    # -----------------------------------------
    "obstetrics": ("PHYS", "PA", "Analysis"),
    "labor and delivery": ("PHYS", "PA", "Analysis"),
    "complications of pregnancy": ("PHYS", "PA", "Analysis"),
    "high-risk pregnancy": ("PHYS", "PA", "Analysis"),
    "fetal monitoring": ("PHYS", "RR", "Application"),
    "pediatric nursing": ("PHYS", "PA", "Analysis"),
    "pediatric assessment": ("PHYS", "RR", "Application"),
    "childhood illnesses": ("PHYS", "PA", "Analysis"),
    "neonatal care": ("PHYS", "PA", "Analysis"),

    # -----------------------------------------
    # ANATOMY & PHYSIOLOGY (PHYS → PA)
    # -----------------------------------------
    "anatomy": ("PHYS", "PA", "Knowledge"),
    "physiology": ("PHYS", "PA", "Knowledge"),
    "cardiovascular system": ("PHYS", "PA", "Knowledge"),
    "respiratory system": ("PHYS", "PA", "Knowledge"),
    "nervous system": ("PHYS", "PA", "Knowledge"),
    "digestive system": ("PHYS", "PA", "Knowledge"),
    "urinary system": ("PHYS", "PA", "Knowledge"),
    "endocrine system": ("PHYS", "PA", "Knowledge"),
    "immune system": ("PHYS", "PA", "Knowledge"),
    "integumentary system": ("PHYS", "PA", "Knowledge"),
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_classification(topic: str) -> Tuple[str, str, str]:
    """
    Get NCLEX classification for a given topic.

    Args:
        topic: The nursing topic (case-insensitive)

    Returns:
        Tuple of (category_code, subcategory_code, cognitive_level)

    Example:
        >>> get_classification("Cardiac Medications")
        ("PHYS", "PHAR", "Analysis")

        >>> get_classification("unknown topic")
        ("PHYS", "PA", "Application")  # Default fallback
    """
    # Normalize the topic for matching
    topic_lower = topic.lower().strip()

    # Try exact match first (fastest)
    if topic_lower in TOPIC_MAP:
        return TOPIC_MAP[topic_lower]

    # Try partial match - check if any key is contained in the topic
    for key, classification in TOPIC_MAP.items():
        if key in topic_lower:
            return classification

    # Try reverse partial match - check if topic is contained in any key
    for key, classification in TOPIC_MAP.items():
        if topic_lower in key:
            return classification

    # Default fallback: Physiological Adaptation (most common)
    return ("PHYS", "PA", "Application")


def get_full_category_name(code: str) -> str:
    """
    Convert category code to full name.

    Args:
        code: The category code (e.g., "PHYS")

    Returns:
        Full category name (e.g., "Physiological Integrity")
    """
    return NCLEX_CATEGORIES.get(code, code)


def get_full_subcategory_name(code: str) -> str:
    """
    Convert subcategory code to full name.

    Args:
        code: The subcategory code (e.g., "PHAR")

    Returns:
        Full subcategory name (e.g., "Pharmacological Therapies")
    """
    return NCLEX_SUBCATEGORIES.get(code, code)


def build_query_key(language: str, difficulty: str, topic: str) -> str:
    """
    Build a composite key for efficient Firebase queries.

    This combines language, difficulty, and topic into a single string
    that can be used for fast equality queries in Firestore.

    Args:
        language: Language code (e.g., "en", "fr")
        difficulty: Difficulty level (e.g., "easy", "medium", "hard")
        topic: The nursing topic

    Returns:
        Composite query key (e.g., "en_medium_cardiac medications")

    Example:
        >>> build_query_key("en", "medium", "Cardiac Medications")
        "en_medium_cardiac medications"
    """
    topic_lower = topic.lower().strip()
    return f"{language}_{difficulty}_{topic_lower}"


def build_category_key(language: str, difficulty: str, topic: str) -> str:
    """
    Build a composite key based on NCLEX category for fallback queries.

    When there aren't enough questions for a specific topic, we can
    fall back to questions in the same NCLEX category.

    Args:
        language: Language code (e.g., "en", "fr")
        difficulty: Difficulty level (e.g., "easy", "medium", "hard")
        topic: The nursing topic (used to determine category)

    Returns:
        Composite category key (e.g., "en_medium_PHYS_PHAR")
    """
    cat, subcat, _ = get_classification(topic)
    return f"{language}_{difficulty}_{cat}_{subcat}"


# ============================================
# VALIDATION HELPERS
# ============================================

VALID_DIFFICULTIES = ["easy", "medium", "hard"]
VALID_LANGUAGES = ["en", "fr", "es", "de", "pt", "it", "zh", "ja", "ko", "ar"]


def validate_difficulty(difficulty: str) -> str:
    """
    Validate and normalize difficulty level.

    Args:
        difficulty: Input difficulty string

    Returns:
        Normalized difficulty (defaults to "medium" if invalid)
    """
    normalized = difficulty.lower().strip()
    if normalized in VALID_DIFFICULTIES:
        return normalized
    return "medium"


def validate_language(language: str) -> str:
    """
    Validate and normalize language code.

    Args:
        language: Input language code

    Returns:
        Normalized language code (defaults to "en" if invalid)
    """
    normalized = language.lower().strip()[:2]  # Take first 2 chars
    if normalized in VALID_LANGUAGES:
        return normalized
    return "en"
