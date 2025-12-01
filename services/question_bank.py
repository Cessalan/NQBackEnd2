"""
Question Bank Service
====================

This service manages the NCLEX Question Bank stored in Firebase Firestore.
It provides methods to:
- Retrieve questions from the bank (fast, no LLM needed)
- Save newly generated questions to enrich the bank
- Update question statistics after students answer

The "flywheel" effect:
1. First request for a topic → generates questions via LLM → saves to bank
2. Future requests for same topic → instant retrieval from bank
3. Over time, the bank grows and fewer LLM calls are needed

Firebase Collection Structure:
-----------------------------
questionBank/{questionId}/
    ├── q          : Question text
    ├── opts       : Array of options ["A) ...", "B) ...", "C) ...", "D) ..."]
    ├── ans        : Correct answer string
    ├── just       : Justification/rationale HTML
    ├── lang       : Language code ("en", "fr", etc.)
    ├── diff       : Difficulty ("easy", "medium", "hard")
    ├── topic      : Primary topic (lowercase)
    ├── cat        : NCLEX category code ("PHYS", "SECE", etc.)
    ├── subcat     : NCLEX subcategory code ("PHAR", "SIC", etc.)
    ├── cog        : Cognitive level ("Knowledge", "Application", "Analysis")
    ├── ldt        : Composite key for topic queries (lang_diff_topic)
    ├── ldc        : Composite key for category queries (lang_diff_cat_subcat)
    ├── src        : Source type ("generated", "imported")
    ├── chatId     : Source chat ID (if generated from a session)
    ├── status     : Status ("active", "retired")
    ├── createdAt  : Timestamp of creation
    ├── served     : Number of times this question was served
    └── correct    : Number of times answered correctly

Usage Example:
-------------
    from services.question_bank import question_bank

    # Get questions from the bank
    questions, count = await question_bank.get_questions(
        topic="cardiac medications",
        language="en",
        difficulty="medium",
        count=4
    )

    # Save a generated question to the bank
    question_id = await question_bank.save_question(
        question_data=generated_question,
        topic="cardiac medications",
        language="en",
        difficulty="medium",
        chat_id="abc123"
    )

    # Update stats after a student answers
    await question_bank.update_stats(question_id, was_correct=True)
"""

import random
import logging
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime

from firebase_admin import firestore

from constants.nclex_classification import (
    get_classification,
    build_query_key,
    build_category_key,
    get_full_category_name,
    get_full_subcategory_name,
    validate_difficulty,
    validate_language,
)

# Set up logging
logger = logging.getLogger(__name__)


class QuestionBankService:
    """
    Service for managing the NCLEX Question Bank in Firebase Firestore.

    This class handles all CRUD operations for the question bank,
    optimized for Firebase's query limitations using composite keys.
    """

    # Firestore collection name
    COLLECTION_NAME = "questionBank"

    def __init__(self):
        """
        Initialize the Question Bank Service.

        Note: Firebase must be initialized before creating this instance.
        The firebase_admin.initialize_app() is called in main.py on startup.
        """
        self._db = None  # Lazy initialization

    @property
    def db(self):
        """
        Lazy-load the Firestore client.

        This prevents errors if the service is imported before
        Firebase is initialized.
        """
        if self._db is None:
            self._db = firestore.client()
        return self._db

    @property
    def collection(self):
        """Get the questionBank collection reference."""
        return self.db.collection(self.COLLECTION_NAME)

    # ==========================================
    # RETRIEVE QUESTIONS
    # ==========================================

    async def get_questions(
        self,
        topic: str,
        language: str = "en",
        difficulty: str = "medium",
        count: int = 4,
        exclude_ids: Optional[List[str]] = None,
        question_type: Optional[str] = None
    ) -> Tuple[List[Dict], int]:
        """
        Retrieve questions from the bank for a given topic.

        This method uses a two-tier strategy:
        1. First, try to find exact topic matches
        2. If not enough, fall back to same NCLEX category

        Args:
            topic: The nursing topic to get questions for
            language: Language code (e.g., "en", "fr")
            difficulty: Difficulty level ("easy", "medium", "hard")
            count: Number of questions to retrieve
            exclude_ids: List of question IDs to exclude (avoid repeats)
            question_type: Optional question type filter ("mcq", "sata", "casestudy")
                          If not specified, returns MCQ questions only for backwards compatibility

        Returns:
            Tuple of (list of questions, number retrieved from bank)

        Example:
            >>> questions, from_bank = await question_bank.get_questions(
            ...     topic="cardiac medications",
            ...     language="en",
            ...     difficulty="medium",
            ...     count=4,
            ...     question_type="mcq"
            ... )
            >>> print(f"Got {from_bank} questions from bank")
        """
        # Validate inputs
        language = validate_language(language)
        difficulty = validate_difficulty(difficulty)
        exclude_set = set(exclude_ids) if exclude_ids else set()

        # Normalize question type (default to mcq for backwards compatibility)
        if question_type is None:
            question_type = "mcq"
        question_type = question_type.lower()

        # For casestudy/ordering/bowtie types, skip the bank entirely for now
        # since most existing questions are MCQ. Force generation of new questions.
        if question_type in ["casestudy", "ordering", "bowtie"]:
            logger.info(
                f"Question bank: Skipping bank for question_type='{question_type}' - "
                f"forcing LLM generation for topic='{topic}'"
            )
            return [], 0

        # Build the composite query key for topic-based lookup
        query_key = build_query_key(language, difficulty, topic)

        # Strategy 1: Try exact topic match
        questions = await self._query_by_key("ldt", query_key, count * 2, exclude_set, question_type)

        # Strategy 2: If not enough, try NCLEX category fallback
        if len(questions) < count:
            # Get IDs we already have to avoid duplicates
            existing_ids = {q["id"] for q in questions}
            exclude_set.update(existing_ids)

            # Build category-based query key
            category_key = build_category_key(language, difficulty, topic)

            # Get more questions from same NCLEX category
            additional = await self._query_by_key(
                "ldc", category_key, count * 2, exclude_set, question_type
            )
            questions.extend(additional)

        # Shuffle to avoid predictable ordering
        random.shuffle(questions)

        # Limit to requested count
        final_questions = questions[:count]
        from_bank_count = len(final_questions)

        logger.info(
            f"Question bank: Retrieved {from_bank_count}/{count} for topic='{topic}', "
            f"lang='{language}', diff='{difficulty}', qtype='{question_type}'"
        )

        return final_questions, from_bank_count

    async def _query_by_key(
        self,
        field: str,
        value: str,
        limit: int,
        exclude_ids: Set[str],
        question_type: str = "mcq"
    ) -> List[Dict]:
        """
        Query questions using a composite key field.

        This is a helper method that performs the actual Firestore query.
        Uses a single WHERE clause for efficiency.

        Args:
            field: The composite key field to query ("ldt" or "ldc")
            value: The composite key value to match
            limit: Maximum number of documents to fetch
            exclude_ids: Set of question IDs to skip
            question_type: Question type to filter by ("mcq", "sata", "casestudy")

        Returns:
            List of question dictionaries
        """
        try:
            # Build and execute the query
            # Note: We query with status="active" to only get usable questions
            # Using filter= keyword to avoid deprecation warning
            from google.cloud.firestore_v1.base_query import FieldFilter
            query = (
                self.collection
                .where(filter=FieldFilter(field, "==", value))
                .where(filter=FieldFilter("status", "==", "active"))
                .limit(limit)
            )

            # Execute query (this is synchronous in firebase-admin)
            docs = query.stream()

            # Convert to question format, skipping excluded IDs and filtering by type
            results = []
            for doc in docs:
                if doc.id not in exclude_ids:
                    data = doc.to_dict()
                    # Filter by question type (default to mcq for old questions without qtype)
                    doc_qtype = data.get("qtype", "mcq")
                    if doc_qtype == question_type:
                        question = self._document_to_question(doc)
                        results.append(question)

            return results

        except Exception as e:
            logger.error(f"Error querying question bank: {e}")
            return []

    def _document_to_question(self, doc) -> Dict:
        """
        Convert a Firestore document to the question format expected by frontend.

        This ensures the question object matches the structure that
        the quiz components expect, handling different question types.

        Args:
            doc: Firestore document snapshot

        Returns:
            Dictionary matching the frontend question format
        """
        data = doc.to_dict()

        # Get question type (default to mcq for backwards compatibility)
        question_type = data.get("qtype", "mcq")

        # Calculate correct answer index for frontend (MCQ only)
        options = data.get("opts", [])
        answer = data.get("ans", "")

        # For MCQ, calculate correct index
        correct_index = 0
        if question_type == "mcq" and isinstance(answer, str):
            try:
                correct_index = options.index(answer)
            except ValueError:
                correct_index = 0

        # Build base question object
        question = {
            # Include ID for tracking and exclusion
            "id": doc.id,

            # Core question data
            "question": data.get("q", ""),
            "options": options,
            "answer": answer,
            "justification": data.get("just", ""),

            # Question type for frontend routing
            "questionType": question_type,

            # Topic for analytics
            "topic": data.get("topic", ""),

            # Metadata matching existing frontend expectations
            "metadata": {
                "sourceLanguage": data.get("lang", "en"),
                "topic": data.get("topic", ""),
                "category": "nursing",
                "difficulty": data.get("diff", "medium"),
                "correctAnswerIndex": correct_index,
                "sourceDocument": "question_bank",
                "keywords": [],
                "nclex_category": get_full_category_name(data.get("cat", "")),
                "nclex_subcategory": get_full_subcategory_name(data.get("subcat", "")),
                "questionType": question_type,
            },

            # Flag to indicate this came from the bank
            "from_bank": True
        }

        # Add SATA-specific fields
        if question_type == "sata":
            question["scoringType"] = data.get("scoringType", "partial")

        # Add case study-specific fields
        if question_type == "casestudy":
            question["caseStudy"] = data.get("caseStudy", {})
            question["correctOrder"] = data.get("correctOrder", [])
            question["scoringType"] = data.get("scoringType", "partial")

        return question

    # ==========================================
    # SAVE QUESTIONS
    # ==========================================

    async def save_question(
        self,
        question_data: Dict,
        topic: str,
        language: str,
        difficulty: str,
        chat_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Save a generated question to the question bank.

        This is called after generating a new question via LLM to
        enrich the bank for future instant retrieval.

        Args:
            question_data: The generated question dictionary containing:
                - question: The question text
                - options: List of answer options
                - answer: The correct answer
                - justification: Explanation/rationale
            topic: The nursing topic
            language: Language code
            difficulty: Difficulty level
            chat_id: Source chat ID (optional, for tracking)

        Returns:
            The document ID of the saved question, or None if duplicate

        Example:
            >>> question_id = await question_bank.save_question(
            ...     question_data={
            ...         "question": "Which medication...",
            ...         "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
            ...         "answer": "B) ...",
            ...         "justification": "B is correct because..."
            ...     },
            ...     topic="cardiac medications",
            ...     language="en",
            ...     difficulty="medium",
            ...     chat_id="abc123"
            ... )
        """
        try:
            print(f"📝 [QuestionBank.save_question] Starting save for topic='{topic}'...")

            # Validate inputs
            language = validate_language(language)
            difficulty = validate_difficulty(difficulty)

            # Ensure topic is never empty - use question's topic or fallback
            topic_to_use = topic.lower().strip() if topic else ""
            if not topic_to_use:
                # Try to get topic from question_data
                topic_to_use = question_data.get("topic", "").lower().strip()
            if not topic_to_use:
                # Last fallback
                topic_to_use = "general nursing"
                print(f"⚠️ [QuestionBank.save_question] No topic provided, using fallback: '{topic_to_use}'")

            topic_lower = topic_to_use

            print(f"📝 [QuestionBank.save_question] Validated: lang={language}, diff={difficulty}, topic={topic_lower}")

            # Get NCLEX classification for this topic
            cat, subcat, cog = get_classification(topic_lower)
            print(f"📝 [QuestionBank.save_question] NCLEX classification: {cat}/{subcat}/{cog}")

            # Build composite keys for efficient queries
            ldt_key = build_query_key(language, difficulty, topic_lower)
            ldc_key = build_category_key(language, difficulty, topic_lower)
            print(f"📝 [QuestionBank.save_question] Composite keys: ldt={ldt_key}, ldc={ldc_key}")

            # Check for duplicates before saving
            question_text = question_data.get("question", "")
            if await self._is_duplicate(question_text, language):
                logger.info(f"Duplicate question detected, skipping save")
                return None

            # Determine question type (mcq, sata, casestudy, etc.)
            question_type = question_data.get("questionType", "mcq")

            # Build the document to save
            doc_data = {
                # Core question content (short keys to save storage)
                "q": question_text,
                "opts": question_data.get("options", []),
                "ans": question_data.get("answer", ""),
                "just": question_data.get("justification", ""),
                "qtype": question_type,  # Question type for filtering

                # Classification (flat fields for efficient queries)
                "lang": language,
                "diff": difficulty,
                "topic": topic_lower,
                "cat": cat,
                "subcat": subcat,
                "cog": cog,

                # Composite keys for fast queries
                "ldt": ldt_key,
                "ldc": ldc_key,

                # Metadata
                "src": "generated",
                "chatId": chat_id,
                "status": "active",
                "createdAt": firestore.SERVER_TIMESTAMP,

                # Statistics (initialized to zero)
                "served": 0,
                "correct": 0
            }

            # For SATA questions, store answer as array
            if question_type == "sata" and isinstance(question_data.get("answer"), list):
                doc_data["ans"] = question_data.get("answer", [])
                doc_data["scoringType"] = question_data.get("scoringType", "partial")

            # For case study questions, store additional fields
            if question_type == "casestudy":
                doc_data["caseStudy"] = question_data.get("caseStudy", {})
                doc_data["correctOrder"] = question_data.get("correctOrder", [])
                doc_data["scoringType"] = question_data.get("scoringType", "partial")

            # Add to Firestore
            print(f"📝 [QuestionBank.save_question] Adding to Firestore collection '{self.COLLECTION_NAME}'...")
            _, doc_ref = self.collection.add(doc_data)
            doc_id = doc_ref.id

            print(f"✅ [QuestionBank.save_question] SUCCESS! Document ID: {doc_id}")
            logger.info(
                f"Question saved to bank: id={doc_id}, topic='{topic_lower}', "
                f"cat={cat}/{subcat}, lang={language}"
            )

            return doc_id

        except Exception as e:
            print(f"❌ [QuestionBank.save_question] ERROR: {e}")
            import traceback
            print(f"❌ [QuestionBank.save_question] Traceback:\n{traceback.format_exc()}")
            logger.error(f"Error saving question to bank: {e}")
            return None

    async def _is_duplicate(self, question_text: str, language: str) -> bool:
        """
        Check if a similar question already exists in the bank.

        Uses a simple prefix-based comparison. This catches exact
        duplicates and questions that start the same way.

        Args:
            question_text: The question text to check
            language: Language code (only compare within same language)

        Returns:
            True if a duplicate exists, False otherwise
        """
        try:
            # Normalize and take first 100 characters for comparison
            normalized = question_text.lower().strip()[:100]

            if not normalized:
                return False

            # Query a sample of questions in the same language
            # We limit to 100 to keep this fast
            # Using filter= keyword to avoid deprecation warning
            from google.cloud.firestore_v1.base_query import FieldFilter
            query = (
                self.collection
                .where(filter=FieldFilter("lang", "==", language))
                .where(filter=FieldFilter("status", "==", "active"))
                .limit(100)
            )

            docs = query.stream()

            # Check each question for similarity
            for doc in docs:
                existing_text = doc.to_dict().get("q", "").lower().strip()[:100]
                if normalized == existing_text:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
            # If we can't check, assume not duplicate to avoid losing questions
            return False

    # ==========================================
    # UPDATE STATISTICS
    # ==========================================

    async def update_stats(self, question_id: str, was_correct: bool) -> bool:
        """
        Update question statistics after a student answers.

        This tracks how often questions are served and how often
        they're answered correctly. Over time, this data can be
        used to calibrate difficulty.

        Args:
            question_id: The document ID of the question
            was_correct: Whether the student answered correctly

        Returns:
            True if update succeeded, False otherwise

        Example:
            >>> await question_bank.update_stats("abc123", was_correct=True)
        """
        try:
            doc_ref = self.collection.document(question_id)

            # Use Firestore Increment to atomically update counters
            updates = {
                "served": firestore.Increment(1)
            }

            if was_correct:
                updates["correct"] = firestore.Increment(1)

            doc_ref.update(updates)

            logger.debug(f"Updated stats for question {question_id}, correct={was_correct}")
            return True

        except Exception as e:
            logger.error(f"Error updating question stats: {e}")
            return False

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    async def get_bank_stats(self) -> Dict:
        """
        Get overall statistics about the question bank.

        Useful for admin dashboards and monitoring.

        Returns:
            Dictionary with bank statistics
        """
        try:
            # Count total questions
            # Using filter= keyword to avoid deprecation warning
            from google.cloud.firestore_v1.base_query import FieldFilter
            active_query = self.collection.where(filter=FieldFilter("status", "==", "active"))
            active_docs = list(active_query.stream())
            total_active = len(active_docs)

            # Count by language
            language_counts = {}
            category_counts = {}

            for doc in active_docs:
                data = doc.to_dict()

                lang = data.get("lang", "unknown")
                language_counts[lang] = language_counts.get(lang, 0) + 1

                cat = data.get("cat", "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1

            return {
                "total_active": total_active,
                "by_language": language_counts,
                "by_category": category_counts,
                "last_checked": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting bank stats: {e}")
            return {"error": str(e)}


# ==========================================
# SINGLETON INSTANCE
# ==========================================
# Create a single instance to be imported throughout the app

question_bank = QuestionBankService()
