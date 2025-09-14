from langdetect import detect

class LanguageDetector:
    """Handles language detection and persistence"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language from text, default to English if detection fails"""
        try:
            lang = detect(text)  # Uses langdetect library
            # Map to our supported languages
            lang_map = {
                'fr': 'french',
                'en': 'english',
                'es': 'spanish',
                'de': 'german'
            }
            return lang_map.get(lang, 'english')
        except:
            return 'english'
    
    @staticmethod
    def get_language_instruction(language: str) -> str:
        """Get explicit language instruction for LLM"""
        instructions = {
            'french': "Tu DOIS répondre UNIQUEMENT en français. Toutes tes réponses, questions et explications doivent être en français.",
            'english': "You MUST respond ONLY in English. All your responses, questions and explanations must be in English.",
            'spanish': "DEBES responder SOLO en español. Todas tus respuestas, preguntas y explicaciones deben ser en español."
        }
        return instructions.get(language, instructions['english'])