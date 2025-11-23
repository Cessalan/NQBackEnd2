from openai import AsyncOpenAI
import os

class LanguageDetector:
    """
    Reliable language detection using gpt-4o-mini.
    Cost: ~$0.000015 per detection (still nearly free).
    """
    
    _client = None
    _cache = {}
    
    @staticmethod
    def _get_client():
        if LanguageDetector._client is None:
            LanguageDetector._client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        return LanguageDetector._client
    
    @staticmethod
    async def detect_language(text: str, chat_history: list = None) -> str:
        """
        Detect language from text with chat context awareness.

        Args:
            text: Current user input
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]

        Returns: Language name in lowercase (e.g., 'english', 'french', 'tagalog')
        """
        try:
            # Handle edge cases
            if not text or len(text.strip()) < 2:
                print(f"⚠️ Text too short, defaulting to English")
                return 'english'

            # Check cache
            cache_key = text.lower().strip()[:100]
            if cache_key in LanguageDetector._cache:
                cached = LanguageDetector._cache[cache_key]
                print(f"🌐 Language (cached): {cached}")
                return cached

            # Build context-aware prompt
            client = LanguageDetector._get_client()

            # Extract recent chat context (last 3-5 messages)
            context_snippet = ""
            if chat_history and len(chat_history) > 0:
                recent_messages = chat_history[-5:]  # Last 5 messages for context
                context_parts = []
                for msg in recent_messages:
                    if isinstance(msg, dict) and 'content' in msg:
                        content = msg['content'][:200]  # First 200 chars
                        context_parts.append(content)

                if context_parts:
                    context_snippet = "\n".join(context_parts[-3:])  # Last 3 for brevity

            # Build prompt with context
            if context_snippet:
                prompt = f"""Previous conversation context:
{context_snippet}

Current user message:
{text}

Based on the conversation history and current message, what language is the user communicating in?
Respond with ONLY the language name in lowercase (e.g., 'english', 'french', 'spanish', 'chinese', 'tagalog')."""
            else:
                prompt = f"Language: {text}"

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You detect the language of text. Consider the conversation history to determine the language. Respond with ONLY the language name in lowercase (e.g., 'english', 'french', 'spanish', 'chinese', 'tagalog'). Nothing else."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=10,
                temperature=0
            )

            # Parse response
            detected = response.choices[0].message.content.strip().lower()

            # Clean up - take only first word if multiple returned
            detected = detected.split()[0] if detected else 'english'

            # Remove any punctuation
            detected = ''.join(c for c in detected if c.isalpha())

            # Validate it's not empty
            if not detected:
                print(f"⚠️ Empty detection result, defaulting to English")
                return 'english'

            # Cache and return
            LanguageDetector._cache[cache_key] = detected
            print(f"🌐 Detected language: {detected} (with context: {bool(context_snippet)})")
            return detected

        except Exception as e:
            print(f"❌ Language detection failed: {e}")
            import traceback
            traceback.print_exc()
            return 'english'