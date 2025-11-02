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
    async def detect_language(text: str) -> str:
        """
        Detect language from text.
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
            
            # Call gpt-4o-mini (more reliable than instruct)
            client = LanguageDetector._get_client()
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You detect the language of text. Respond with ONLY the language name in lowercase (e.g., 'english', 'french', 'spanish', 'chinese', 'tagalog'). Nothing else."
                    },
                    {
                        "role": "user",
                        "content": f"Language: {text}"
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
            print(f"🌐 Detected language: {detected}")
            return detected
            
        except Exception as e:
            print(f"❌ Language detection failed: {e}")
            import traceback
            traceback.print_exc()
            return 'english'