# Audio Generator Service - ElevenLabs for French, OpenAI TTS for English
# Generates audio lectures/summaries based on user intent

import json
import os
import base64
import asyncio
import httpx
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from datetime import datetime

# ElevenLabs setup
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AVAILABLE = bool(ELEVENLABS_API_KEY)
if ELEVENLABS_AVAILABLE:
    print("✅ ElevenLabs API key found")
else:
    print("⚠️ ELEVENLABS_API_KEY not set. Will use fallback TTS for French.")

# Google Cloud TTS setup (fallback for French)
GOOGLE_TTS_AVAILABLE = False
try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
    print("✅ Google Cloud TTS available (fallback)")
except ImportError:
    print("⚠️ Google Cloud TTS not installed. Run: pip install google-cloud-texttospeech")


class AudioGenerator:
    """Generates audio content using ElevenLabs (French) and OpenAI TTS (English)"""

    # Intent configurations
    INTENT_CONFIG = {
        "teach": {
            "name": "Full Lesson",
            "description": "Structured lesson with examples and clinical context",
            "default_duration": "5min",
            "durations": ["2min", "5min", "10min"],
            "word_counts": {"2min": 300, "5min": 750, "10min": 1500}
        },
        "summarize": {
            "name": "Quick Summary",
            "description": "Key points only, concise overview",
            "default_duration": "2min",
            "durations": ["1min", "2min", "3min"],
            "word_counts": {"1min": 150, "2min": 300, "3min": 450}
        },
        "deep_dive": {
            "name": "Deep Dive",
            "description": "Comprehensive, detailed exploration",
            "default_duration": "10min",
            "durations": ["5min", "10min", "15min"],
            "word_counts": {"5min": 750, "10min": 1500, "15min": 2250}
        },
        "simplify": {
            "name": "Simple Explanation",
            "description": "Beginner-friendly, uses analogies",
            "default_duration": "3min",
            "durations": ["2min", "3min", "5min"],
            "word_counts": {"2min": 300, "3min": 450, "5min": 750}
        },
        "progress": {
            "name": "Progress Report",
            "description": "Your stats, achievements, and recommendations",
            "default_duration": "2min",
            "durations": ["1min", "2min"],
            "word_counts": {"1min": 150, "2min": 300}
        }
    }

    # Intent detection keywords
    INTENT_KEYWORDS = {
        "teach": ["teach me", "explain", "help me understand", "learn about", "tell me about", "what is", "how does"],
        "summarize": ["summarize", "summary", "quick overview", "key points", "brief", "tldr", "main points"],
        "deep_dive": ["deep dive", "in depth", "comprehensive", "everything about", "detailed", "thorough", "all about"],
        "simplify": ["simply", "simple", "like i'm 5", "eli5", "basics", "beginner", "easy way", "dumb it down"],
        "progress": ["my progress", "how am i doing", "my stats", "my performance", "how did i do", "my results"]
    }

    def __init__(self, session):
        self.session = session
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def detect_intent(self, user_message: str) -> str:
        """Detect user intent from message"""
        message_lower = user_message.lower()

        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return intent

        # Default to teach if no clear intent
        return "teach"

    def extract_topic(self, user_message: str, intent: str) -> str:
        """Extract topic from user message"""
        message_lower = user_message.lower()

        # Remove intent keywords to get topic
        for keyword in self.INTENT_KEYWORDS.get(intent, []):
            message_lower = message_lower.replace(keyword, "")

        # Clean up
        topic = message_lower.strip()
        for word in ["about", "on", "the", "a", "an", "please", "can you", "could you"]:
            topic = topic.replace(word, " ")

        topic = " ".join(topic.split()).strip()

        # If empty, use general topic
        if not topic or len(topic) < 3:
            topic = "the uploaded materials"

        return topic.title()

    async def get_audio_options(self, user_message: str) -> dict:
        """
        Analyze user message and return audio generation options
        This is called first to show the confirmation card
        """
        intent = self.detect_intent(user_message)
        topic = self.extract_topic(user_message, intent)
        config = self.INTENT_CONFIG[intent]

        return {
            "status": "audio_options",
            "intent": intent,
            "topic": topic,
            "style_name": config["name"],
            "style_description": config["description"],
            "durations": config["durations"],
            "default_duration": config["default_duration"]
        }

    async def generate_audio_stream(
        self,
        topic: str,
        intent: str,
        duration: str,
        language: str = "english",
        progress_data: Optional[dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate audio content with streaming progress updates
        """
        config = self.INTENT_CONFIG.get(intent, self.INTENT_CONFIG["teach"])
        word_count = config["word_counts"].get(duration, 750)

        # Phase 1: Start
        yield json.dumps({
            "status": "audio_generating",
            "message": f"Creating {config['name'].lower()} script...",
            "phase": "script"
        }) + "\n"

        # Get context
        if intent == "progress":
            context = self._format_progress_context(progress_data)
        else:
            context = await self._get_document_context(topic)

        if not context and intent != "progress":
            yield json.dumps({
                "status": "audio_error",
                "message": "No document content found. Please upload some files first."
            }) + "\n"
            return

        # Phase 2: Generate script
        try:
            script = await self._generate_script(
                topic=topic,
                intent=intent,
                context=context,
                word_count=word_count,
                language=language
            )

            yield json.dumps({
                "status": "audio_script_ready",
                "message": "Script ready, converting to speech...",
                "phase": "tts",
                "script_preview": script[:200] + "..." if len(script) > 200 else script
            }) + "\n"

        except Exception as e:
            print(f"Script generation error: {e}")
            yield json.dumps({
                "status": "audio_error",
                "message": f"Failed to generate script: {str(e)}"
            }) + "\n"
            return

        # Phase 3: Text-to-Speech
        try:
            yield json.dumps({
                "status": "audio_tts_progress",
                "message": "Generating audio...",
                "progress": 50
            }) + "\n"

            audio_base64, audio_duration = await self._text_to_speech(script, language)

            yield json.dumps({
                "status": "audio_ready",
                "audio_base64": audio_base64,
                "audio_duration": audio_duration,
                "topic": topic,
                "intent": intent,
                "script": script
            }) + "\n"

            yield json.dumps({
                "status": "audio_complete"
            }) + "\n"

        except Exception as e:
            print(f"TTS error: {e}")
            yield json.dumps({
                "status": "audio_error",
                "message": f"Failed to generate audio: {str(e)}"
            }) + "\n"

    async def _generate_script(
        self,
        topic: str,
        intent: str,
        context: str,
        word_count: int,
        language: str
    ) -> str:
        """Generate audio script based on intent"""

        prompt = self._get_script_prompt(topic, intent, context, word_count, language)

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=word_count * 2  # Allow some buffer
        )

        return response.choices[0].message.content.strip()

    def _get_script_prompt(
        self,
        topic: str,
        intent: str,
        context: str,
        word_count: int,
        language: str
    ) -> str:
        """Get prompt based on intent type"""

        if language == "french":
            lang_instruction = """Respond entirely in French.
CRITICAL FOR NATURAL FRENCH SPEECH:
- Use natural spoken French, not written/formal French
- Include teaching phrases like "Alors...", "Voyons...", "N'oubliez pas...", "En fait..."
- Use rhetorical questions to engage: "Pourquoi est-ce important?" "Vous voyez?"
- Add conversational connectors: "Donc", "Ensuite", "Par exemple", "Autrement dit"
- Sound like a warm, experienced French nursing professor speaking to students
- Vary sentence length - mix short punchy sentences with longer explanations
- Use "vous" (formal) but keep the tone warm and approachable"""
        else:
            lang_instruction = "Respond in English."

        base_instructions = f"""
You are creating an audio script that will be converted to speech.
Target length: approximately {word_count} words.
{lang_instruction}

IMPORTANT AUDIO GUIDELINES:
- Write naturally as if speaking directly to a student in a classroom
- Use short sentences that flow well when spoken
- Include brief pauses by using periods or ellipses
- Avoid bullet points, numbers, or formatting - this is for audio
- Don't say "in this audio" or reference the format
- Start directly with the content
- Sound warm, encouraging, and knowledgeable
"""

        if intent == "teach":
            return f"""{base_instructions}

You are a warm, knowledgeable nursing professor giving a mini-lecture.

STRUCTURE:
1. Hook: Start with why this topic matters in nursing practice
2. Core concepts: Explain the main ideas clearly
3. Examples: Give clinical scenarios or patient examples
4. Application: How to use this knowledge in practice
5. Quick recap: Summarize the key takeaways

TOPIC: {topic}

SOURCE MATERIAL:
{context}

Write the teaching script now:"""

        elif intent == "summarize":
            return f"""{base_instructions}

You are giving a quick, efficient summary - just the key points.

STRUCTURE:
1. One sentence intro
2. Three to five main points
3. One sentence conclusion

Be direct and concise. No fluff.

TOPIC: {topic}

SOURCE MATERIAL:
{context}

Write the summary script now:"""

        elif intent == "deep_dive":
            return f"""{base_instructions}

You are giving a comprehensive, detailed lecture covering all aspects.

STRUCTURE:
1. Overview and context
2. Detailed explanation of mechanisms/processes
3. Connections to related topics
4. Clinical implications and applications
5. Important considerations and edge cases
6. Summary of key points

Be thorough but keep it engaging.

TOPIC: {topic}

SOURCE MATERIAL:
{context}

Write the deep dive script now:"""

        elif intent == "simplify":
            return f"""{base_instructions}

You are explaining this to someone brand new to nursing - make it super simple.

STRUCTURE:
1. Use a relatable analogy from everyday life
2. Give the simplest possible definition
3. Provide a concrete real-world example
4. Explain why it matters for patient care

Use everyday language. Avoid medical jargon or explain any terms you use.

TOPIC: {topic}

SOURCE MATERIAL:
{context}

Write the simplified explanation now:"""

        elif intent == "progress":
            return f"""{base_instructions}

You are a supportive study coach giving a personal progress update.

Include:
1. Celebrate their achievements (level, streak, XP)
2. Highlight recent quiz performance
3. Note topics they're strong in
4. Gently mention areas for improvement
5. End with encouragement and next steps

Be warm and motivating, like a supportive mentor.

STUDENT DATA:
{context}

Write the progress report script now:"""

        return f"""{base_instructions}

TOPIC: {topic}

SOURCE MATERIAL:
{context}

Write an informative audio script:"""

    async def _text_to_speech(self, text: str, language: str) -> tuple[str, float]:
        """Convert text to speech - ElevenLabs for French (primary), Google Cloud fallback, OpenAI for English"""

        if language == "french":
            # Try ElevenLabs first (best quality)
            if ELEVENLABS_AVAILABLE:
                try:
                    return await self._elevenlabs_tts(text, language)
                except Exception as e:
                    print(f"⚠️ ElevenLabs TTS failed: {e}")

            # Fallback to Google Cloud TTS
            if GOOGLE_TTS_AVAILABLE:
                try:
                    return await self._google_tts(text, language)
                except Exception as e:
                    print(f"⚠️ Google TTS failed: {e}")

            # Final fallback to OpenAI
            return await self._openai_tts(text, language)
        else:
            return await self._openai_tts(text, language)

    async def _elevenlabs_tts(self, text: str, language: str) -> tuple[str, float]:
        """ElevenLabs TTS - native French teacher voice"""

        # Best native French voices for teaching:
        # - Juliette (French native, warm teacher) - voice_id: 5Q0t7uMcjvnagumLfvZi (Community)
        # - Amelie (French native, clear) - Use voice search for French natives
        # For now using Charlotte with optimized settings for teaching
        voice_id = "XB0fDUnXU5powFXDhCwa"  # Charlotte

        print(f"🎙️ Using ElevenLabs TTS for {language}...")

        # Add natural teaching pauses to the text
        teaching_text = self._add_teaching_pauses(text)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "text": teaching_text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.35,  # Lower = more expressive, dynamic (teacher-like)
                        "similarity_boost": 0.80,  # Keep voice clear
                        "style": 0.65,  # Higher = more emotional, engaging
                        "use_speaker_boost": True
                    }
                }
            )
            response.raise_for_status()
            audio_bytes = response.content

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Estimate duration
        word_count = len(text.split())
        estimated_duration = (word_count / 150) * 60

        print(f"✅ ElevenLabs TTS complete: {len(audio_bytes)} bytes")
        return audio_base64, estimated_duration

    def _add_teaching_pauses(self, text: str) -> str:
        """Add natural pauses for a teacher-like delivery"""
        import re

        # Add thoughtful pauses after key teaching phrases
        teaching_phrases = [
            (r"(Alors,?)", r"\1..."),  # "So..."
            (r"(Donc,?)", r"\1..."),  # "So/Therefore..."
            (r"(Voyons,?)", r"\1..."),  # "Let's see..."
            (r"(Écoutez,?)", r"\1..."),  # "Listen..."
            (r"(Regardez,?)", r"\1..."),  # "Look..."
            (r"(En fait,?)", r"\1..."),  # "Actually..."
            (r"(Autrement dit,?)", r"\1..."),  # "In other words..."
            (r"(C'est-à-dire,?)", r"\1..."),  # "That is to say..."
            (r"(Par exemple,?)", r"\1..."),  # "For example..."
            (r"(N'oubliez pas,?)", r"\1..."),  # "Don't forget..."
            (r"(Souvenez-vous,?)", r"\1..."),  # "Remember..."
            (r"(Attention,?)", r"\1..."),  # "Watch out/Note..."
        ]

        for pattern, replacement in teaching_phrases:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Add slight pauses before important transitions
        text = re.sub(r"\. (Premièrement|Deuxièmement|Troisièmement|Ensuite|Puis|Enfin|Finalement)",
                      r"... \1", text, flags=re.IGNORECASE)

        # Add pauses around questions (rhetorical teaching style)
        text = re.sub(r"\? ", r"?... ", text)

        # Add emphasis pause before "très" (very) and "vraiment" (really)
        text = re.sub(r" (très|vraiment) ", r"... \1 ", text, flags=re.IGNORECASE)

        return text

    async def _google_tts(self, text: str, language: str) -> tuple[str, float]:
        """Google Cloud TTS - high quality neural voices for French"""

        print(f"🎙️ Using Google Cloud TTS for {language}...")

        # Run in thread pool since google-cloud-texttospeech is synchronous
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            self._google_tts_sync,
            text
        )

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Estimate duration
        word_count = len(text.split())
        estimated_duration = (word_count / 150) * 60

        print(f"✅ Google TTS complete: {len(audio_bytes)} bytes")
        return audio_base64, estimated_duration

    def _google_tts_sync(self, text: str) -> bytes:
        """Synchronous Google Cloud TTS call with natural pacing"""
        client = texttospeech.TextToSpeechClient()

        # Convert text to SSML with natural pauses and pacing
        ssml_text = self._text_to_ssml(text)

        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

        # Use Studio voice for most natural sound (if available)
        # Fallback order: Studio > Wavenet > Neural2
        # fr-FR-Studio-A is the most natural French female voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="fr-FR",
            name="fr-FR-Wavenet-C",  # Wavenet female - very natural
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,  # Slightly slower for clarity
            pitch=1.0,  # Slightly higher for warmth
            volume_gain_db=0.0,
            effects_profile_id=["headphone-class-device"]  # Optimized for headphones
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        return response.audio_content

    def _text_to_ssml(self, text: str) -> str:
        """Convert plain text to SSML with natural pauses and emphasis"""
        import re

        # Escape special XML characters
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")

        # Add medium pauses after periods (natural sentence breaks)
        text = re.sub(r'\. ', r'.<break time="400ms"/> ', text)

        # Add shorter pauses after commas
        text = re.sub(r', ', r',<break time="200ms"/> ', text)

        # Add pauses after colons
        text = re.sub(r': ', r':<break time="300ms"/> ', text)

        # Add pauses after semicolons
        text = re.sub(r'; ', r';<break time="300ms"/> ', text)

        # Add longer pauses after question marks
        text = re.sub(r'\? ', r'?<break time="500ms"/> ', text)

        # Add longer pauses after exclamation marks
        text = re.sub(r'! ', r'!<break time="400ms"/> ', text)

        # Add pauses around ellipsis for dramatic effect
        text = re.sub(r'\.\.\.', r'<break time="600ms"/>', text)

        # Add emphasis to key nursing/medical terms (common patterns)
        emphasis_words = [
            "important", "essentiel", "critique", "attention",
            "rappel", "noter", "retenir", "crucial", "clé"
        ]
        for word in emphasis_words:
            pattern = re.compile(rf'\b({word})\b', re.IGNORECASE)
            text = pattern.sub(r'<emphasis level="moderate">\1</emphasis>', text)

        # Wrap in prosody for overall natural speech
        ssml = f'''<speak>
            <prosody rate="95%" pitch="+5%">
                {text}
            </prosody>
        </speak>'''

        return ssml

    async def _openai_tts(self, text: str, language: str) -> tuple[str, float]:
        """OpenAI TTS - nova for English, shimmer for French fallback"""
        voice = "shimmer" if language == "french" else "nova"

        print(f"🎙️ Using OpenAI TTS ({voice}) for {language}...")

        response = await self.client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            response_format="mp3"
        )

        audio_bytes = response.content
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Estimate duration
        word_count = len(text.split())
        estimated_duration = (word_count / 150) * 60

        print(f"✅ OpenAI TTS complete: {len(audio_bytes)} bytes")
        return audio_base64, estimated_duration

    async def _get_document_context(self, topic: str) -> str:
        """Get relevant document context from vectorstore"""
        try:
            session = self.session

            if session.vectorstore is None and session.documents:
                from tools.quiztools import load_vectorstore_from_firebase
                session.vectorstore = await load_vectorstore_from_firebase(session)
                session.vectorstore_loaded = True

            if session.vectorstore:
                docs = session.vectorstore.similarity_search(query=topic, k=30)

                # Filter and combine chunks
                good_chunks = []
                for doc in docs:
                    content = doc.page_content.strip()
                    if len(content) > 80:
                        good_chunks.append(content)

                context = "\n\n".join(good_chunks[:20])

                if len(context) > 10000:
                    context = context[:10000]

                print(f"🎙️ Audio context: {len(good_chunks)} chunks, {len(context)} chars")
                return context
            else:
                print("⚠️ No vectorstore for audio")
                return ""

        except Exception as e:
            print(f"Error getting audio context: {e}")
            return ""

    def _format_progress_context(self, progress_data: Optional[dict]) -> str:
        """Format progress data for the script"""
        if not progress_data:
            return "No progress data available yet. The student is just getting started."

        return f"""
Student Progress:
- Current Level: {progress_data.get('level', 1)}
- Total XP: {progress_data.get('total_xp', 0)}
- XP to Next Level: {progress_data.get('xp_needed', 100)}
- Current Streak: {progress_data.get('streak', 0)} days
- Daily Goal Progress: {progress_data.get('daily_correct', 0)}/{progress_data.get('daily_goal', 5)} correct answers

Recent Quiz Performance:
{progress_data.get('recent_quizzes', 'No quizzes taken yet')}

Topics Studied:
{progress_data.get('topics_studied', 'None yet')}
"""
