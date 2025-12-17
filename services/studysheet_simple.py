import json
import os
from typing import AsyncGenerator

import anthropic
from openai import OpenAI


class SimpleStudySheetGenerator:
    """Generates study sheets using GPT-5-mini streaming with Anthropic fallback."""

    def __init__(self, session):
        self.session = session
        self.client = anthropic.Anthropic()
        self.openai_client = OpenAI()

    async def generate_study_sheet_stream(
        self,
        topic: str,
        language: str = "english"
    ) -> AsyncGenerator[str, None]:
        """Stream a study sheet using Claude; fall back to ChatGPT when needed."""

        context = await self._get_document_context(topic)

        if not context:
            yield json.dumps({
                "status": "study_sheet_error",
                "message": "No document content found"
            }) + "\n"
            return

        yield json.dumps({
            "status": "study_sheet_start",
            "topic": topic
        }) + "\n"

        prompt = self._get_prompt(topic, context, language)
        anthropic_error = None

        # Primary path: OpenAI GPT-5-mini streaming
        try:
            async for chunk in self._stream_with_openai(prompt):
                yield chunk
            return
        except Exception as e:
            print(f"OpenAI study sheet stream failed, falling back to Anthropic: {e}")
            yield json.dumps({
                "status": "study_sheet_provider",
                "provider": "openai-gpt-5-mini",
                "message": "OpenAI unavailable, switching to Anthropic fallback"
            }) + "\n"

        # Secondary path: Anthropic streaming (preferred for formatting fidelity)
        try:
            for chunk in self._stream_with_anthropic(prompt):
                yield chunk
            return
        except Exception as e:
            anthropic_error = e
            print(f"Anthropic study sheet failed, falling back to OpenAI single shot: {e}")
            yield json.dumps({
                "status": "study_sheet_provider",
                "provider": "anthropic",
                "message": "Anthropic unavailable, switching to ChatGPT fallback"
            }) + "\n"

        # Fallback path: Single-response ChatGPT output (keeps the same prompt)
        try:
            fallback_content = self._generate_with_openai(prompt)
            fallback_content = self._strip_code_fences(fallback_content)

            if not fallback_content:
                raise ValueError("Empty response from OpenAI fallback")

            yield json.dumps({
                "status": "study_sheet_chunk",
                "content": fallback_content
            }) + "\n"
            yield json.dumps({
                "status": "study_sheet_complete"
            }) + "\n"
        except Exception as openai_error:
            print(f"OpenAI fallback failed: {openai_error}")
            detail = f"Anthropic error: {anthropic_error}" if anthropic_error else "Upstream providers unavailable"
            yield json.dumps({
                "status": "study_sheet_error",
                "message": f"{detail}; OpenAI error: {openai_error}"
            }) + "\n"

    def _get_prompt(self, topic: str, context: str, language: str) -> str:
        """Generate prompt for study sheet."""

        if language == "french":
            return f"""Cree un guide d'etude infirmier sur: {topic}

Informations source:
{context}

Format requis (suivre exactement):
- Titres de section: une ligne en MAJUSCULES
- Sous-titres: une ligne se terminant par deux-points
- Listes: numerotees 1. 2. 3.
- Texte: phrases completes et claires

Exemple de format:

INTRODUCTION

Ce guide couvre les concepts essentiels de [sujet].

Concepts cles:
1. Premier concept avec explication detaillee
2. Deuxieme concept avec explication detaillee

SECTION SUIVANTE

Contenu de la section...

Ecris maintenant le guide complet:"""

        return f"""Create a nursing study guide about: {topic}

Source information:
{context}

Required format (follow exactly):
- Section titles: a line in ALL CAPS
- Subtitles: a line ending with a colon
- Lists: numbered 1. 2. 3.
- Text: complete clear sentences

Example format:

INTRODUCTION

This guide covers the essential concepts of [topic].

Key concepts:
1. First concept with detailed explanation
2. Second concept with detailed explanation

        NEXT SECTION

        Section content here...

        Write the complete guide now:"""

    def _stream_with_anthropic(self, prompt: str):
        """Stream study sheet content from Anthropic."""
        with self.client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield json.dumps({
                        "status": "study_sheet_chunk",
                        "content": text
                    }) + "\n"

        yield json.dumps({
            "status": "study_sheet_complete"
        }) + "\n"

    def _generate_with_openai(self, prompt: str) -> str:
        """Fallback generator using OpenAI ChatGPT."""
        response = self.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_STUDY_SHEET_MODEL", "gpt-5-mini"),
            max_tokens=8000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content if response.choices else ""

    async def _stream_with_openai(self, prompt: str):
        """Stream study sheet content from OpenAI (GPT-5-mini by default)."""
        response = self.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_STUDY_SHEET_MODEL", "gpt-5-mini"),
            temperature=0.3,
            max_tokens=8000,
            stream=True,
            messages=[{"role": "user", "content": prompt}]
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield json.dumps({
                    "status": "study_sheet_chunk",
                    "content": chunk.choices[0].delta.content
                }) + "\n"

        yield json.dumps({
            "status": "study_sheet_complete"
        }) + "\n"

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove common markdown fences the model might add."""
        if not text:
            return ""

        cleaned = text.strip()

        if cleaned.startswith("```html"):
            cleaned = cleaned[len("```html"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        return cleaned

    async def _get_document_context(self, topic: str) -> str:
        """Get document context from vectorstore."""
        try:
            session = self.session

            if session.vectorstore is None and session.documents:
                from tools.quiztools import load_vectorstore_from_firebase
                session.vectorstore = await load_vectorstore_from_firebase(session)
                session.vectorstore_loaded = True

            if session.vectorstore:
                docs = session.vectorstore.similarity_search(query=topic, k=50)

                # Filter quality chunks
                good_chunks = []
                for doc in docs:
                    content = doc.page_content.strip()
                    if len(content) > 80:
                        good_chunks.append(content)

                context = "\n\n".join(good_chunks[:40])

                max_chars = 120000  # protect downstream model while allowing rich context
                if len(context) > max_chars:
                    context = context[:max_chars]

                print(f"\\U0001f4da Study sheet: {len(good_chunks)} chunks, {len(context)} chars")
                return context
            else:
                print("No vectorstore available for study sheet generation")
                return ""

        except Exception as e:
            print(f"Error: {e}")
            return ""
