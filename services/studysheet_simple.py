# Simple Study Sheet Generator - Using Claude

import json
from typing import AsyncGenerator
import anthropic


class SimpleStudySheetGenerator:
    """Generates study sheets using Claude for better quality"""

    def __init__(self, session):
        self.session = session
        self.client = anthropic.Anthropic()

    async def generate_study_sheet_stream(
        self,
        topic: str,
        language: str = "english"
    ) -> AsyncGenerator[str, None]:
        """Stream a study sheet using Claude."""

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

        try:
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

        except Exception as e:
            print(f"Error generating study sheet: {e}")
            yield json.dumps({
                "status": "study_sheet_error",
                "message": str(e)
            }) + "\n"

    def _get_prompt(self, topic: str, context: str, language: str) -> str:
        """Generate prompt for study sheet"""

        if language == "french":
            return f"""Crée un guide d'étude infirmier sur: {topic}

Informations source:
{context}

Format requis (suivre exactement):
- Titres de section: une ligne en MAJUSCULES
- Sous-titres: une ligne se terminant par deux-points
- Listes: numérotées 1. 2. 3.
- Texte: phrases complètes et claires

Exemple de format:

INTRODUCTION

Ce guide couvre les concepts essentiels de [sujet].

Concepts clés:
1. Premier concept avec explication détaillée
2. Deuxième concept avec explication détaillée

SECTION SUIVANTE

Contenu de la section...

Écris maintenant le guide complet:"""

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

    async def _get_document_context(self, topic: str) -> str:
        """Get document context from vectorstore"""
        try:
            session = self.session

            if session.vectorstore is None and session.documents:
                from tools.quiztools import load_vectorstore_from_firebase
                session.vectorstore = await load_vectorstore_from_firebase(session)
                session.vectorstore_loaded = True

            if session.vectorstore:
                docs = session.vectorstore.similarity_search(query=topic, k=25)

                # Filter quality chunks
                good_chunks = []
                for doc in docs:
                    content = doc.page_content.strip()
                    if len(content) > 80:
                        good_chunks.append(content)

                context = "\n\n".join(good_chunks[:15])

                if len(context) > 8000:
                    context = context[:8000]

                print(f"📚 Study sheet: {len(good_chunks)} chunks, {len(context)} chars")
                return context
            else:
                print("⚠️ No vectorstore")
                return ""

        except Exception as e:
            print(f"Error: {e}")
            return ""
