"""
Mindmap Generator Service
Generates hierarchical mindmap data from document content using LLM.
"""

from typing import AsyncGenerator, Dict, Any, List
from langchain_openai import ChatOpenAI
from models.session import PersistentSessionContext
import json
import os


async def stream_mindmap_data(
    topic: str,
    depth: str,
    session: PersistentSessionContext,
    chat_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate mindmap data from documents and stream the result.

    Args:
        topic: Optional focus topic (empty = entire document)
        depth: How deep to go ("shallow", "medium", "deep")
        session: Current session context with vectorstore
        chat_id: Chat ID for context

    Yields:
        - { status: "mindmap_generating", message: "..." }
        - { status: "mindmap_complete", mindmap_data: {...} }
        - { status: "error", message: "..." }
    """

    # Step 1: Notify frontend we're starting
    yield {
        "status": "mindmap_generating",
        "message": "Analyzing document structure..."
    }

    # Step 2: Get document content from vectorstore
    if not session.vectorstore:
        yield {
            "status": "error",
            "message": "No documents found. Please upload files first."
        }
        return

    # Search for relevant content - use multiple queries to ensure comprehensive coverage
    query = topic if topic else "main concepts key topics important ideas structure overview"

    try:
        # Get more chunks for comprehensive coverage
        docs = session.vectorstore.similarity_search(query, k=20)
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        yield {
            "status": "error",
            "message": "Could not retrieve document content."
        }
        return

    if not docs:
        yield {
            "status": "error",
            "message": "Could not retrieve document content."
        }
        return

    # Combine document content
    content = "\n\n".join([doc.page_content for doc in docs])

    # Limit content size to avoid token limits - increased for comprehensive coverage
    max_content_length = 15000
    if len(content) > max_content_length:
        content = content[:max_content_length]

    yield {
        "status": "mindmap_generating",
        "message": "Extracting concepts and relationships..."
    }

    # Step 3: Generate mindmap structure with LLM
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        openai_api_key=api_key
    )

    # Determine depth instructions - prioritize comprehensive coverage
    depth_instructions = {
        "shallow": "Extract ALL main topics (minimum 5-8) as primary branches. Include every important concept.",
        "medium": "Extract ALL main topics, each with ALL relevant sub-concepts. Do NOT omit any important information. Aim for 6-10 main topics with 3-5 sub-concepts each.",
        "deep": "Extract EVERYTHING: all main topics, all sub-concepts, and all specific details. This is for students who need comprehensive review. Aim for 8-12 main topics, each with multiple levels of detail."
    }

    # Language handling
    language = session.user_language if hasattr(session, 'user_language') else 'english'

    prompt = f"""You are creating a comprehensive study mindmap for students who need to quickly understand ALL important content from this document. This is for last-minute study - DO NOT OMIT ANY IMPORTANT CONCEPT.

Document Content:
{content}

CRITICAL INSTRUCTIONS:
1. Extract EVERY important concept, fact, and piece of information from the document
2. {depth_instructions.get(depth, depth_instructions["medium"])}
3. Keep labels concise (2-6 words) but ensure they capture the actual content, not just headers
4. The "details" array is CRUCIAL - include specific facts, examples, numbers, lists, and key points from the document
5. DO NOT just extract section headers - extract the ACTUAL CONTENT and what students need to know
6. If the document mentions benefits, causes, effects, symptoms, treatments, etc. - LIST THEM ALL in the details array
7. Think like a student: "What do I need to memorize from this section?"

Language for all content: {language}

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
    "central_topic": "Main Document Topic",
    "nodes": [
        {{
            "id": "root",
            "label": "Central Topic",
            "type": "central",
            "children": ["node_1", "node_2"],
            "parent": null,
            "summary": "Brief overview of the document",
            "details": ["Key fact 1", "Key fact 2", "Key fact 3"]
        }},
        {{
            "id": "node_1",
            "label": "Main Concept",
            "type": "main",
            "children": ["node_1a", "node_1b"],
            "parent": "root",
            "summary": "What this concept is about",
            "details": ["Specific fact from document", "Another important point", "Example or statistic mentioned"]
        }},
        {{
            "id": "node_1a",
            "label": "Sub-concept",
            "type": "sub",
            "children": [],
            "parent": "node_1",
            "summary": "Explanation of sub-concept",
            "details": ["Detail 1", "Detail 2"]
        }}
    ],
    "edges": [
        {{ "source": "root", "target": "node_1" }},
        {{ "source": "node_1", "target": "node_1a" }}
    ]
}}

REQUIREMENTS:
- Every node MUST have a "details" array with specific information from the document
- DO NOT leave details empty - if a concept is mentioned, extract what the document says about it
- Node types: "central" (root only), "main" (first level), "sub" (second level), "detail" (third level)
- Every node except root must have a parent
- Every parent-child relationship must have a corresponding edge
- IDs must be unique and referenced correctly
- Return ONLY the JSON, no other text
- PRIORITIZE COMPLETENESS: It's better to have more nodes than to miss important content
"""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are a study assistant creating comprehensive mindmaps for students. Your goal is to extract ALL important information - never omit concepts. Include specific facts, examples, and details in each node. Return ONLY valid JSON, no markdown code blocks."},
            {"role": "user", "content": prompt}
        ])

        # Parse response
        response_text = response.content.strip()

        # Clean markdown code blocks if present
        if response_text.startswith("```"):
            # Find the first newline after ``` and last ```
            lines = response_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    # Skip the opening ``` line (might have 'json' after it)
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        response_text = response_text.strip()

        mindmap_data = json.loads(response_text)

        # Validate structure
        if "nodes" not in mindmap_data or "edges" not in mindmap_data:
            raise ValueError("Missing nodes or edges in response")

        if len(mindmap_data["nodes"]) == 0:
            raise ValueError("No nodes in mindmap")

        print(f"✅ Mindmap generated with {len(mindmap_data['nodes'])} nodes and {len(mindmap_data['edges'])} edges")

        # Step 4: Send complete mindmap
        yield {
            "status": "mindmap_complete",
            "mindmap_data": mindmap_data
        }

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse mindmap JSON: {e}")
        print(f"Response was: {response_text[:500]}")
        yield {
            "status": "error",
            "message": "Failed to generate mindmap structure. Please try again."
        }
    except ValueError as e:
        print(f"❌ Invalid mindmap structure: {e}")
        yield {
            "status": "error",
            "message": f"Invalid mindmap structure: {str(e)}"
        }
    except Exception as e:
        print(f"❌ Mindmap generation error: {e}")
        yield {
            "status": "error",
            "message": "An error occurred while generating the mindmap. Please try again."
        }
