# Backend Requirements for Mindmap Generation

## Overview
This document outlines the backend changes needed to support mindmap generation from uploaded documents, triggered through chat conversation.

---

## How It Works (Same Pattern as Quiz/Flashcards)

### Flow
```
1. User types: "Create a mindmap from my notes"
2. LLM detects intent → calls generate_mindmap_stream tool
3. Tool returns: { status: "mindmap_streaming_initiated", metadata: {...} }
4. Orchestrator catches this → calls stream_mindmap_data()
5. Streaming sends: { status: "mindmap_generating" } then { status: "mindmap_complete", mindmap_data: {...} }
6. Frontend renders the mindmap
```

---

## Implementation Steps

### Step 1: Create Mindmap Tool
**File**: `tools/quiztools.py` (or new `tools/mindmap_tools.py`)

```python
@tool
async def generate_mindmap_stream(
    topic: str = "",
    depth: str = "medium"
) -> Dict[str, Any]:
    """
    Generate a visual mindmap from the student's uploaded documents.

    Use this when students ask to:
    - Create a mindmap
    - Visualize concepts
    - Show a concept map
    - Map out the material
    - Create a visual summary
    - "carte mentale" (French)

    Args:
        topic: Optional focus topic (empty = entire document)
        depth: How deep to go ("shallow" = main topics only, "medium" = 2 levels, "deep" = 3+ levels)

    Returns:
        Signal to orchestrator to begin mindmap generation
    """
    print("🧠 MINDMAP TOOL: Initiating mindmap generation")

    try:
        session = get_session()

        # Determine source - must have documents
        if not session.documents:
            return {
                "status": "error",
                "message": "No documents uploaded. Please upload study materials first."
            }

        # Normalize depth
        depth_map = {
            "shallow": "shallow", "simple": "shallow", "basic": "shallow",
            "medium": "medium", "normal": "medium", "standard": "medium",
            "deep": "deep", "detailed": "deep", "comprehensive": "deep"
        }
        normalized_depth = depth_map.get(depth.lower(), "medium")

        return {
            "status": "mindmap_streaming_initiated",
            "metadata": {
                "topic": topic,
                "depth": normalized_depth,
                "language": session.user_language,
                "document_count": len(session.documents)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Mindmap generation failed: {str(e)}"
        }
```

### Step 2: Create Mindmap Streaming Function
**File**: `services/mindmap_generator.py` (new file)

```python
from typing import AsyncGenerator, Dict, Any, List
from langchain_openai import ChatOpenAI
from models.session import PersistentSessionContext
import json

async def stream_mindmap_data(
    topic: str,
    depth: str,
    session: PersistentSessionContext,
    chat_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate mindmap data from documents and stream the result.

    Yields:
        - { status: "mindmap_generating", message: "..." }
        - { status: "mindmap_complete", mindmap_data: {...} }
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

    # Search for relevant content
    query = topic if topic else "main concepts and key topics"
    docs = session.vectorstore.similarity_search(query, k=10)

    if not docs:
        yield {
            "status": "error",
            "message": "Could not retrieve document content."
        }
        return

    # Combine document content
    content = "\n\n".join([doc.page_content for doc in docs])

    yield {
        "status": "mindmap_generating",
        "message": "Extracting concepts and relationships..."
    }

    # Step 3: Generate mindmap structure with LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Determine depth instructions
    depth_instructions = {
        "shallow": "Extract only 4-6 main topics as primary branches. No sub-concepts.",
        "medium": "Extract 4-8 main topics, each with 2-4 sub-concepts (2 levels deep).",
        "deep": "Extract 4-8 main topics, each with sub-concepts, and those with further details (3 levels deep)."
    }

    prompt = f"""Analyze this document content and create a hierarchical mindmap structure.

Document Content:
{content[:8000]}

Instructions:
1. Identify the MAIN TOPIC (central node) - this should summarize the entire document
2. {depth_instructions.get(depth, depth_instructions["medium"])}
3. Keep labels concise (2-5 words maximum)
4. Add a brief summary (1 sentence) for each node
5. Create edges connecting parent nodes to children

Language: {session.user_language}

Return ONLY valid JSON in this exact format:
{{
    "central_topic": "Main Document Topic",
    "nodes": [
        {{
            "id": "root",
            "label": "Central Topic Name",
            "type": "central",
            "children": ["node_1", "node_2"],
            "parent": null,
            "summary": "One sentence overview"
        }},
        {{
            "id": "node_1",
            "label": "Main Concept 1",
            "type": "main",
            "children": ["node_1a", "node_1b"],
            "parent": "root",
            "summary": "Description of this concept"
        }},
        {{
            "id": "node_1a",
            "label": "Sub-concept 1a",
            "type": "sub",
            "children": [],
            "parent": "node_1",
            "summary": "Details about sub-concept"
        }}
    ],
    "edges": [
        {{ "source": "root", "target": "node_1" }},
        {{ "source": "node_1", "target": "node_1a" }}
    ]
}}

IMPORTANT:
- Every node except root must have a parent
- Every parent-child relationship must have a corresponding edge
- Node types: "central" (root only), "main" (first level), "sub" (second level), "detail" (third level)
- IDs must be unique and referenced correctly in children/parent/edges
"""

    response = await llm.ainvoke([
        {"role": "system", "content": "You extract document structure into mindmap JSON. Return only valid JSON."},
        {"role": "user", "content": prompt}
    ])

    # Parse response
    try:
        response_text = response.content.strip()
        # Clean markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        mindmap_data = json.loads(response_text)

        # Validate structure
        if "nodes" not in mindmap_data or "edges" not in mindmap_data:
            raise ValueError("Missing nodes or edges in response")

        # Step 4: Send complete mindmap
        yield {
            "status": "mindmap_complete",
            "mindmap_data": mindmap_data
        }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Failed to parse mindmap JSON: {e}")
        yield {
            "status": "error",
            "message": "Failed to generate mindmap structure. Please try again."
        }
```

### Step 3: Update Orchestrator
**File**: `services/orchestrator.py`

Add this in the tool execution section (around line 288, after flashcard handling):

```python
elif tool_name == "generate_mindmap_stream":
    print("🧠 Mindmap tool called")

    from tools.quiztools import generate_mindmap_stream
    result = await generate_mindmap_stream.ainvoke(tool_args)

    if result.get("status") == "mindmap_streaming_initiated":
        print("🌊 Starting mindmap streaming from orchestrator")

        metadata = result.get("metadata", {})

        from services.mindmap_generator import stream_mindmap_data

        async for chunk in stream_mindmap_data(
            topic=metadata.get("topic", ""),
            depth=metadata.get("depth", "medium"),
            session=self.session,
            chat_id=self.session.chat_id
        ):
            status = chunk.get("status")

            if status == "mindmap_generating":
                yield json.dumps({
                    "status": "mindmap_generating",
                    "message": chunk.get("message")
                }) + "\n"

            elif status == "mindmap_complete":
                yield json.dumps({
                    "status": "mindmap_complete",
                    "mindmap_data": chunk.get("mindmap_data")
                }) + "\n"

            elif status == "error":
                yield json.dumps({
                    "status": "error",
                    "message": chunk.get("message")
                }) + "\n"
    else:
        yield json.dumps({
            "status": "error",
            "message": result.get("message", "Mindmap generation failed")
        }) + "\n"

    return
```

### Step 4: Register Tool in NursingTools
**File**: `tools/quiztools.py`

In the `NursingTools` class `get_tools()` method, add:

```python
def get_tools(self):
    return [
        search_documents,
        summarize_document,
        check_student_progress,
        respond_to_student,
        generate_quiz_stream,
        generate_flashcards_stream,
        generate_study_sheet_stream,
        generate_mindmap_stream,  # ADD THIS
    ]
```

---

## Data Structure

### Mindmap Response Schema

```json
{
    "status": "mindmap_complete",
    "mindmap_data": {
        "central_topic": "Cardiovascular System",
        "nodes": [
            {
                "id": "root",
                "label": "Cardiovascular System",
                "type": "central",
                "children": ["node_1", "node_2", "node_3"],
                "parent": null,
                "summary": "Study of the heart and blood vessels"
            },
            {
                "id": "node_1",
                "label": "Heart Anatomy",
                "type": "main",
                "children": ["node_1a", "node_1b"],
                "parent": "root",
                "summary": "Structure and chambers of the heart"
            },
            {
                "id": "node_1a",
                "label": "Four Chambers",
                "type": "sub",
                "children": [],
                "parent": "node_1",
                "summary": "Left/right atria and ventricles"
            }
        ],
        "edges": [
            { "source": "root", "target": "node_1" },
            { "source": "root", "target": "node_2" },
            { "source": "node_1", "target": "node_1a" }
        ]
    }
}
```

### Node Types
| Type | Description | Visual |
|------|-------------|--------|
| `central` | Root node (document title) | Largest, distinct color |
| `main` | Primary branches (main topics) | Medium size |
| `sub` | Secondary branches (sub-topics) | Smaller |
| `detail` | Tertiary branches (details) | Smallest |

---

## Intent Detection Keywords

The LLM should detect mindmap intent from these phrases:

### English
- "create a mindmap"
- "mind map this"
- "concept map"
- "visualize the concepts"
- "map out the topics"
- "visual summary"
- "show me a diagram"

### French
- "carte mentale"
- "créer une carte mentale"
- "schéma conceptuel"
- "visualiser les concepts"
- "carte des concepts"

---

## WebSocket Message Flow

```
Frontend → Backend: { type: "chat_message", input: "Create a mindmap" }

Backend → Frontend: { type: "stream_chunk", data: { status: "tool_executing", tool_name: "generate_mindmap_stream" } }

Backend → Frontend: { type: "stream_chunk", data: { status: "mindmap_generating", message: "Analyzing..." } }

Backend → Frontend: { type: "stream_chunk", data: { status: "mindmap_complete", mindmap_data: {...} } }

Backend → Frontend: { type: "stream_complete" }
```

---

## Quality Guidelines

1. **Central Topic**: Clear, concise (2-5 words)
2. **Main Branches**: 4-8 topics (not overwhelming)
3. **Sub-concepts**: 2-4 per main branch
4. **Labels**: Maximum 5 words
5. **Summaries**: 1 sentence maximum
6. **Language**: Match user's language (English/French)

---

## Testing Checklist

- [ ] Tool registered in NursingTools.get_tools()
- [ ] Orchestrator handles "generate_mindmap_stream" tool calls
- [ ] LLM detects mindmap intent correctly
- [ ] Mindmap generates from uploaded documents
- [ ] Error handling for no documents uploaded
- [ ] French language support works
- [ ] WebSocket streaming format correct
- [ ] JSON parsing handles edge cases

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `services/mindmap_generator.py` | CREATE | Mindmap streaming logic |
| `tools/quiztools.py` | MODIFY | Add `generate_mindmap_stream` tool |
| `services/orchestrator.py` | MODIFY | Handle mindmap tool in process_message |

---

*Created: 2025-12-16*
*Status: Ready for Implementation*
