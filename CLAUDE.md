# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) chatbot system for answering questions about course materials. It uses:
- **Vector embeddings** for semantic search (ChromaDB + sentence-transformers)
- **Anthropic Claude** for AI-powered responses with tool-calling
- **FastAPI** backend with vanilla JavaScript frontend
- **Tool-based architecture** where Claude decides when/how to search

**Important**: Always use `uv` for package management — never use `pip` directly. This project uses `uv` as its Python package manager (see `pyproject.toml`).

## Running the Application

```bash
# Install dependencies (requires Python 3.13+ and uv)
uv sync

# Set up environment (requires Anthropic API key)
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY=your_key_here

# Run the application
chmod +x run.sh && ./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Architecture

### Three-Layer Design

```
Frontend (vanilla JS)  ←→  Backend (FastAPI)  ←→  RAG Core (orchestrator)
                                              ↓
                            [Components: AI Generator, Vector Store,
                             Document Processor, Session Manager, Tools]
```

### Key Backend Components

| File | Responsibility |
|------|----------------|
| `app.py` | FastAPI entry point, API endpoints (`/api/query`, `/api/courses`), document loading on startup |
| `rag_system.py` | **Central orchestrator** - wires all components together, handles `query()` and document ingestion |
| `ai_generator.py` | Claude API client with tool support, manages conversation history |
| `search_tools.py` | Tool registry and `CourseSearchTool` - semantic search with source tracking |
| `vector_store.py` | ChromaDB wrapper - **dual collections**: `course_catalog` (metadata) and `course_content` (chunks) |
| `document_processor.py` | Parses course documents, chunks text with overlap, extracts course/lesson metadata |
| `session_manager.py` | In-memory conversation history per session ID |
| `config.py` | Configuration dataclass with defaults |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk`, `QueryResponse` |

### Query Flow (Critical Path)

1. Frontend sends `POST /api/query` with `{query, session_id}`
2. `app.py:query_documents()` creates/retrieves session, calls `rag_system.query()`
3. `rag_system.py:query()`:
   - Gets conversation history from `SessionManager`
   - Calls `ai_generator.generate_response()` with tool definitions
   - Retrieves sources from `ToolManager`
   - Updates conversation history
4. `ai_generator.py` sends prompt to Claude with tools available
5. If Claude needs to search, it calls `CourseSearchTool`
6. `CourseSearchTool` queries `VectorStore` (semantic search on `course_content`, filters by course/lesson)
7. Search results returned to Claude, which synthesizes final answer
8. Response returned as `{answer, sources}`

### Document Processing Flow

1. Documents loaded from `/docs` folder on app startup (via `app.py` startup event)
2. `DocumentProcessor` parses document format:
   ```
   Course Title: [name]
   Course Link: [url]
   Course Instructor: [name]

   Lesson [number]: [title]
   Lesson Link: [url]
   [lesson content...]

   Lesson [number]: [title]
   ...
   ```
3. Text chunked with overlap (config: 800 chars chunk, 100 overlap)
4. Chunks stored in `course_content` collection with metadata
5. Course metadata stored in `course_catalog` collection for course resolution

### Dual Collection Pattern (Important)

The `VectorStore` maintains two ChromaDB collections:
- **`course_catalog`**: Stores course metadata (title, link, instructor) for semantic course name matching
- **`course_content`**: Stores actual text chunks with course/lesson context

This separation allows efficient course resolution before content search.

### Tool Architecture

Tools inherit from abstract `Tool` base class. `CourseSearchTool`:
- Receives query parameters from Claude (course name, lesson number, search query)
- Performs semantic search with filters
- Tracks `last_sources` for UI display
- Registered with `ToolManager` which passes tool definitions to Claude

### Configuration (`config.py`)

Key settings (dataclass defaults, override via env vars):
- `ANTHROPIC_MODEL`: `claude-sonnet-4-20250514`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2`
- `CHUNK_SIZE`: 800, `CHUNK_OVERLAP`: 100
- `MAX_RESULTS`: 5, `MAX_HISTORY`: 2
- `CHROMA_PATH`: `./chroma_db`

## Document Format Expectation

Documents in `/docs` must follow the structured format with `Course Title:`, `Course Link:`, `Course Instructor:`, and `Lesson [number]:` markers. The parser relies on this format for metadata extraction.

## Session Management

Sessions are **in-memory only** (no persistence). Session IDs are auto-generated UUIDs. Conversation history limited to `MAX_HISTORY` exchanges. For production, consider persistent storage.

## Extension Points

- **Add new tools**: Implement `Tool` base class, register with `ToolManager`
- **New document formats**: Extend `DocumentProcessor.process_course_document()`
- **Different vector stores**: Create new implementation, update `RAGSystem.__init__()`
- **Persistent sessions**: Replace `SessionManager` with database-backed implementation
