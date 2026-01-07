"""
Shared fixtures for RAG system tests.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, AsyncMock, patch
from typing import Generator, Any
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from fastapi.testclient import TestClient


@pytest.fixture
def sample_course():
    """Sample Course object for testing."""
    return Course(
        title="Machine Learning Basics",
        course_link="https://example.com/ml",
        instructor="Dr. Smith",
        lessons=[
            Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/l1"),
            Lesson(lesson_number=2, title="Linear Regression"),
        ],
    )


@pytest.fixture
def sample_chunks():
    """Sample CourseChunk objects for testing."""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            course_title="Machine Learning Basics",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Linear regression models the relationship between variables using a linear approach.",
            course_title="Machine Learning Basics",
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Neural networks are computing systems inspired by biological neural networks.",
            course_title="Deep Learning Advanced",
            lesson_number=1,
            chunk_index=0,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """
    Mock VectorStore with configurable behavior.
    Tests can set max_results attribute to test the bug (0) vs fixed (>0) behavior.
    """
    store = MagicMock()
    store.max_results = 0  # Default: simulates the bug

    def mock_search(query, course_name=None, lesson_number=None, limit=None):
        """Mock search that respects max_results setting."""
        # If max_results is 0, return empty results (bug behavior)
        if store.max_results == 0 and limit is None:
            return SearchResults.empty("No results (max_results=0)")

        # Determine result count based on limit or max_results
        result_count = limit if limit is not None else store.max_results

        if result_count <= 0:
            return SearchResults.empty("No results (max_results=0)")

        # Return sample results
        return SearchResults(
            documents=[
                "Machine learning is a subset of artificial intelligence.",
                "Linear regression models relationships between variables.",
            ][:result_count],
            metadata=[
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 1},
            ][:result_count],
            distances=[0.1, 0.2][:result_count],
        )

    store.search.side_effect = mock_search
    store._resolve_course_name.return_value = "Machine Learning Basics"
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    store.get_course_link.return_value = "https://example.com/course"

    return store


@pytest.fixture
def mock_vector_store_fixed():
    """Mock VectorStore with max_results=5 (fixed behavior)."""
    store = MagicMock()
    store.max_results = 5  # Fixed: returns results

    def mock_search(query, course_name=None, lesson_number=None, limit=None):
        """Mock search that returns results."""
        result_count = limit if limit is not None else store.max_results

        return SearchResults(
            documents=[
                "Machine learning is a subset of artificial intelligence.",
                "Linear regression models relationships between variables.",
            ][:result_count],
            metadata=[
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 1},
            ][:result_count],
            distances=[0.1, 0.2][:result_count],
        )

    store.search.side_effect = mock_search
    store._resolve_course_name.return_value = "Machine Learning Basics"
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    store.get_course_link.return_value = "https://example.com/course"

    return store


@pytest.fixture
def mock_ai_generator():
    """Mock AIGenerator for testing."""
    generator = MagicMock()

    # Mock successful connection test
    generator.test_connection.return_value = (True, "Connection successful")

    # Mock generate_response to return different results based on query
    def mock_generate(query, conversation_history=None, tools=None, tool_manager=None):
        if "course" in query.lower() or "lesson" in query.lower():
            # For course questions, simulate tool usage
            if tool_manager and tools:
                return "Based on the course materials, here's what I found..."
        return "Direct answer to general knowledge question."

    generator.generate_response.side_effect = mock_generate

    return generator


@pytest.fixture
def test_document(tmp_path, sample_course):
    """Create a test document file in proper format."""
    content = f"""Course Title: {sample_course.title}
Course Link: {sample_course.course_link}
Course Instructor: {sample_course.instructor}

Lesson 1: {sample_course.lessons[0].title}
Lesson Link: {sample_course.lessons[0].lesson_link}
This is the content of lesson 1.
Machine learning is a subset of artificial intelligence that focuses on algorithms.
It enables computers to learn from data and make predictions.

Lesson 2: {sample_course.lessons[1].title}
This lesson covers linear regression, a fundamental algorithm in machine learning.
Linear regression models the relationship between dependent and independent variables.
"""

    doc_file = tmp_path / "test_course.txt"
    doc_file.write_text(content)
    return str(doc_file)


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager for testing."""
    manager = MagicMock()
    manager.create_session.return_value = "test_session_123"
    manager.get_conversation_history.return_value = None
    return manager


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for API testing."""
    rag = MagicMock()

    # Mock query method
    rag.query.return_value = (
        "This is a test response based on the course materials.",
        [
            {
                "name": "Machine Learning Basics - Lesson 1",
                "link": "https://example.com/ml#lesson1"
            },
            {
                "name": "Machine Learning Basics - Lesson 2",
                "link": "https://example.com/ml#lesson2"
            }
        ]
    )

    # Mock session manager
    rag.session_manager = MagicMock()
    rag.session_manager.create_session.return_value = "new_session_456"
    rag.session_manager.get_conversation_history.return_value = None

    # Mock get_course_analytics
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Machine Learning Basics", "Deep Learning Advanced"]
    }

    # Mock AI generator for connection test
    rag.ai_generator = MagicMock()
    rag.ai_generator.test_connection.return_value = (True, "Connection successful")

    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.

    This fixture creates a minimal FastAPI app with the same endpoints
    as the production app but without the static file mount, which causes
    issues in test environments where the frontend directory doesn't exist.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class NewSessionResponse(BaseModel):
        session_id: str

    # Create test app
    app = FastAPI(title="Course Materials RAG System (Test)")

    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Store mock rag_system in app state for access in endpoints
    app.state.rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources."""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics."""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/new", response_model=NewSessionResponse)
    async def create_new_session():
        """Create a new conversation session."""
        try:
            session_id = mock_rag_system.session_manager.create_session()
            return NewSessionResponse(session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check."""
        return {"status": "healthy", "message": "RAG System API is running"}

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)
