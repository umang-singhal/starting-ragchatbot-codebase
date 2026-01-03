"""
Shared fixtures for RAG system tests.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


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
        ]
    )


@pytest.fixture
def sample_chunks():
    """Sample CourseChunk objects for testing."""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            course_title="Machine Learning Basics",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Linear regression models the relationship between variables using a linear approach.",
            course_title="Machine Learning Basics",
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are computing systems inspired by biological neural networks.",
            course_title="Deep Learning Advanced",
            lesson_number=1,
            chunk_index=0
        )
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
                "Linear regression models relationships between variables."
            ][:result_count],
            metadata=[
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 1}
            ][:result_count],
            distances=[0.1, 0.2][:result_count]
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
                "Linear regression models relationships between variables."
            ][:result_count],
            metadata=[
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Machine Learning Basics", "lesson_number": 1, "chunk_index": 1}
            ][:result_count],
            distances=[0.1, 0.2][:result_count]
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
