"""
Tests for rag_system.py - RAG System integration tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system import RAGSystem


@pytest.mark.integration
class TestRAGSystem:
    """Tests for RAGSystem query flow."""

    def test_rag_query_with_content_question_buggy(
        self, mock_vector_store, mock_ai_generator, mock_session_manager
    ):
        """
        Test full query flow for course content question with BUG (max_results=0).
        This should demonstrate the bug - empty sources returned.
        """
        # Create a minimal config
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 0  # BUG: This causes empty results

        # Create RAG system with mocked components
        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.AIGenerator", return_value=mock_ai_generator),
            patch("rag_system.SessionManager", return_value=mock_session_manager),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.vector_store = mock_vector_store
            rag_system.ai_generator = mock_ai_generator
            rag_system.session_manager = mock_session_manager

            # Mock the query to simulate empty results due to bug
            mock_ai_generator.generate_response.return_value = (
                "No relevant content found in the course materials."
            )

            response, sources = rag_system.query(
                "What is covered in Lesson 1?", session_id="test_session"
            )

            # With bug, sources should be empty
            assert response is not None
            # Empty sources due to max_results=0 bug
            assert len(sources) == 0, "Bug: max_results=0 returns empty sources"

    def test_rag_query_with_content_question_fixed(
        self, mock_vector_store_fixed, mock_ai_generator, mock_session_manager
    ):
        """
        Test full query flow for course content question with FIX (max_results>0).
        This should return actual results with sources.
        """
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5  # FIXED: Returns results

        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store_fixed),
            patch("rag_system.AIGenerator", return_value=mock_ai_generator),
            patch("rag_system.SessionManager", return_value=mock_session_manager),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.vector_store = mock_vector_store_fixed
            rag_system.ai_generator = mock_ai_generator
            rag_system.session_manager = mock_session_manager

            # Mock the query to return results
            mock_ai_generator.generate_response.return_value = (
                "Based on the course materials, Lesson 1 covers machine learning fundamentals."
            )

            # Mock tool manager to return sources
            mock_tool_manager = MagicMock()
            mock_tool_manager.get_last_sources.return_value = [
                {"name": "Machine Learning Basics - Lesson 1", "link": "https://example.com/l1"}
            ]
            rag_system.tool_manager = mock_tool_manager

            response, sources = rag_system.query(
                "What is covered in Lesson 1?", session_id="test_session"
            )

            assert response is not None
            # With fix, sources should be populated
            assert len(sources) > 0, "Fixed: max_results>0 returns sources"
            assert sources[0]["name"] == "Machine Learning Basics - Lesson 1"

    def test_rag_query_with_general_question(
        self, mock_vector_store, mock_ai_generator, mock_session_manager
    ):
        """Test query flow for general knowledge question (no tool needed)."""
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5

        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.AIGenerator", return_value=mock_ai_generator),
            patch("rag_system.SessionManager", return_value=mock_session_manager),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.vector_store = mock_vector_store
            rag_system.ai_generator = mock_ai_generator
            rag_system.session_manager = mock_session_manager

            # Mock the query to return direct answer (no tool used)
            mock_ai_generator.generate_response.return_value = "Paris is the capital of France."

            # Mock tool manager to return no sources (general knowledge)
            mock_tool_manager = MagicMock()
            mock_tool_manager.get_last_sources.return_value = []
            rag_system.tool_manager = mock_tool_manager

            response, sources = rag_system.query(
                "What is the capital of France?", session_id="test_session"
            )

            assert response is not None
            # Sources should be empty for general knowledge
            assert len(sources) == 0

    def test_rag_query_creates_session_if_not_provided(
        self, mock_vector_store, mock_ai_generator, mock_session_manager
    ):
        """Test that query creates session when session_id is None."""
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5

        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.AIGenerator", return_value=mock_ai_generator),
            patch("rag_system.SessionManager", return_value=mock_session_manager),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.vector_store = mock_vector_store
            rag_system.ai_generator = mock_ai_generator
            rag_system.session_manager = mock_session_manager

            mock_ai_generator.generate_response.return_value = "Response"
            mock_tool_manager = MagicMock()
            mock_tool_manager.get_last_sources.return_value = []
            rag_system.tool_manager = mock_tool_manager

            # Query without session_id
            response, sources = rag_system.query("Test query")

            assert response is not None

    def test_rag_query_updates_conversation_history(
        self, mock_vector_store, mock_ai_generator, mock_session_manager
    ):
        """Test that exchanges are added to session history."""
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5

        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.AIGenerator", return_value=mock_ai_generator),
            patch("rag_system.SessionManager", return_value=mock_session_manager),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.vector_store = mock_vector_store
            rag_system.ai_generator = mock_ai_generator
            rag_system.session_manager = mock_session_manager

            # Use a general knowledge question to avoid triggering tool use side_effect
            # This ensures we get a direct response without tool calling
            mock_ai_generator.generate_response.reset_mock()
            mock_ai_generator.generate_response.return_value = "AI response to question"
            mock_ai_generator.generate_response.side_effect = None  # Clear side_effect

            mock_tool_manager = MagicMock()
            mock_tool_manager.get_last_sources.return_value = []
            rag_system.tool_manager = mock_tool_manager

            session_id = "test_session"

            # First query - use general knowledge query
            rag_system.query("What is a general question?", session_id)

            # Check that add_exchange was called
            mock_session_manager.add_exchange.assert_called_once()
            # Verify the arguments contain the expected values
            call_args = mock_session_manager.add_exchange.call_args
            assert call_args[0][0] == session_id
            assert "general question" in call_args[0][1]
            assert call_args[0][2] == "AI response to question"

    def test_rag_get_course_analytics(
        self, mock_vector_store, mock_ai_generator, mock_session_manager
    ):
        """Test getting course statistics."""
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5

        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.AIGenerator", return_value=mock_ai_generator),
            patch("rag_system.SessionManager", return_value=mock_session_manager),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.vector_store = mock_vector_store
            rag_system.ai_generator = mock_ai_generator
            rag_system.session_manager = mock_session_manager

            # Mock vector store methods
            mock_vector_store.get_course_count.return_value = 5
            mock_vector_store.get_existing_course_titles.return_value = [
                "Course 1",
                "Course 2",
                "Course 3",
                "Course 4",
                "Course 5",
            ]

            analytics = rag_system.get_course_analytics()

            assert analytics["total_courses"] == 5
            assert len(analytics["course_titles"]) == 5

    def test_rag_initialization(self):
        """Test RAG system initialization with all components."""
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test_key"
        config.ANTHROPIC_MODEL = "test-model"

        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.ToolManager") as mock_tm,
            patch("rag_system.CourseSearchTool") as mock_st,
            patch("rag_system.CourseOutlineTool") as mock_ot,
        ):

            rag_system = RAGSystem(config)

            # Verify components were initialized
            assert rag_system.config == config
            assert rag_system.tool_manager is not None

            # Verify tools were registered
            assert mock_tm.return_value.register_tool.call_count == 2
