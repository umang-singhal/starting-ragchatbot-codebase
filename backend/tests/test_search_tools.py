"""
Tests for search_tools.py - CourseSearchTool and CourseOutlineTool.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


@pytest.mark.unit
class TestCourseSearchTool:
    """Tests for CourseSearchTool.execute() method."""

    def test_course_search_tool_with_query_only_buggy(self, mock_vector_store):
        """
        Test search with just query parameter when max_results=0 (BUG).
        This should return empty results due to the bug.
        """
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="machine learning")

        # With max_results=0, we get error from vector store
        assert "No results" in result or "No relevant content" in result
        assert len(tool.last_sources) == 0

    def test_course_search_tool_with_query_only_fixed(self, mock_vector_store_fixed):
        """
        Test search with just query parameter when max_results>0 (FIXED).
        This should return actual results.
        """
        tool = CourseSearchTool(mock_vector_store_fixed)
        result = tool.execute(query="machine learning")

        # With max_results>0, we get results
        assert "Machine Learning Basics" in result
        assert len(tool.last_sources) > 0

    def test_course_search_tool_with_query_and_course_fixed(self, mock_vector_store_fixed):
        """Test search with query and course name (fixed behavior)."""
        tool = CourseSearchTool(mock_vector_store_fixed)

        result = tool.execute(
            query="neural networks",
            course_name="Machine Learning"
        )

        assert "Machine Learning Basics" in result
        assert len(tool.last_sources) > 0
        # Source name includes lesson number when lesson is specified
        assert "Machine Learning Basics" in tool.last_sources[0]["name"]

    def test_course_search_tool_with_query_and_lesson_fixed(self, mock_vector_store_fixed):
        """Test search with query and lesson number (fixed behavior)."""
        tool = CourseSearchTool(mock_vector_store_fixed)

        result = tool.execute(
            query="algorithms",
            course_name="Machine Learning Basics",
            lesson_number=1
        )

        assert "Lesson 1" in result
        assert "Machine Learning Basics" in result

    def test_course_search_tool_empty_results(self, mock_vector_store):
        """Test search when no results found (bug behavior)."""
        # max_results=0 causes empty results with error message
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        # Should contain error message from vector store
        assert "No results" in result or "No relevant content" in result
        assert len(tool.last_sources) == 0

    def test_course_search_tool_error_handling(self):
        """Test search when vector store returns error."""
        # Create a mock that returns an error
        error_store = MagicMock()
        error_store.search.return_value = SearchResults.empty(
            error_msg="Search error: Database connection failed"
        )

        tool = CourseSearchTool(error_store)
        result = tool.execute(query="test")

        assert "Search error" in result

    def test_course_search_tool_sources_tracking(self, mock_vector_store_fixed):
        """Test that sources are properly tracked with links."""
        tool = CourseSearchTool(mock_vector_store_fixed)

        tool.execute(query="test", course_name="Machine Learning Basics", lesson_number=1)

        sources = tool.last_sources
        assert len(sources) > 0
        assert "name" in sources[0]
        assert "link" in sources[0]
        assert sources[0]["name"] == "Machine Learning Basics - Lesson 1"
        assert sources[0]["link"] == "https://example.com/lesson1"

    def test_course_search_tool_get_tool_definition(self):
        """Test that tool definition is properly structured."""
        tool = CourseSearchTool(MagicMock())
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]


@pytest.mark.unit
class TestCourseOutlineTool:
    """Tests for CourseOutlineTool.execute() method."""

    def test_course_outline_tool_success(self, mock_vector_store_fixed):
        """Test course outline retrieval."""
        # Set up mock for course catalog get
        mock_catalog = MagicMock()
        mock_catalog.get.return_value = {
            'metadatas': [{
                'title': 'Machine Learning Basics',
                'instructor': 'Dr. Smith',
                'course_link': 'https://example.com/ml',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/l1"}]'
            }]
        }
        mock_vector_store_fixed.course_catalog = mock_catalog

        tool = CourseOutlineTool(mock_vector_store_fixed)
        result = tool.execute(course_title="Machine Learning")

        assert "Course:" in result
        assert "Lessons:" in result
        assert len(tool.last_sources) > 0

    def test_course_outline_tool_not_found(self, mock_vector_store):
        """Test outline with non-existent course."""
        mock_vector_store._resolve_course_name.return_value = None

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="NonExistent")

        assert "No course found matching" in result

    def test_course_outline_tool_get_tool_definition(self):
        """Test that tool definition is properly structured."""
        tool = CourseOutlineTool(MagicMock())
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "course_title" in definition["input_schema"]["properties"]


@pytest.mark.unit
class TestToolManager:
    """Tests for ToolManager."""

    def test_tool_manager_registration(self):
        """Test tool registration and execution."""
        manager = ToolManager()

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool"
        }
        mock_tool.execute.return_value = "Tool executed"

        manager.register_tool(mock_tool)

        assert "test_tool" in manager.tools
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"

    def test_tool_manager_execute_tool(self):
        """Test executing tool through manager."""
        manager = ToolManager()

        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        mock_tool.execute.return_value = "Tool executed"

        manager.register_tool(mock_tool)
        result = manager.execute_tool("test_tool", param1="value")

        assert result == "Tool executed"
        mock_tool.execute.assert_called_once_with(param1="value")

    def test_tool_manager_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist."""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool")

        assert "not found" in result.lower()

    def test_tool_manager_get_last_sources(self, mock_vector_store_fixed):
        """Test retrieving sources from tools."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store_fixed)

        # Set sources on the tool
        search_tool.last_sources = [{"name": "Test", "link": "http://test.com"}]

        manager.register_tool(search_tool)
        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["name"] == "Test"

    def test_tool_manager_reset_sources(self, mock_vector_store_fixed):
        """Test resetting sources from all tools."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store_fixed)
        search_tool.last_sources = [{"name": "Test", "link": "http://test.com"}]

        manager.register_tool(search_tool)
        manager.reset_sources()

        assert len(search_tool.last_sources) == 0
        assert len(manager.get_last_sources()) == 0
