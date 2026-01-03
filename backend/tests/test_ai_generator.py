"""
Tests for ai_generator.py - AI response generation and tool calling.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_generator import AIGenerator


@pytest.mark.unit
class TestAIGenerator:
    """Tests for AIGenerator class."""

    def test_ai_generator_without_tools(self):
        """Test response generation without tools."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Direct answer to question"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }

        response = generator.generate_response("What is Python?")

        assert response == "Direct answer to question"
        assert mock_client.messages.create.called

    def test_ai_generator_with_tool_call(self):
        """
        Test that AI calls search tool for content questions.
        This simulates the tool calling flow.
        """
        # Mock Anthropic client
        mock_client = MagicMock()

        # Mock initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"

        # Create tool_use block
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tool_123"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {
            "query": "machine learning",
            "course_name": "ML Course"
        }

        initial_response.content = [tool_use_block]

        # Mock final response after tool execution
        final_response = Mock()
        final_content = Mock()
        final_content.text = "Machine learning is a subset of AI..."
        final_response.content = [final_content]

        # Set up create to return different responses on sequential calls
        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = (
            "[ML Course - Lesson 1]\n"
            "Machine learning is a subset of artificial intelligence..."
        )

        response = generator.generate_response(
            "Tell me about machine learning in the ML Course",
            tools=[{
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object"}
            }],
            tool_manager=mock_tool_manager
        )

        # Assert tool was called with correct parameters
        mock_tool_manager.execute_tool.assert_called_once()
        call_args = mock_tool_manager.execute_tool.call_args
        assert call_args[0][0] == "search_course_content"
        assert call_args[1]["query"] == "machine learning"
        assert call_args[1]["course_name"] == "ML Course"

        assert response == "Machine learning is a subset of AI..."

    def test_ai_general_knowledge_no_tool_call(self):
        """
        Test that general knowledge questions don't trigger tools.
        The AI should answer directly without using search tools.
        """
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Paris is the capital of France"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }

        # Mock tool manager
        mock_tool_manager = MagicMock()

        response = generator.generate_response(
            "What is the capital of France?",
            tools=[{
                "name": "search_course_content",
                "description": "Search course materials"
            }],
            tool_manager=mock_tool_manager
        )

        # Tool should NOT be called for general knowledge
        mock_tool_manager.execute_tool.assert_not_called()
        assert "Paris" in response

    def test_ai_conversation_history_included(self):
        """Test that conversation history is included in context."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Follow-up answer"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }

        history = "User: First question\nAssistant: First answer"
        response = generator.generate_response(
            "Follow-up question",
            conversation_history=history
        )

        # Check that history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_prompt = call_args.kwargs["system"]
        assert "Previous conversation:" in system_prompt
        assert "First question" in system_prompt
        assert response == "Follow-up answer"

    def test_ai_connection_success(self):
        """Test API connection check."""
        # Mock successful connection
        mock_client = MagicMock()
        mock_response = Mock()
        mock_client.messages.create.return_value = mock_response

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"

        success, message = generator.test_connection()

        assert success is True
        assert "successful" in message.lower()

    def test_ai_connection_failure(self):
        """Test API connection failure handling."""
        # Mock failed connection
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"

        success, message = generator.test_connection()

        assert success is False
        assert "failed" in message.lower()

    def test_ai_tool_execution_with_empty_results(self):
        """Test AI behavior when search tool returns empty results."""
        # Mock Anthropic client
        mock_client = MagicMock()

        # Mock initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"

        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tool_123"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "nonexistent topic"}

        initial_response.content = [tool_use_block]

        # Mock final response (AI should handle empty results gracefully)
        final_response = Mock()
        final_content = Mock()
        final_content.text = "No relevant content found in the course materials."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }

        # Mock tool manager to return empty results
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "No relevant content found."

        response = generator.generate_response(
            "Tell me about nonexistent topic",
            tools=[{
                "name": "search_course_content",
                "description": "Search course materials"
            }],
            tool_manager=mock_tool_manager
        )

        # Tool was called but returned empty results
        mock_tool_manager.execute_tool.assert_called_once()
        assert "No relevant content found" in response or "course materials" in response

    def test_ai_system_prompt_structure(self):
        """Test that system prompt is properly structured."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }

        generator.generate_response("Test question")

        # Check that API was called with proper structure
        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert "messages" in call_args.kwargs
        assert "model" in call_args.kwargs

        # Check system prompt contains expected content
        system_prompt = call_args.kwargs["system"]
        assert "AI assistant" in system_prompt
        assert "course materials" in system_prompt
