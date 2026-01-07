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
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

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
        tool_use_block.input = {"query": "machine learning", "course_name": "ML Course"}

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
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = (
            "[ML Course - Lesson 1]\n" "Machine learning is a subset of artificial intelligence..."
        )

        response = generator.generate_response(
            "Tell me about machine learning in the ML Course",
            tools=[
                {
                    "name": "search_course_content",
                    "description": "Search course materials",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_manager=mock_tool_manager,
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
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager
        mock_tool_manager = MagicMock()

        response = generator.generate_response(
            "What is the capital of France?",
            tools=[{"name": "search_course_content", "description": "Search course materials"}],
            tool_manager=mock_tool_manager,
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
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        history = "User: First question\nAssistant: First answer"
        response = generator.generate_response("Follow-up question", conversation_history=history)

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
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager to return empty results
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "No relevant content found."

        response = generator.generate_response(
            "Tell me about nonexistent topic",
            tools=[{"name": "search_course_content", "description": "Search course materials"}],
            tool_manager=mock_tool_manager,
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
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

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

    def test_ai_sequential_tool_calling_two_rounds(self):
        """
        Test that Claude can make 2 sequential tool calls.
        Example flow:
        1. User asks a question requiring two searches
        2. Claude calls first tool (e.g., get_course_outline)
        3. Claude calls second tool (e.g., search_course_content)
        4. Claude provides final answer
        """
        # Mock Anthropic client
        mock_client = MagicMock()

        # Mock first response with tool use (get_course_outline)
        first_response = Mock()
        first_response.stop_reason = "tool_use"

        tool_use_block_1 = Mock()
        tool_use_block_1.type = "tool_use"
        tool_use_block_1.id = "tool_123"
        tool_use_block_1.name = "get_course_outline"
        tool_use_block_1.input = {"course_name": "Python Course"}

        first_response.content = [tool_use_block_1]

        # Mock second response with another tool use (search_course_content)
        second_response = Mock()
        second_response.stop_reason = "tool_use"

        tool_use_block_2 = Mock()
        tool_use_block_2.type = "tool_use"
        tool_use_block_2.id = "tool_456"
        tool_use_block_2.name = "search_course_content"
        tool_use_block_2.input = {"query": "decorators"}

        second_response.content = [tool_use_block_2]

        # Mock final response
        final_response = Mock()
        final_content = Mock()
        final_content.text = "Lesson 4 covers decorators and generators..."
        final_response.content = [final_content]

        # Set up create to return different responses on sequential calls
        mock_client.messages.create.side_effect = [first_response, second_response, final_response]

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "get_course_outline", "description": "Get course outline"}
        ]
        mock_tool_manager.execute_tool.side_effect = [
            "Lesson 4: Decorators and Generators",
            "[Content about decorators...]",
        ]

        response = generator.generate_response(
            "What does lesson 4 of Python Course cover?",
            tools=[{"name": "get_course_outline", "description": "Get course outline"}],
            tool_manager=mock_tool_manager,
        )

        # Both tools were called
        assert mock_tool_manager.execute_tool.call_count == 2
        assert response == "Lesson 4 covers decorators and generators..."

    def test_ai_tool_calling_stops_at_max_rounds(self):
        """
        Test that even if Claude wants to make a 3rd tool call,
        it must provide a final answer after 2 rounds.
        """
        # Mock Anthropic client
        mock_client = MagicMock()

        # Round 1: First tool call
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        tool_use_block_1 = Mock()
        tool_use_block_1.type = "tool_use"
        tool_use_block_1.id = "tool_1"
        tool_use_block_1.name = "search_course_content"
        tool_use_block_1.input = {"query": "topic1"}
        first_response.content = [tool_use_block_1]

        # Round 2: Second tool call
        second_response = Mock()
        second_response.stop_reason = "tool_use"
        tool_use_block_2 = Mock()
        tool_use_block_2.type = "tool_use"
        tool_use_block_2.id = "tool_2"
        tool_use_block_2.name = "search_course_content"
        tool_use_block_2.input = {"query": "topic2"}
        second_response.content = [tool_use_block_2]

        # Final response (forced completion without tools)
        final_response = Mock()
        final_content = Mock()
        final_content.text = "Based on the search results..."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [first_response, second_response, final_response]

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search course materials"}
        ]
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        response = generator.generate_response(
            "Multi-part question",
            tools=[{"name": "search_course_content", "description": "Search course materials"}],
            tool_manager=mock_tool_manager,
        )

        # Exactly 2 tools were called (max_rounds=2)
        assert mock_tool_manager.execute_tool.call_count == 2
        # Final API call was made WITHOUT tools (forcing completion)
        assert mock_client.messages.create.call_count == 3
        assert "Based on the search results" in response

    def test_ai_early_termination_no_tools(self):
        """
        Test that if Claude doesn't call tools in round 1,
        it returns immediately without starting round 2.
        """
        # Mock Anthropic client - Claude responds directly without tools
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
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager (should not be called)
        mock_tool_manager = MagicMock()

        response = generator.generate_response(
            "What is the capital of France?",
            tools=[{"name": "search_course_content", "description": "Search course materials"}],
            tool_manager=mock_tool_manager,
        )

        # Tool was NOT called for general knowledge
        mock_tool_manager.execute_tool.assert_not_called()
        # Only 1 API call was made
        assert mock_client.messages.create.call_count == 1
        assert "Paris" in response

    def test_ai_tool_execution_error_stops_loop(self):
        """
        Test that if a tool fails during execution,
        the loop stops and returns an error message.
        """
        # Mock Anthropic client
        mock_client = MagicMock()

        # Mock response with tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"

        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tool_123"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "test"}

        tool_response.content = [tool_use_block]
        mock_client.messages.create.return_value = tool_response

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager that raises an exception
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search course materials"}
        ]
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        response = generator.generate_response(
            "Test question",
            tools=[{"name": "search_course_content", "description": "Search course materials"}],
            tool_manager=mock_tool_manager,
        )

        # Should return error message
        assert "error" in response.lower() or "encountered an error" in response.lower()

    def test_ai_multiple_tools_single_round(self):
        """
        Test that Claude can call multiple tools in a single response.
        All tools execute, then next round begins.
        """
        # Mock Anthropic client
        mock_client = MagicMock()

        # First response with TWO tool calls
        first_response = Mock()
        first_response.stop_reason = "tool_use"

        tool_use_block_1 = Mock()
        tool_use_block_1.type = "tool_use"
        tool_use_block_1.id = "tool_1"
        tool_use_block_1.name = "get_course_outline"
        tool_use_block_1.input = {"course_name": "Python Course"}

        tool_use_block_2 = Mock()
        tool_use_block_2.type = "tool_use"
        tool_use_block_2.id = "tool_2"
        tool_use_block_2.name = "get_course_outline"
        tool_use_block_2.input = {"course_name": "Java Course"}

        first_response.content = [tool_use_block_1, tool_use_block_2]

        # Final response
        final_response = Mock()
        final_content = Mock()
        final_content.text = "Here are the outlines..."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [first_response, final_response]

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "get_course_outline", "description": "Get course outline"}
        ]
        mock_tool_manager.execute_tool.side_effect = [
            "Python Course Outline...",
            "Java Course Outline...",
        ]

        response = generator.generate_response(
            "Show me outlines for Python and Java courses",
            tools=[{"name": "get_course_outline", "description": "Get course outline"}],
            tool_manager=mock_tool_manager,
        )

        # Both tools were executed in same round
        assert mock_tool_manager.execute_tool.call_count == 2
        assert response == "Here are the outlines..."

    def test_ai_mixed_content_blocks_filters_tool_use(self):
        """
        Test that _extract_text_from_response properly filters out tool_use blocks
        and only returns text from text blocks. This prevents tool call details
        from being returned to the user.
        """
        # Mock Anthropic client
        mock_client = MagicMock()

        # Create a response with mixed content: text block + tool_use block
        # This simulates a scenario where Claude includes both text and tool calls
        mixed_response = Mock()
        mixed_response.stop_reason = "tool_use"

        # Text block with some commentary
        text_block = Mock()
        text_block.text = "I'll search for that information."

        # Tool use block (should be filtered out) - simulate real API where
        # tool_use blocks don't have a .text attribute
        tool_use_block = Mock(spec=["type", "id", "name", "input"])
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tool_123"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "test", "course_name": "Test Course"}

        # Response contains both text and tool_use blocks
        mixed_response.content = [text_block, tool_use_block]

        # Create generator with mocked client
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_client
        generator.model = "test-model"
        generator.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}

        # Test the _extract_text_from_response method
        extracted_text = generator._extract_text_from_response(mixed_response)

        # Should only return the text from the text block, not the tool_use block
        assert extracted_text == "I'll search for that information."
        # Should NOT contain tool_use block content
        assert "search_course_content" not in extracted_text

    def test_ai_response_with_only_tool_use_blocks(self):
        """
        Test that _extract_text_from_response handles responses with only tool_use blocks
        (no text blocks) by returning an empty string rather than crashing.
        """
        # Mock response with only tool_use blocks (no text)
        tool_only_response = Mock()

        # Tool use block - use spec to simulate real API where tool_use blocks
        # don't have a .text attribute
        tool_use_block = Mock(spec=["type", "id", "name", "input"])
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tool_123"
        tool_use_block.name = "search_course_content"

        tool_only_response.content = [tool_use_block]

        # Create generator
        generator = AIGenerator.__new__(AIGenerator)
        generator.model = "test-model"

        # Should return empty string rather than crashing
        extracted_text = generator._extract_text_from_response(tool_only_response)
        assert extracted_text == ""
