import anthropic
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class _ToolCallingState:
    """Internal state for sequential tool calling loop"""

    messages: List[Dict[str, Any]]
    current_round: int
    max_rounds: int
    last_response: Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool for questions about specific course content or detailed educational materials
- You may make up to 2 sequential tool calls per query if needed
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Course Outline Tool Usage:
- Use the course outline tool for questions about course structure, topics covered, lesson lists, or syllabus information
- Examples: "what topics are covered in X", "show me the outline of Y", "what lessons are in this course"
- You may combine tools: first get outline, then search specific lessons

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def test_connection(self) -> tuple[bool, str]:
        """
        Test the connection to the LLM API.

        Returns:
            tuple[bool, str]: (success, message)
        """
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=10, messages=[{"role": "user", "content": "test"}]
            )
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def _extract_text_from_response(self, response) -> str:
        """
        Extract text content from a Claude API response, handling mixed content blocks.

        When Claude uses tools or responds with multiple content blocks, the response
        may contain a mix of text and tool_use blocks. This method extracts and
        concatenates only the text blocks, filtering out tool_use blocks.

        Args:
            response: The Claude API response object

        Returns:
            Concatenated text from all text blocks in the response
        """
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                # This is a text block
                text_parts.append(block.text)
            # tool_use blocks are ignored - they don't contain user-facing text

        if not text_parts:
            # Fallback: if no text blocks found, this might be an edge case
            # Return empty string rather than crashing
            return ""

        return "".join(text_parts)

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response (extract text, filtering out tool_use blocks)
        return self._extract_text_from_response(response)

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager, max_rounds: int = 2
    ) -> str:
        """
        Handle execution of tool calls with support for sequential tool calling.

        Args:
            initial_response: The first response containing tool use requests
            base_params: Base API parameters from the initial request
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Final response text after all tool execution rounds complete
        """
        # Initialize state
        state = self._initialize_tool_calling_state(initial_response, base_params, max_rounds)

        # Main loop for sequential tool calling
        while state.current_round < state.max_rounds:
            # Check if we have more tools to execute
            if not self._has_tool_use_blocks(state.last_response):
                return self._extract_text_from_response(state.last_response)

            # Execute tools for this round
            execution_result = self._execute_tools_for_round(state, tool_manager)

            # Handle tool execution failure
            if execution_result["error"]:
                return self._handle_tool_error(execution_result["error"])

            # Update messages with tool results
            state.messages.extend(execution_result["tool_messages"])

            # Prepare for next round
            state.current_round += 1

            # Check if we should make final call (no more tools allowed)
            if state.current_round >= state.max_rounds:
                return self._make_final_api_call(state, base_params)

            # Make intermediate API call (tools still available)
            state.last_response = self._make_intermediate_api_call(state, base_params, tool_manager)

        # Should not reach here, but handle edge case
        return self._extract_text_from_response(state.last_response)

    def _initialize_tool_calling_state(
        self, initial_response, base_params: Dict[str, Any], max_rounds: int
    ) -> _ToolCallingState:
        """Initialize the state for sequential tool calling loop"""
        messages = base_params["messages"].copy()
        return _ToolCallingState(
            messages=messages,
            current_round=0,
            max_rounds=max_rounds,
            last_response=initial_response,
        )

    def _has_tool_use_blocks(self, response) -> bool:
        """Check if response contains any tool_use content blocks"""
        return any(block.type == "tool_use" for block in response.content)

    def _execute_tools_for_round(self, state: _ToolCallingState, tool_manager) -> Dict[str, Any]:
        """
        Execute all tools from the current response.

        Returns:
            Dict with keys:
            - 'error': Optional error message if tool execution failed
            - 'tool_messages': List of messages to add to conversation (assistant + user)
        """
        tool_results = []

        # Add assistant's tool_use response
        assistant_message = {"role": "assistant", "content": state.last_response.content}
        tool_messages = [assistant_message]

        # Execute each tool
        for content_block in state.last_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    return {"error": f"Tool execution failed: {str(e)}", "tool_messages": []}

        # Add tool results as user message
        if tool_results:
            tool_messages.append({"role": "user", "content": tool_results})

        return {"error": None, "tool_messages": tool_messages}

    def _handle_tool_error(self, error_message: str) -> str:
        """Generate a user-friendly error response when tool execution fails"""
        return f"I encountered an error while searching: {error_message}"

    def _make_intermediate_api_call(
        self, state: _ToolCallingState, base_params: Dict[str, Any], tool_manager
    ):
        """
        Make an intermediate API call with tools still available.
        This allows Claude to continue reasoning and potentially call more tools.
        """
        api_params = {
            **self.base_params,
            "messages": state.messages,
            "system": base_params["system"],
            "tools": tool_manager.get_tool_definitions(),
            "tool_choice": {"type": "auto"},
        }

        return self.client.messages.create(**api_params)

    def _make_final_api_call(self, state: _ToolCallingState, base_params: Dict[str, Any]) -> str:
        """
        Make the final API call without tools to force completion.
        This is called when max_rounds is reached.
        """
        api_params = {
            **self.base_params,
            "messages": state.messages,
            "system": base_params["system"],
        }

        final_response = self.client.messages.create(**api_params)
        return self._extract_text_from_response(final_response)
