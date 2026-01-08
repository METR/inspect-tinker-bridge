"""Tests for utils module."""

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

from inspect_tinker_bridge import utils


class TestExtractContent:
    """Tests for _extract_content function."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            pytest.param("Hello, world!", "Hello, world!", id="string_content"),
            pytest.param("", "", id="empty_string"),
            pytest.param("  spaces  ", "  spaces  ", id="preserves_whitespace"),
        ],
    )
    def test_string_content(self, content: str, expected: str) -> None:
        """Test extraction of string content."""
        msg = ChatMessageUser(content=content)
        assert utils._extract_content(msg) == expected


class TestChatMessageToDict:
    """Tests for chat_message_to_dict function."""

    def test_user_message(self) -> None:
        """Test conversion of user message."""
        msg = ChatMessageUser(content="What is 2+2?")
        result = utils.chat_message_to_dict(msg)

        assert result["role"] == "user"
        assert result["content"] == "What is 2+2?"
        assert "tool_calls" not in result

    def test_system_message(self) -> None:
        """Test conversion of system message."""
        msg = ChatMessageSystem(content="You are a helpful assistant.")
        result = utils.chat_message_to_dict(msg)

        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_assistant_message_without_tools(self) -> None:
        """Test conversion of assistant message without tool calls."""
        msg = ChatMessageAssistant(content="The answer is 4.")
        result = utils.chat_message_to_dict(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "The answer is 4."
        assert "tool_calls" not in result

    def test_assistant_message_with_tool_calls(self) -> None:
        """Test conversion of assistant message with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            function="bash",
            arguments={"command": "echo hello"},
            type="function",
        )
        msg = ChatMessageAssistant(content="Running command.", tool_calls=[tool_call])
        result = utils.chat_message_to_dict(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Running command."
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["function"]["name"] == "bash"
        # arguments converted to str representation
        assert tc["function"]["arguments"] == "{'command': 'echo hello'}"

    def test_tool_message(self) -> None:
        """Test conversion of tool response message."""
        msg = ChatMessageTool(
            content="hello\n",
            tool_call_id="call_123",
            function="bash",
        )
        result = utils.chat_message_to_dict(msg)

        assert result["role"] == "tool"
        assert result["content"] == "hello\n"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "bash"


class TestChatMessageToTinker:
    """Tests for chat_message_to_tinker function."""

    def test_user_message(self) -> None:
        """Test conversion of user message to Tinker format."""
        msg = ChatMessageUser(content="Hello")
        result = utils.chat_message_to_tinker(msg)

        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_tool_message_preserves_metadata(self) -> None:
        """Test that tool message preserves tool_call_id and name."""
        msg = ChatMessageTool(
            content="output",
            tool_call_id="abc",
            function="test_func",
        )
        result = utils.chat_message_to_tinker(msg)

        assert result["role"] == "tool"
        assert result.get("tool_call_id") == "abc"
        assert result.get("name") == "test_func"


class TestChatMessagesToTinker:
    """Tests for chat_messages_to_tinker function."""

    def test_empty_list(self) -> None:
        """Test conversion of empty message list."""
        assert utils.chat_messages_to_tinker([]) == []

    def test_multiple_messages(self) -> None:
        """Test conversion of multiple messages preserves order."""
        messages = [
            ChatMessageSystem(content="System prompt"),
            ChatMessageUser(content="User query"),
            ChatMessageAssistant(content="Assistant response"),
        ]
        # List type is invariant but this is safe since we're passing subtypes
        result = utils.chat_messages_to_tinker(messages)  # pyright: ignore[reportArgumentType]

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
