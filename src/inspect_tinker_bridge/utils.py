"""
Utility functions and constants for the Inspect-Tinker bridge.
"""

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelName,
)
from tinker_cookbook.renderers import Message

from inspect_tinker_bridge.types import MessageDict, ToolCallDict

# Dummy model name for TaskState construction
BRIDGE_MODEL_NAME = ModelName("bridge/bridge-model")


def _extract_content(msg: ChatMessage) -> str:
    """Extract string content from a ChatMessage."""
    content = msg.content
    if isinstance(content, str):
        return content
    # Content is a list of content parts - extract text
    text_parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(str(text))
    return "\n".join(text_parts) if text_parts else ""


def chat_message_to_tinker(msg: ChatMessage) -> Message:
    """
    Convert an Inspect ChatMessage to a Tinker Message dict.

    Preserves:
    - role: user, assistant, system, tool
    - content: text content
    - tool_calls: for assistant messages with tool use
    - tool_call_id: for tool response messages
    - name: tool function name for tool responses
    """
    match msg:
        case ChatMessageUser() | ChatMessageSystem():
            return Message(role=msg.role, content=_extract_content(msg))
        case ChatMessageAssistant(tool_calls=tool_calls):
            result: Message = Message(
                role=msg.role,
                content=_extract_content(msg),
            )
            if tool_calls:
                # Tinker uses a different ToolCall format, but for basic compatibility
                # we store tool calls in a format that can be reconstructed
                from tinker_cookbook.renderers import ToolCall

                result["tool_calls"] = [
                    ToolCall(
                        function=ToolCall.FunctionBody(
                            name=tc.function,
                            arguments=tc.arguments
                            if isinstance(tc.arguments, str)
                            else str(tc.arguments),
                        ),
                        id=tc.id,
                    )
                    for tc in tool_calls
                ]
            return result
        case ChatMessageTool(tool_call_id=tool_call_id, function=function):
            result = Message(role=msg.role, content=_extract_content(msg))
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
            if function:
                result["name"] = function
            return result


def chat_messages_to_tinker(messages: list[ChatMessage]) -> list[Message]:
    """Convert a list of Inspect ChatMessages to Tinker Messages."""
    return [chat_message_to_tinker(msg) for msg in messages]


def chat_message_to_dict(msg: ChatMessage) -> MessageDict:
    """
    Convert an Inspect ChatMessage to a serializable dict for HuggingFace dataset.

    This is a simpler format than Tinker Message, suitable for storage.
    """
    match msg:
        case ChatMessageUser() | ChatMessageSystem():
            return MessageDict(role=msg.role, content=_extract_content(msg))
        case ChatMessageAssistant(tool_calls=tool_calls):
            result = MessageDict(
                role=msg.role,
                content=_extract_content(msg),
            )
            if tool_calls:
                result["tool_calls"] = [
                    ToolCallDict(
                        id=tc.id,
                        type=getattr(tc, "type", "function"),
                        function={
                            "name": tc.function,
                            "arguments": tc.arguments
                            if isinstance(tc.arguments, str)
                            else str(tc.arguments),
                        },
                    )
                    for tc in tool_calls
                ]
            return result
        case ChatMessageTool(tool_call_id=tool_call_id, function=function):
            result = MessageDict(role=msg.role, content=_extract_content(msg))
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
            if function:
                result["name"] = function
            return result
