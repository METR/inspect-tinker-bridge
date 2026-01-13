"""
Type definitions for the Inspect-Tinker bridge.

This module provides TypedDict definitions for structured data that flows
through the bridge, enabling better type checking and IDE support.
"""

import json
from typing import Literal, Required, TypedDict


class ToolCallFunctionDict(TypedDict):
    """Function details in a tool call."""

    name: str
    arguments: str


class ToolCallDict(TypedDict):
    """Tool call in OpenAI-style format."""

    id: str
    type: Literal["function"]
    function: ToolCallFunctionDict


class MessageDict(TypedDict, total=False):
    """
    Message in OpenAI-style chat format.

    Required keys: role, content
    Optional keys: tool_calls, tool_call_id, name
    """

    role: Required[Literal["system", "user", "assistant", "tool"]]
    content: Required[str]
    tool_calls: list[ToolCallDict]
    tool_call_id: str
    name: str


class SampleInfoDict(TypedDict, total=False):
    """
    Info dict stored in HuggingFace dataset for each sample.

    Contains all Inspect-specific metadata needed for scoring and sandbox setup.
    """

    inspect_sample_id: str | int | None
    inspect_input_raw: str | list[MessageDict]
    inspect_target_raw: str | list[str] | None
    inspect_choices: list[str] | None
    inspect_metadata: str  # JSON-serialized dict for pyarrow compatibility
    inspect_sandbox: str | tuple[str, str] | None
    inspect_files: dict[str, str] | None
    inspect_setup: str | None
    inspect_task_name: str


class DatasetRowDict(TypedDict):
    """Row in the HuggingFace dataset produced by dataset conversion."""

    prompt: list[MessageDict]
    answer: str | None
    info: SampleInfoDict
    id: str | int | None


def parse_metadata_json(metadata_raw: str) -> dict[str, object]:
    """
    Parse JSON metadata string from SampleInfoDict.

    Handles empty strings safely and validates the result is a dict.

    Args:
        metadata_raw: JSON string (empty string returns empty dict)

    Returns:
        Parsed metadata dict

    Raises:
        ValueError: If parsed value is not a dict
    """
    parsed = json.loads(metadata_raw or "{}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected metadata to be a dict, got {type(parsed).__name__}")
    return parsed
