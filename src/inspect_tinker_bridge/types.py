"""
Type definitions for the Inspect-Tinker bridge.

This module provides TypedDict definitions for structured data that flows
through the bridge, enabling better type checking and IDE support.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Required, TypedDict

if TYPE_CHECKING:
    from inspect_ai.scorer import Score
    from inspect_ai.util._sandbox.environment import SandboxEnvironment


class ToolCallFunctionDict(TypedDict):
    """Function details in a tool call."""

    name: str
    arguments: str


class ToolCallDict(TypedDict):
    """Tool call in OpenAI-style format."""

    id: str | None  # Can be None for formats that don't include IDs (e.g., Harmony)
    type: Literal["function"]
    function: ToolCallFunctionDict


class MessageDict(TypedDict, total=False):
    """
    Message in OpenAI-style chat format.

    Required keys: role, content
    Optional keys: tool_calls, tool_call_id, name, reasoning_content
    """

    role: Required[Literal["system", "user", "assistant", "tool"]]
    content: Required[str]
    tool_calls: list[ToolCallDict]
    tool_call_id: str
    name: str
    reasoning_content: str  # Chain-of-thought / thinking tokens


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


@dataclass
class ScoringContext:
    """Full context passed to custom reward functions.

    Attributes:
        conversation: Full conversation history (shallow copy, safe to read).
        sample_info: Sample metadata including target answers.
        scores: Scorer results keyed by name (None if scorer returned None).
        individual_rewards: Float rewards per scorer (0.0 for None scores).
        base_reward: Combined reward that would be returned without callback.
        current_turn: Current turn number in episode.
        max_turns: Maximum turns configured for episode.
        answer: Ground truth answer from dataset (may be None).
        sandbox_envs: Sandbox environments (available for multi-turn with sandbox).
    """

    conversation: list[MessageDict]
    sample_info: SampleInfoDict
    scores: dict[str, "Score | None"]
    individual_rewards: dict[str, float]
    base_reward: float
    current_turn: int
    max_turns: int
    answer: str | None
    sandbox_envs: "dict[str, SandboxEnvironment] | None" = None


# Custom reward function type - accepts sync or async, returns float or (float, dict)
CustomRewardFn = Callable[
    [ScoringContext],
    float
    | tuple[float, dict[str, float]]
    | Awaitable[float | tuple[float, dict[str, float]]],
]
