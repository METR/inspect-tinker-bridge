"""
Scoring bridge: Convert Inspect scorers to Tinker reward functions.

This module provides the core mechanism to call Inspect scorers within the
Tinker environment framework.
"""

import json
import logging
from typing import Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.scorer import Score, Scorer, Target, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall
from inspect_ai.util._sandbox.environment import SandboxEnvironment

from inspect_tinker_bridge import utils
from inspect_tinker_bridge.types import MessageDict, SampleInfoDict, parse_metadata_json

logger = logging.getLogger(__name__)

# ToolCall type values
_TOOL_CALL_TYPE_FUNCTION: Literal["function"] = "function"


async def run_inspect_scorer(
    conversation: list[MessageDict],
    info: SampleInfoDict,
    scorer: Scorer,
    sandbox_envs: dict[str, SandboxEnvironment] | None = None,
) -> Score | None:
    """
    Run an Inspect scorer and return the full Score object.

    Args:
        conversation: The full conversation (prompt + completions) as list of dicts
        info: The info dict from the sample containing Inspect metadata
        scorer: The Inspect scorer to use
        sandbox_envs: Optional sandbox environments for sandbox-based scorers

    Returns:
        Score object with full details, or None if scorer returned None
    """
    sample_id = info.get("inspect_sample_id")
    logger.debug(f"Running scorer for sample_id={sample_id}")

    # Get the raw target from info
    target_raw = info.get("inspect_target_raw")
    target = Target(target_raw) if target_raw is not None else Target("")

    # Build messages list for TaskState
    messages = _build_inspect_messages(conversation)

    # Build model output from the last assistant message
    model_output = _build_model_output(conversation)

    # Get original input from info (pre-solver, matches native Inspect semantics)
    input_raw = info.get("inspect_input_raw", "")
    original_input: str | list[ChatMessage]
    if isinstance(input_raw, str):
        original_input = input_raw
    else:
        original_input = _build_inspect_messages(input_raw)

    # Deserialize metadata from JSON string
    metadata_raw = info.get("inspect_metadata", "{}")
    metadata = parse_metadata_json(metadata_raw)

    # sample_id can be None but TaskState expects int | str
    # Use a default when None
    effective_sample_id: int | str = sample_id if sample_id is not None else "unknown"

    task_state = TaskState(
        model=utils.BRIDGE_MODEL_NAME,
        sample_id=effective_sample_id,
        epoch=0,
        input=original_input,
        messages=messages,
        target=target,
        output=model_output,
        metadata=metadata,
    )

    # Run scorer with sandbox context if available
    score: Score | None
    if sandbox_envs is not None:
        from inspect_tinker_bridge import sandbox as sandbox_module

        logger.debug(f"Running scorer with sandbox context for sample_id={sample_id}")
        async with sandbox_module.sandbox_context(sandbox_envs):
            score = await scorer(task_state, target)
    else:
        logger.debug(f"Running scorer without sandbox for sample_id={sample_id}")
        score = await scorer(task_state, target)

    if score is not None:
        logger.debug(
            f"Scorer completed for sample_id={sample_id}: value={score.value}, "
            f"answer={score.answer}"
        )
    else:
        logger.warning(f"Scorer returned None for sample_id={sample_id}")

    return score


def score_to_reward(score: Score | None) -> float:
    """
    Convert an Inspect Score to a float reward.

    Args:
        score: Inspect Score object (or None)

    Returns:
        Float reward value (typically 0.0-1.0)
    """
    if score is None:
        return 0.0
    # score.value can be None at runtime even though types say otherwise
    if score.value is None:  # pyright: ignore[reportUnnecessaryComparison]
        return 0.0
    converter = value_to_float()
    return converter(score.value)


async def compute_reward(
    conversation: list[MessageDict],
    info: SampleInfoDict,
    scorers: list[Scorer],
    sandbox_envs: dict[str, SandboxEnvironment] | None = None,
    weights: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute combined reward from multiple Inspect scorers.

    Args:
        conversation: The full conversation as list of dicts
        info: The info dict from the sample
        scorers: List of Inspect scorers to run
        sandbox_envs: Optional sandbox environments
        weights: Optional weights for each scorer (must match len(scorers))

    Returns:
        Tuple of (combined_reward, individual_rewards_dict)
    """
    if not scorers:
        return 0.0, {}

    if weights is not None:
        if len(weights) != len(scorers):
            raise ValueError(
                f"weights has {len(weights)} elements but have {len(scorers)} scorers"
            )
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        if sum(weights) == 0:
            raise ValueError("weights sum to zero, cannot compute weighted average")

    individual_rewards: dict[str, float] = {}
    total_weight = sum(weights) if weights else len(scorers)

    for i, scorer in enumerate(scorers):
        scorer_name = _get_scorer_name(scorer)
        score = await run_inspect_scorer(
            conversation=conversation,
            info=info,
            scorer=scorer,
            sandbox_envs=sandbox_envs,
        )
        reward = score_to_reward(score)
        individual_rewards[f"{scorer_name}_{i}"] = reward

    # Compute weighted average
    if weights:
        combined = sum(
            individual_rewards[f"{_get_scorer_name(scorer)}_{i}"] * w
            for i, (scorer, w) in enumerate(zip(scorers, weights, strict=True))
        )
        combined /= total_weight
    else:
        combined = sum(individual_rewards.values()) / len(individual_rewards)

    return combined, individual_rewards


def _get_scorer_name(scorer: Scorer) -> str:
    """Extract a unique name from a scorer."""
    # Try registry info first
    try:
        from inspect_ai._util.registry import registry_info

        info = registry_info(scorer)
        return info.name.split("/")[-1]
    except (ValueError, AttributeError):
        pass

    # Fall back to qualname
    qualname = getattr(scorer, "__qualname__", "")
    if ".<locals>." in qualname:
        return qualname.split(".<locals>.")[0]
    return getattr(scorer, "__name__", scorer.__class__.__name__)


def _parse_tool_arguments(args: str) -> dict[str, object]:
    """Parse tool arguments JSON, returning raw string as fallback on error."""
    try:
        return json.loads(args)
    except json.JSONDecodeError as e:
        truncated = args[:100] + ("..." if len(args) > 100 else "")
        logger.warning(f"Failed to parse tool arguments: {e} (args: {truncated})")
        return {"_raw": args}


def _build_inspect_messages(messages: list[MessageDict]) -> list[ChatMessage]:
    """Convert message dicts to Inspect ChatMessage objects."""
    result: list[ChatMessage] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            result.append(ChatMessageSystem(content=content))
        elif role == "user":
            result.append(ChatMessageUser(content=content))
        elif role == "assistant":
            tool_calls = None
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        function=tc["function"]["name"],
                        arguments=_parse_tool_arguments(tc["function"]["arguments"]),
                        type=_TOOL_CALL_TYPE_FUNCTION,
                    )
                    for tc in msg["tool_calls"]
                ]
            result.append(ChatMessageAssistant(content=content, tool_calls=tool_calls))
        elif role == "tool":
            result.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=msg.get("tool_call_id"),
                    function=msg.get("name"),
                )
            )
        else:
            raise ValueError(f"Unknown role: {role}")

    return result


def _build_model_output(conversation: list[MessageDict]) -> ModelOutput:
    """Build ModelOutput from the last assistant message."""
    for msg in reversed(conversation):
        if msg["role"] == "assistant":
            # Skip submit tool calls - the real answer is in an earlier message
            if _is_submit_tool_call(msg):
                continue
            return ModelOutput.from_content(
                model=str(utils.BRIDGE_MODEL_NAME),
                content=msg.get("content", ""),
            )
    # No assistant message found - return empty
    return ModelOutput.from_content(model=str(utils.BRIDGE_MODEL_NAME), content="")


def _is_submit_tool_call(msg: MessageDict) -> bool:
    """Check if an assistant message is a submit tool call."""
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return False
    # Use .get() for safety when data comes from external sources (HF dataset)
    return any(tc.get("function", {}).get("name") == "submit" for tc in tool_calls)
