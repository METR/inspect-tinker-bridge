"""Rollout saving utilities for Inspect-Tinker bridge.

This module provides a pre-configured rollout saving wrapper that knows how to
serialize ScoringContext and Score objects from inspect-tinker-bridge.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.rl.rollout_saving import (
    compute_message_tokens,
)
from tinker_cookbook.rl.rollout_saving import (
    with_rollout_saving as _with_rollout_saving,
)

from inspect_tinker_bridge.scoring import score_to_reward
from inspect_tinker_bridge.types import ScoringContext

logger = logging.getLogger(__name__)

RolloutRewardFnSig = Callable[[ScoringContext], tuple[float, dict[str, float]]]


def _safe_json_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form.

    Attempts to serialize the object to JSON. If that fails, returns a string
    representation to avoid crashing during rollout saving.
    """
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError) as e:
        logger.warning(f"Non-serializable metadata converted to string: {e}")
        return {"_raw": str(obj)}


def with_rollout_saving(
    inner_fn: RolloutRewardFnSig,
    output_path: Path,
    renderer: Renderer,
    samples_per_batch: int,
    save_every: int = 10,
) -> RolloutRewardFnSig:
    """Wrap a reward function to save rollouts periodically.

    This is a convenience wrapper around tinker_cookbook.rl.rollout_saving.with_rollout_saving
    that knows how to serialize ScoringContext and Score objects.

    Args:
        inner_fn: The original reward function to wrap
        output_path: Path to the rollouts.jsonl file
        renderer: Tinker Renderer instance (used for renderer_name and tokenizer)
        samples_per_batch: Number of samples per batch (batch_size * group_size)
        save_every: Save rollouts every N steps (default: 10)

    Returns:
        Wrapped function that periodically saves selected rollouts to JSONL

    Raises:
        ValueError: If samples_per_batch or save_every are not positive
    """
    if samples_per_batch <= 0:
        raise ValueError(f"samples_per_batch must be positive, got {samples_per_batch}")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")

    tokenizer = renderer.tokenizer

    # Get the renderer's system message (e.g. reasoning effort, date) so we
    # can include it in rollout records.  This is prepended at render-time by
    # build_generation_prompt() but isn't part of env.conversation.
    renderer_system_msg: dict[str, Any] | None = None
    if hasattr(renderer, "_get_system_message"):
        msg = renderer._get_system_message()  # pyright: ignore[reportAttributeAccessIssue]
        if msg is not None:
            renderer_system_msg = dict(msg)  # pyright: ignore[reportUnknownArgumentType]
            renderer_system_msg["role"] = "system"

    def build_record(
        ctx: ScoringContext,
        total_reward: float,
        rewards: dict[str, float],
        step: int,
        renderer_name: str,
    ) -> dict[str, Any]:
        """Build rollout record with inspect-specific score details."""
        # Build score details inline
        score_details: dict[str, dict[str, Any]] = {}
        for key, score in ctx.scores.items():
            if score is not None:
                score_details[key] = {
                    "value": score_to_reward(score),
                    "answer": score.answer,
                    "explanation": score.explanation,
                    "metadata": _safe_json_serializable(score.metadata or {}),
                }
            else:
                logger.warning(f"Missing score for {key}")
                score_details[key] = {
                    "value": 0.0,
                    "answer": None,
                    "explanation": None,
                    "metadata": {},
                }

        # Prepend renderer system message to get the full prompt the model saw
        conversation = [dict(msg) for msg in ctx.conversation]
        if renderer_system_msg is not None:
            conversation = [renderer_system_msg] + conversation

        # Compute token counts
        token_counts = [compute_message_tokens(msg, tokenizer) for msg in conversation]

        return {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "sample_id": ctx.sample_info.get("inspect_sample_id"),
            "conversation": conversation,
            "token_counts": token_counts,
            "sample_info": dict(ctx.sample_info),
            "scores": score_details,
            "individual_rewards": rewards,
            "total_reward": total_reward,
            "renderer_name": renderer_name,
        }

    return _with_rollout_saving(
        inner_fn,
        output_path,
        renderer,
        samples_per_batch=samples_per_batch,
        save_every=save_every,
        build_record=build_record,
    )
