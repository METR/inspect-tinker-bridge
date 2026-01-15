"""Rollout saving utilities for Inspect-Tinker bridge.

This module provides a pre-configured rollout saving wrapper that knows how to
serialize ScoringContext and Score objects from inspect-tinker-bridge.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.rl.rollout_saving import (
    compute_message_tokens,
    with_rollout_saving as _with_rollout_saving,
)

from inspect_tinker_bridge.scoring import score_to_reward
from inspect_tinker_bridge.types import ScoringContext

# Type alias for reward functions that take ScoringContext
RewardFn = Callable[[ScoringContext], tuple[float, dict[str, float]]]


def _build_score_details(ctx: ScoringContext) -> dict[str, dict[str, Any]]:
    """Convert ScoringContext.scores to serializable score_details dict."""
    details: dict[str, dict[str, Any]] = {}
    for key, score in ctx.scores.items():
        if score is not None:
            details[key] = {
                "value": score_to_reward(score),
                "answer": score.answer,
                "metadata": score.metadata or {},
            }
        else:
            details[key] = {
                "value": 0.0,
                "answer": None,
                "metadata": {},
            }
    return details


def with_rollout_saving(
    inner_fn: RewardFn,
    output_path: Path,
    renderer: Renderer,
    samples_per_batch: int,
    save_every: int = 10,
) -> RewardFn:
    """Wrap a reward function to save rollouts periodically.

    This is a convenience wrapper around tinker_cookbook.rl.rollout_saving.with_rollout_saving
    that knows how to serialize ScoringContext and Score objects.

    Saves 3 rollouts (best, worst, random) every `save_every` steps.

    Args:
        inner_fn: The original reward function to wrap
        output_path: Path to the rollouts.jsonl file
        renderer: Tinker Renderer instance (used for renderer_name and tokenizer)
        samples_per_batch: Number of samples per batch (batch_size * group_size)
        save_every: Save rollouts every N steps (default: 10)

    Returns:
        Wrapped function that periodically saves selected rollouts to JSONL
    """
    tokenizer = renderer.tokenizer

    def build_record(
        ctx: ScoringContext,
        total_reward: float,
        rewards: dict[str, float],
        step: int,
        renderer_name: str,
    ) -> dict[str, Any]:
        """Build rollout record with inspect-specific score details."""
        score_details = _build_score_details(ctx)

        # Compute token counts
        token_counts = [
            compute_message_tokens(dict(msg), tokenizer) for msg in ctx.conversation
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "sample_id": ctx.sample_info.get("inspect_sample_id"),
            "conversation": [dict(msg) for msg in ctx.conversation],
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
