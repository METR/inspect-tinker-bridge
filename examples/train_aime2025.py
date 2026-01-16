"""
Example: Train Qwen3-30B-A3B-Instruct on AIME 2025 using Inspect-Tinker bridge.

AIME (American Invitational Mathematics Examination) 2025 problems test advanced
high school mathematics. This trains with RL on the 30 AIME problems.

Run with:
    cd examples && uv run --env-file ../.env python train_aime2025.py

View rollouts with:
    uv run python -m tinker_cookbook.rollout_viewer.app /tmp/inspect-tinker-bridge/train_aime2025/rollouts.jsonl --watch

To use a different model, change MODEL_NAME below.
"""

import asyncio
import logging
from pathlib import Path

import chz
from inspect_evals.aime2025 import aime2025
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from inspect_tinker_bridge import load_environment
from inspect_tinker_bridge.rollout_saving import with_rollout_saving
from inspect_tinker_bridge.types import ScoringContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Qwen3-30B-A3B-Instruct: MoE model with ~30B active params, efficient for training
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = model_info.get_recommended_renderer_name(MODEL_NAME)


# Output directory for logs and rollouts
LOG_PATH = Path("/tmp/inspect-tinker-bridge/train_aime2025")


def base_reward_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
    """Base reward function that uses Inspect scorer results."""
    return ctx.base_reward, ctx.individual_rewards


@chz.chz
class Aime2025DatasetBuilder(RLDatasetBuilder):
    """
    Build an RLDataset from AIME 2025 using the Inspect-Tinker bridge.
    """

    batch_size: int = 4
    group_size: int = 8
    max_samples: int | None = None
    save_rollouts: bool = True

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(MODEL_NAME)
        renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)

        # Optionally wrap reward function to save rollouts
        custom_reward_fn = None
        if self.save_rollouts:
            LOG_PATH.mkdir(parents=True, exist_ok=True)
            rollouts_path = LOG_PATH / "rollouts.jsonl"
            custom_reward_fn = with_rollout_saving(
                base_reward_fn,
                output_path=rollouts_path,
                renderer=renderer,
                samples_per_batch=self.batch_size * self.group_size,
                save_every=1,  # Save rollouts every step for visibility
            )
            logger.info(f"Rollout saving enabled: {rollouts_path}")

        dataset = load_environment(
            aime2025,
            renderer=renderer,
            env_type="single_turn",
            max_samples=self.max_samples,
            batch_size=self.batch_size,
            num_envs_per_group=self.group_size,
            custom_reward_fn=custom_reward_fn,
        )

        logger.info(f"Created AIME 2025 RLDataset with {len(dataset)} batches")
        return dataset, None


def build_config() -> train.Config:
    """Build the training configuration for AIME 2025."""
    builder = Aime2025DatasetBuilder(
        batch_size=4,
        group_size=8,
        max_samples=None,  # Use all 30 AIME problems
        save_rollouts=True,
    )

    return train.Config(
        model_name=MODEL_NAME,
        dataset_builder=builder,
        learning_rate=4e-5,  # Standard RL learning rate
        max_tokens=1024,  # Math solutions need more tokens for reasoning
        temperature=1.0,
        lora_rank=32,
        loss_fn="importance_sampling",
        log_path=str(LOG_PATH),
        eval_every=0,
        save_every=5,
        num_substeps=1,
    )


async def main() -> None:
    """Run RL training on AIME 2025."""
    logger.info("Starting RL training on AIME 2025")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Renderer: {RENDERER_NAME}")

    config = build_config()
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(main())
