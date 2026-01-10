"""
Example: Train on multi-turn coding task with Docker sandbox.

This demonstrates the Inspect-Tinker bridge with:
- Multi-turn agent loop (model can use tools iteratively)
- Docker sandbox for code execution
- Per-step reward tracking

Run with:
    cd examples && uv run --env-file ../.env python train_multiturn.py
"""

import asyncio
import json
import logging
from pathlib import Path

import chz
from tinker_cookbook import model_info
from tinker_cookbook import renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from coding_task import coding_task
from inspect_tinker_bridge import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RENDERER_NAME = model_info.get_recommended_renderer_name(MODEL_NAME)
LOG_DIR = Path("/tmp/inspect-tinker-bridge/train_multiturn")


@chz.chz
class MultiTurnDatasetBuilder(RLDatasetBuilder):
    """
    Build an RLDataset from a multi-turn Inspect task.
    """

    batch_size: int = 2
    group_size: int = 2
    max_samples: int | None = None
    max_turns: int = 10

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(MODEL_NAME)
        renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)

        dataset = load_environment(
            coding_task,
            renderer=renderer,
            env_type="multi_turn",
            max_samples=self.max_samples,
            max_turns=self.max_turns,
            batch_size=self.batch_size,
            num_envs_per_group=self.group_size,
            sandbox_type="docker",
        )

        logger.info(f"Created multi-turn RLDataset with {len(dataset)} batches")
        return dataset, None


def build_config() -> train.Config:
    """Build the training configuration."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    builder = MultiTurnDatasetBuilder(
        batch_size=2,
        group_size=2,
        max_samples=4,
        max_turns=10,
    )

    return train.Config(
        model_name=MODEL_NAME,
        dataset_builder=builder,
        learning_rate=1e-5,
        max_tokens=512,
        temperature=1.0,
        lora_rank=32,
        loss_fn="importance_sampling",
        log_path=str(LOG_DIR),
        eval_every=0,
        save_every=5,
        num_substeps=1,
        num_groups_to_log=2,
    )


async def main() -> None:
    """Run training with reward curve tracking."""
    logger.info("Starting multi-turn RL training on coding task")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Renderer: {RENDERER_NAME}")
    logger.info(f"Log directory: {LOG_DIR}")

    config = build_config()
    await train.main(config)

    # Print summary of training metrics
    metrics_file = LOG_DIR / "metrics.jsonl"
    if metrics_file.exists():
        logger.info("\n=== Training Summary ===")
        with open(metrics_file) as f:
            for line in f:
                metrics = json.loads(line)
                step = metrics.get("progress/batch", "?")
                reward = metrics.get("env/all/reward/total", "?")
                correct = metrics.get("env/all/correct", "?")
                turns = metrics.get("env/all/turns_per_episode", "?")
                logger.info(
                    f"Step {step}: reward={reward:.3f}, correct={correct:.3f}, turns={turns:.1f}"
                    if isinstance(reward, (int, float))
                    else f"Step {step}: reward={reward}, correct={correct}, turns={turns}"
                )


if __name__ == "__main__":
    asyncio.run(main())
