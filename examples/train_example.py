"""
Example: Train Llama-3.1-8B on a simple math task using Inspect-Tinker bridge.

Run with:
    cd examples && uv run --env-file ../.env python train_example.py

To use a different model, set MODEL_NAME below.
"""

import asyncio
import logging

import chz
from tinker_cookbook import model_info
from tinker_cookbook import renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from simple_math_task import simple_math
from inspect_tinker_bridge import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Use Llama-3.1-8B-Instruct (known to work with Tinker)
# Alternatives: "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RENDERER_NAME = model_info.get_recommended_renderer_name(MODEL_NAME)


@chz.chz
class InspectDatasetBuilder(RLDatasetBuilder):
    """
    Build an RLDataset from an Inspect task using our bridge.
    """

    batch_size: int = 4
    group_size: int = 4
    max_samples: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(MODEL_NAME)
        renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)

        dataset = load_environment(
            simple_math,
            renderer=renderer,
            env_type="single_turn",
            max_samples=self.max_samples,
            batch_size=self.batch_size,
            num_envs_per_group=self.group_size,
        )

        logger.info(f"Created RLDataset with {len(dataset)} batches")
        return dataset, None


def build_config() -> train.Config:
    """Build the training configuration."""
    builder = InspectDatasetBuilder(
        batch_size=4,
        group_size=4,
        max_samples=16,
    )

    return train.Config(
        model_name=MODEL_NAME,
        dataset_builder=builder,
        learning_rate=1e-5,
        max_tokens=64,
        temperature=1.0,
        lora_rank=32,
        loss_fn="importance_sampling",
        log_path="/tmp/inspect-tinker-bridge/train_example",
        eval_every=0,
        save_every=10,
        num_substeps=1,
    )


async def main() -> None:
    """Run training."""
    logger.info("Starting RL training on simple math task")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Renderer: {RENDERER_NAME}")

    config = build_config()
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(main())
