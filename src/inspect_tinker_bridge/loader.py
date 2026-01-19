"""
Main loader: Convert Inspect tasks to Tinker RLDataset.
"""

import logging
from collections.abc import Callable
from typing import Literal

from inspect_ai import Task
from tinker_cookbook.renderers import Renderer

from inspect_tinker_bridge import dataset as ds
from inspect_tinker_bridge import scoring
from inspect_tinker_bridge import tasks
from inspect_tinker_bridge.env import InspectRLDataset
from inspect_tinker_bridge.sandbox import SandboxConfig
from inspect_tinker_bridge.types import CustomRewardFn

logger = logging.getLogger(__name__)


def load_environment(
    task: Callable[..., Task],
    renderer: Renderer,
    *,
    env_type: Literal["single_turn", "multi_turn"] = "single_turn",
    max_samples: int | None = None,
    max_turns: int = 10,
    num_envs_per_group: int = 1,
    batch_size: int = 1,
    num_epochs: int = 1,
    sandbox_type: str | None = None,
    sandbox_config: str | None = None,
    submit_instruction: str
    | None = "You must call submit() when you are done to complete the task.",
    custom_reward_fn: CustomRewardFn | None = None,
    custom_reward_fn_timeout: float = scoring.CUSTOM_REWARD_FN_TIMEOUT,
    **task_kwargs: object,
) -> InspectRLDataset:
    """
    Load an Inspect task and convert it to a Tinker RLDataset.

    Args:
        task: A callable that returns an Inspect Task
        renderer: Tinker Renderer for tokenization and message formatting
        env_type: Environment type:
            - "single_turn": Single response from model
            - "multi_turn": Multi-turn with tools (requires sandbox)
        max_samples: Limit number of samples from dataset
        max_turns: Max turns for multi-turn environments (default: 10)
        num_envs_per_group: Number of parallel rollouts per problem (for GRPO, etc.)
        batch_size: Number of problems per batch
        num_epochs: Number of passes through the dataset (default: 1)
        sandbox_type: Override sandbox type (e.g., "docker", "local")
        sandbox_config: Sandbox configuration file path
        submit_instruction: Instruction appended to system prompt for multi-turn
            environments explaining how to use the submit tool.
            - str: Use custom instruction
            - None: No instruction added
        custom_reward_fn: Optional function to customize reward computation.
            Receives ScoringContext with full scorer results and returns
            float or (float, dict) with custom reward and optional metrics.
            Can be sync or async.
        custom_reward_fn_timeout: Timeout in seconds for custom_reward_fn (default 30s)
        **task_kwargs: Arguments to pass to the Inspect task function

    Returns:
        InspectRLDataset ready for Tinker RL training.

    Example:
        ```python
        from inspect_evals.gsm8k import gsm8k
        from tinker_cookbook.renderers import Qwen3Renderer
        from tinker_cookbook.tokenizer_utils import Tokenizer

        tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        renderer = Qwen3Renderer(tokenizer)

        dataset = load_environment(
            gsm8k,
            renderer=renderer,
            env_type="single_turn",
            max_samples=100,
            batch_size=8,
            num_envs_per_group=4,
        )

        # Use with Tinker training loop
        builders = dataset.get_batch(0)
        ```
    """
    logger.info(
        f"Loading environment: env_type={env_type}, "
        f"max_samples={max_samples}, max_turns={max_turns}, "
        f"num_envs_per_group={num_envs_per_group}, batch_size={batch_size}, "
        f"num_epochs={num_epochs}"
    )

    # Load and introspect the task
    logger.debug(f"Loading Inspect task with kwargs: {task_kwargs}")
    task_info = tasks.load_inspect_task(task, **task_kwargs)
    logger.info(
        f"Task loaded: name={task_info.name}, sandbox_type={task_info.sandbox_type}, "
        f"num_scorers={len(task_info.scorers)}, num_samples={len(task_info.dataset)}"
    )

    # Validate configuration
    if not task_info.scorers:
        raise ValueError(
            f"Task {task_info.name} has no scorers. "
            "At least one scorer is required for reward computation."
        )

    # Determine if we need a sandbox
    effective_sandbox_type = sandbox_type or task_info.sandbox_type

    # Multi-turn requires sandbox
    if env_type == "multi_turn" and not effective_sandbox_type:
        raise ValueError(
            "Multi-turn environment requires a sandbox. "
            "Either use a task with sandbox configuration or specify sandbox_type."
        )

    # Determine additional system content for multi-turn
    additional_system_content = (
        submit_instruction if env_type == "multi_turn" and submit_instruction else None
    )

    # Convert dataset to HuggingFace format
    logger.debug("Converting dataset to HuggingFace format")
    hf_dataset = ds.inspect_dataset_to_hf(
        task_info.task,
        task_name=task_info.name,
        max_samples=max_samples,
        additional_system_content=additional_system_content,
    )
    logger.info(f"Dataset converted: {len(hf_dataset)} samples")

    # Create sandbox config if needed
    sandbox_cfg = None
    if effective_sandbox_type:
        sandbox_cfg = SandboxConfig(
            sandbox_type=effective_sandbox_type,
            config=sandbox_config,
        )
        logger.info(f"Sandbox configured: type={effective_sandbox_type}")

    # Create and return the RLDataset
    return InspectRLDataset(
        hf_dataset=hf_dataset,
        renderer=renderer,
        scorers=task_info.scorers,
        env_type=env_type,
        max_turns=max_turns,
        sandbox_config=sandbox_cfg,
        num_envs_per_group=num_envs_per_group,
        batch_size=batch_size,
        task_name=task_info.name,
        custom_reward_fn=custom_reward_fn,
        custom_reward_fn_timeout=custom_reward_fn_timeout,
        num_epochs=num_epochs,
    )
