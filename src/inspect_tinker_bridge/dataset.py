"""
Dataset conversion utilities: Inspect Sample -> HuggingFace Dataset.

Uses ground truth solver execution for accurate prompt construction.
"""

import asyncio
import json
import logging
from collections.abc import Coroutine
from concurrent import futures
from typing import TypeVar

from datasets import Dataset as HFDataset
from inspect_ai import Task
from inspect_ai.dataset import Sample

from inspect_tinker_bridge import ground_truth
from inspect_tinker_bridge import utils
from inspect_tinker_bridge.types import DatasetRowDict, MessageDict, SampleInfoDict

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def sample_to_row(
    sample: Sample,
    task: Task,
    task_name: str,
    additional_system_content: str | None = None,
) -> DatasetRowDict:
    """
    Convert an Inspect Sample to a HuggingFace dataset row.

    Uses ground truth solver execution to get the actual transformed messages.

    Args:
        sample: An Inspect Sample object
        task: The Inspect Task (contains solver chain)
        task_name: Name of the task (for tracking)
        additional_system_content: Optional content to append to the system message

    Returns:
        Dictionary with prompt, answer, info, and id fields
    """
    logger.debug(f"Converting sample {sample.id} to dataset row")

    # Get ground truth messages from solver pipeline
    messages = await ground_truth.get_ground_truth_messages(task, sample)
    prompt_messages: list[MessageDict] = [
        utils.chat_message_to_dict(msg) for msg in messages
    ]
    logger.debug(f"Sample {sample.id}: converted {len(messages)} messages to prompt")

    # Append additional content to system message if provided
    if additional_system_content:
        logger.debug(f"Sample {sample.id}: appending additional system content")
        prompt_messages = _append_to_system_message(
            prompt_messages, additional_system_content
        )

    # Convert target to string answer
    answer = _target_to_text(sample.target)

    # Store original input (pre-solver) for TaskState.input in scoring
    # Convert ChatMessage list to dicts for serialization
    input_raw: str | list[MessageDict]
    if isinstance(sample.input, str):
        input_raw = sample.input
    else:
        input_raw = [utils.chat_message_to_dict(msg) for msg in sample.input]

    # Store all Inspect-specific data in info for later use
    # Note: inspect_metadata is serialized to JSON string because pyarrow
    # can't handle dicts with varying schemas across samples
    # Note: sandbox type is complex at runtime (can include SandboxEnvironmentSpec)
    # but we only serialize string/tuple forms
    sandbox_serializable: str | tuple[str, str] | None = None
    if isinstance(sample.sandbox, (str, tuple)):
        sandbox_serializable = sample.sandbox
    elif sample.sandbox is not None:
        raise ValueError(
            f"Unsupported sandbox type for serialization: {type(sample.sandbox)}. "
            "Only string or tuple[str, str] sandbox specs are supported."
        )

    info: SampleInfoDict = {
        "inspect_sample_id": sample.id,
        "inspect_input_raw": input_raw,
        "inspect_target_raw": sample.target,
        "inspect_choices": sample.choices,
        "inspect_metadata": json.dumps(sample.metadata or {}),
        "inspect_sandbox": sandbox_serializable,
        "inspect_files": sample.files,
        "inspect_setup": sample.setup,
        "inspect_task_name": task_name,
    }

    return {
        "prompt": prompt_messages,
        "answer": answer,
        "info": info,
        "id": sample.id,
    }


def _append_to_system_message(
    messages: list[MessageDict], content: str
) -> list[MessageDict]:
    """
    Append content to the system message, or create one if it doesn't exist.

    Args:
        messages: List of message dicts
        content: Content to append to system message

    Returns:
        Updated message list with appended system content
    """
    # Find existing system message
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            # Append to existing system message (content is Required in MessageDict)
            existing = msg["content"]
            separator = "\n\n" if existing else ""
            updated_msg: MessageDict = {
                **msg,
                "content": f"{existing}{separator}{content}",
            }
            return messages[:i] + [updated_msg] + messages[i + 1 :]

    # No system message found - create one at the beginning
    system_msg: MessageDict = {"role": "system", "content": content}
    return [system_msg] + messages


def _target_to_text(target: str | list[str] | None) -> str | None:
    """Convert an Inspect target to a text string."""
    if target is None:
        return None
    if isinstance(target, str):
        return target
    # For list targets (like test cases), join them
    # Empty list returns empty string (distinct from None)
    return "\n".join(str(t) for t in target)


def _run_async_in_thread(coro: Coroutine[object, object, T]) -> T:
    """Run an async coroutine in a separate thread to avoid event loop conflicts."""

    def runner() -> T:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner)
        return future.result()


def inspect_dataset_to_hf(
    task: Task,
    task_name: str,
    max_samples: int | None = None,
    additional_system_content: str | None = None,
) -> HFDataset:
    """
    Convert an Inspect dataset to a HuggingFace Dataset using ground truth.

    Args:
        task: The Inspect Task (contains dataset and solver chain)
        task_name: Name of the task
        max_samples: Optional limit on number of samples to convert
        additional_system_content: Optional content to append to system messages
            (e.g., instructions for using tools)

    Returns:
        A HuggingFace Dataset with columns: prompt, answer, info, id
    """
    total_samples = len(task.dataset)
    samples_to_convert = (
        min(total_samples, max_samples) if max_samples is not None else total_samples
    )
    logger.info(
        f"Converting Inspect dataset to HuggingFace format: {samples_to_convert}/{total_samples} samples"
    )

    rows: list[DatasetRowDict] = []
    for i, sample in enumerate(task.dataset):
        if max_samples is not None and i >= max_samples:
            logger.debug(
                f"Reached max_samples limit ({max_samples}), stopping conversion"
            )
            break

        # Auto-generate ID if not set (some datasets like GSM8K don't set IDs)
        if sample.id is None:
            logger.debug(f"Auto-generating ID for sample {i}")
            sample = Sample(
                input=sample.input,
                target=sample.target,
                id=str(i),
                choices=sample.choices,
                metadata=sample.metadata,
                sandbox=sample.sandbox,
                files=sample.files,
                setup=sample.setup,
            )

        rows.append(
            _run_async_in_thread(
                sample_to_row(sample, task, task_name, additional_system_content)
            )
        )

    logger.info(f"Dataset conversion complete: {len(rows)} samples converted")
    # Cast to Any for HFDataset.from_list which has loose typing
    return HFDataset.from_list(rows)  # type: ignore[arg-type]
