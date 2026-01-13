"""
Task introspection and loading utilities for Inspect AI tasks.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from inspect_ai import Task
from inspect_ai.dataset import Dataset
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

logger = logging.getLogger(__name__)


@dataclass
class InspectTaskInfo:
    """Holds introspected information about an Inspect task."""

    task: Task
    name: str
    dataset: Dataset
    scorers: list[Scorer]
    sandbox_type: str | None
    solver_has_tools: bool
    metadata: dict[str, object] = field(default_factory=dict)


def load_inspect_task(
    task_fn: Callable[..., Task],
    **task_kwargs: object,
) -> InspectTaskInfo:
    """
    Load an Inspect task and extract its components.

    Args:
        task_fn: A callable that returns an Inspect Task (e.g., apps from inspect_evals)
        **task_kwargs: Arguments to pass to the task function

    Returns:
        InspectTaskInfo with extracted task components
    """
    logger.debug(f"Loading task from {task_fn.__name__} with kwargs: {task_kwargs}")
    task = task_fn(**task_kwargs)
    logger.info(f"Task loaded: {task.name or 'unknown'}")

    # Extract sandbox type
    sandbox_type = None
    if task.sandbox is not None:
        if isinstance(task.sandbox, str):
            sandbox_type = task.sandbox
        elif hasattr(task.sandbox, "type"):
            sandbox_type = task.sandbox.type
        logger.debug(f"Extracted sandbox type: {sandbox_type}")

    # Extract scorers (normalize to list)
    # task.scorer can be Scorer | list[Scorer] | None at runtime
    scorers: list[Scorer] = []
    task_scorer = task.scorer
    if task_scorer is not None:
        if isinstance(task_scorer, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            scorers = task_scorer
        else:
            scorers = [task_scorer]
        logger.debug(f"Extracted {len(scorers)} scorer(s)")

    # Check if solver uses tools (simple heuristic for now)
    solver_has_tools = _solver_has_tools(task.solver)
    logger.debug(f"Solver has tools: {solver_has_tools}")

    task_info = InspectTaskInfo(
        task=task,
        name=task.name or "unknown",
        dataset=task.dataset,
        scorers=scorers,
        sandbox_type=sandbox_type,
        solver_has_tools=solver_has_tools,
        metadata=task.metadata or {},
    )
    logger.info(
        f"Task introspection complete: name={task_info.name}, "
        f"dataset_size={len(task_info.dataset)}, num_scorers={len(scorers)}"
    )
    return task_info


def _solver_has_tools(solver: Solver | None) -> bool:
    """Check if a solver chain includes tool usage."""
    if solver is None:
        return False

    solver_str = str(solver)
    tool_indicators = ["use_tools", "react", "tool", "bash", "python"]
    return any(indicator in solver_str.lower() for indicator in tool_indicators)
