"""
Ground truth prompt generation by running Inspect's solver pipeline.

This module provides a reliable way to get the actual transformed messages
that would be presented to a model during evaluation, by running the
solver chain without model inference.
"""

from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

# Solvers that require model inference and should stop execution
GENERATION_SOLVERS = {"generate", "self_critique"}


async def get_ground_truth_messages(task: Task, sample: Sample) -> list[ChatMessage]:
    """
    Run the solver pipeline on a sample to get the transformed messages.

    This executes all prompt-engineering solvers (system_message, prompt_template,
    multiple_choice, chain_of_thought, user_message, use_tools, etc.) WITHOUT
    calling the model. Execution stops before any generation solver.

    Args:
        task: The Inspect Task containing the solver chain
        sample: The Sample to process

    Returns:
        List of ChatMessage objects representing the final prompt
    """
    state = _sample_to_task_state(sample)
    state = await _run_solvers_until_generate(task.solver, state)
    return state.messages


def _sample_to_task_state(sample: Sample) -> TaskState:
    """Convert a Sample to an initial TaskState."""
    # Convert input to messages
    if isinstance(sample.input, str):
        messages: list[ChatMessage] = [ChatMessageUser(content=sample.input)]
    else:
        messages = list(sample.input)

    sample_id = sample.id if sample.id is not None else "0"
    return TaskState(
        model="ground-truth",  # type: ignore[arg-type]
        sample_id=sample_id,
        epoch=0,
        input=sample.input,
        messages=messages,
        target=Target(sample.target) if sample.target else Target(""),
        choices=sample.choices,
        metadata=sample.metadata or {},
    )


def _get_solver_name(solver: Any) -> str:
    """Extract the solver name from a solver function."""
    qualname = getattr(solver, "__qualname__", "")
    if ".<locals>." in qualname:
        return qualname.split(".<locals>.")[0]
    return qualname


async def _run_solvers_until_generate(
    solver: Any,
    state: TaskState,
) -> TaskState:
    """
    Run solvers sequentially, stopping before any generation solver.

    Args:
        solver: The task's solver (may be a Chain or single solver)
        state: Initial TaskState

    Returns:
        TaskState after all prompt-engineering solvers have run
    """

    async def noop_generate(state: TaskState, **kwargs: Any) -> TaskState:
        return state

    # Get list of solvers
    if hasattr(solver, "_solvers"):
        # It's a Chain
        solvers = solver._solvers
    elif solver is not None:
        # Single solver
        solvers = [solver]
    else:
        # No solver
        return state

    # Run each solver until we hit a generation solver
    for s in solvers:
        solver_name = _get_solver_name(s)

        # Stop before generation solvers
        if solver_name in GENERATION_SOLVERS:
            break

        # Run the solver
        state = await s(state, noop_generate)

    return state
