"""
Ground truth prompt generation by running Inspect's solver pipeline.

This module provides a reliable way to get the actual transformed messages
that would be presented to a model during evaluation, by running the
solver chain without model inference.
"""

from typing import Literal

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.model._generate_config import GenerateConfigArgs
from inspect_ai.scorer import Target
from inspect_ai.solver import Solver, TaskState
from typing_extensions import Unpack

# Solvers that require model inference and should stop execution
# This includes both simple generation and agentic solvers
GENERATION_SOLVERS = {
    "generate",
    "self_critique",
    "basic_agent",
    "basic_agent_loop",
    "submit_tool",
    "react",
}


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


def _get_solver_name(solver: Solver) -> str:
    """Extract the solver name from a solver function."""
    qualname = getattr(solver, "__qualname__", "")
    if ".<locals>." in qualname:
        return qualname.split(".<locals>.")[0]
    return qualname


async def _run_solvers_until_generate(
    solver: Solver | None,
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

    async def noop_generate(
        state: TaskState,
        tool_calls: Literal["loop", "single", "none"] = "loop",
        **kwargs: Unpack[GenerateConfigArgs],
    ) -> TaskState:
        return state

    # No solver
    if solver is None:
        return state

    # Get list of solvers - Chain has internal _solvers attribute
    solvers: list[Solver]
    if hasattr(solver, "_solvers"):
        # It's a Chain - access internal list
        solvers = solver._solvers  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
    else:
        # Single solver
        solvers = [solver]

    # Run each solver until we hit a generation solver
    for s in solvers:
        solver_name = _get_solver_name(s)

        # Stop before generation solvers
        if solver_name in GENERATION_SOLVERS:
            break

        # Run the solver
        state = await s(state, noop_generate)

    return state
