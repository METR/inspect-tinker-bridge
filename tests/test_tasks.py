"""Tests for tasks module."""

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match, model_graded_fact
from inspect_ai.solver import generate

from inspect_tinker_bridge import tasks


class TestLoadInspectTask:
    """Tests for load_inspect_task function."""

    def test_extracts_task_name(self) -> None:
        """Test that task name is extracted."""

        def task_fn() -> Task:
            return Task(
                dataset=[Sample(input="test", target="answer")],
                solver=generate(),
                scorer=match(),
                name="my_task",
            )

        info = tasks.load_inspect_task(task_fn)
        assert info.name == "my_task"

    def test_extracts_single_scorer(self) -> None:
        """Test extraction of a single scorer."""

        def task_fn() -> Task:
            return Task(
                dataset=[Sample(input="test", target="answer")],
                solver=generate(),
                scorer=match(),
                name="test",
            )

        info = tasks.load_inspect_task(task_fn)
        assert len(info.scorers) == 1

    def test_extracts_multiple_scorers(self) -> None:
        """Test extraction of multiple scorers."""

        def task_fn() -> Task:
            return Task(
                dataset=[Sample(input="test", target="answer")],
                solver=generate(),
                scorer=[match(), model_graded_fact()],
                name="test",
            )

        info = tasks.load_inspect_task(task_fn)
        assert len(info.scorers) == 2

    def test_extracts_sandbox_type_string(self) -> None:
        """Test extraction of sandbox type when specified as string."""

        def task_fn() -> Task:
            return Task(
                dataset=[Sample(input="test", target="answer")],
                solver=generate(),
                scorer=match(),
                name="test",
                sandbox="docker",
            )

        info = tasks.load_inspect_task(task_fn)
        assert info.sandbox_type == "docker"

    def test_no_sandbox(self) -> None:
        """Test that sandbox_type is None when not specified."""

        def task_fn() -> Task:
            return Task(
                dataset=[Sample(input="test", target="answer")],
                solver=generate(),
                scorer=match(),
                name="test",
            )

        info = tasks.load_inspect_task(task_fn)
        assert info.sandbox_type is None

    def test_passes_task_kwargs(self) -> None:
        """Test that kwargs are passed to task function."""

        def task_fn(custom_arg: str) -> Task:
            return Task(
                dataset=[Sample(input=custom_arg, target="answer")],
                solver=generate(),
                scorer=match(),
                name="test",
            )

        info = tasks.load_inspect_task(task_fn, custom_arg="custom_value")
        # Verify the custom arg was used by checking the dataset
        assert info.dataset[0].input == "custom_value"

    def test_extracts_metadata(self) -> None:
        """Test extraction of task metadata."""

        def task_fn() -> Task:
            return Task(
                dataset=[Sample(input="test", target="answer")],
                solver=generate(),
                scorer=match(),
                name="test",
                metadata={"version": "1.0"},
            )

        info = tasks.load_inspect_task(task_fn)
        assert info.metadata == {"version": "1.0"}


class TestSolverHasTools:
    """Tests for _solver_has_tools function."""

    @pytest.mark.parametrize(
        "solver_str,expected",
        [
            pytest.param("generate()", False, id="no_tools"),
            pytest.param("use_tools([bash()])", True, id="use_tools"),
            pytest.param("react(tools=[bash()])", True, id="react"),
            pytest.param("Chain([tool_use()])", True, id="tool_in_chain"),
        ],
    )
    def test_detects_tools(self, solver_str: str, expected: bool) -> None:
        """Test tool detection in solver strings."""

        class MockSolver:
            def __str__(self) -> str:
                return solver_str

        result = tasks._solver_has_tools(MockSolver())
        assert result == expected

    def test_none_solver(self) -> None:
        """Test that None solver returns False."""
        assert tasks._solver_has_tools(None) is False
