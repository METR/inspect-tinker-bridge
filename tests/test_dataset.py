"""Tests for dataset module."""

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_tinker_bridge.dataset import sample_to_row


@pytest.fixture
def minimal_task() -> Task:
    """A minimal Task for serialization tests."""
    return Task(
        dataset=[Sample(input="dummy", target="dummy", id="dummy")],
        solver=generate(),
        scorer=match(),
        name="test_task",
    )


class TestSampleSandboxSerialization:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("sandbox_input", "expected_serialized"),
        [
            pytest.param(None, None, id="no_sandbox"),
            pytest.param(
                SandboxEnvironmentSpec(type="docker", config="/path/to/compose.yaml"),
                ("docker", "/path/to/compose.yaml"),
                id="docker_with_config",
            ),
            pytest.param(
                SandboxEnvironmentSpec(type="docker", config=None),
                ("docker", None),
                id="docker_no_config",
            ),
            pytest.param(
                SandboxEnvironmentSpec(type="local", config=None),
                ("local", None),
                id="local_sandbox",
            ),
        ],
    )
    async def test_sandbox_serialization(
        self,
        sandbox_input: SandboxEnvironmentSpec | None,
        expected_serialized: tuple[str, str | None] | None,
        minimal_task: Task,
    ) -> None:
        """Test that SandboxEnvironmentSpec is properly serialized to tuple format."""
        sample = Sample(
            input="test input",
            target="test target",
            id="test-1",
            sandbox=sandbox_input,
        )
        row = await sample_to_row(sample, minimal_task, "test_task")
        assert row["info"].get("inspect_sandbox") == expected_serialized
