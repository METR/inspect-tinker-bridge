"""Tests for dataset module."""

import json

import pydantic
import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_tinker_bridge.dataset import sample_to_row


class _StubSandboxConfig(pydantic.BaseModel):
    image: str = "python:3.12"
    replicas: int = 1


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
            pytest.param(
                SandboxEnvironmentSpec(
                    type="k8s",
                    config=_StubSandboxConfig(image="python:3.12", replicas=2),
                ),
                ("k8s", '{"image":"python:3.12","replicas":2}'),
                id="basemodel_config",
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
        assert row["info"]["inspect_sandbox"] == expected_serialized


class TestEvalMetadataSerialization:
    @pytest.mark.asyncio
    async def test_task_metadata_serialized_as_eval_metadata(self) -> None:
        """Test that task.metadata is serialized into inspect_eval_metadata."""
        task = Task(
            dataset=[Sample(input="dummy", target="dummy", id="dummy")],
            solver=generate(),
            scorer=match(),
            name="test_task",
            metadata={"side_task_correctness_scorer_name": "side_task_correctness"},
        )
        sample = Sample(input="test input", target="test target", id="test-1")
        row = await sample_to_row(sample, task, "test_task")
        eval_metadata = json.loads(row["info"]["inspect_eval_metadata"])
        assert (
            eval_metadata["side_task_correctness_scorer_name"]
            == "side_task_correctness"
        )

    @pytest.mark.asyncio
    async def test_empty_task_metadata_serialized(self, minimal_task: Task) -> None:
        """Test that empty/None task.metadata serializes to empty dict."""
        sample = Sample(input="test input", target="test target", id="test-1")
        row = await sample_to_row(sample, minimal_task, "test_task")
        assert row["info"]["inspect_eval_metadata"] == "{}"
