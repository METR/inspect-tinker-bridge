"""
Shared test fixtures for inspect-tinker-bridge tests.
"""

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@pytest.fixture
def simple_sample() -> Sample:
    """A simple Sample for testing."""
    return Sample(
        input="What is 2 + 2?",
        target="4",
        id="test_sample_1",
        metadata={"difficulty": "easy"},
    )


@pytest.fixture
def sample_with_messages() -> Sample:
    """A Sample with ChatMessage input."""
    return Sample(
        input=[ChatMessageUser(content="What is the capital of France?")],
        target="Paris",
        id="test_sample_2",
    )


@pytest.fixture
def simple_task(simple_sample: Sample) -> Task:
    """A simple Task for testing."""
    return Task(
        dataset=[simple_sample],
        solver=generate(),
        scorer=match(),
        name="test_task",
    )


@pytest.fixture
def task_with_sandbox(simple_sample: Sample) -> Task:
    """A Task with sandbox configuration."""
    return Task(
        dataset=[simple_sample],
        solver=generate(),
        scorer=match(),
        name="sandbox_task",
        sandbox="docker",
    )


@pytest.fixture
def sample_info() -> dict[str, object]:
    """Sample info dict as stored in HF dataset."""
    return {
        "inspect_sample_id": "test_sample_1",
        "inspect_input_raw": "What is 2 + 2?",
        "inspect_target_raw": "4",
        "inspect_choices": None,
        "inspect_metadata": '{"difficulty": "easy"}',
        "inspect_sandbox": None,
        "inspect_files": None,
        "inspect_setup": None,
        "inspect_task_name": "test_task",
    }


@pytest.fixture
def prompt_messages() -> list[dict[str, str]]:
    """Prompt messages in dict format."""
    return [{"role": "user", "content": "What is 2 + 2?"}]
