"""Tests for env module."""

from typing import Any

import pytest
from datasets import Dataset as HFDataset
from inspect_ai.scorer import Scorer
from pytest_mock import MockerFixture
from tinker_cookbook.renderers import Message, TextPart, ThinkingPart

from inspect_tinker_bridge import env
from inspect_tinker_bridge.types import MessageDict, SampleInfoDict


class FakeRenderer:
    """Fake Renderer for testing without a real tokenizer.

    Mimics thinking-model renderers (Qwen3, GptOss, DeepSeek, KimiK2) that extract
    reasoning_content as a separate field. The base Renderer wraps thinking in
    <think> tags within content instead - this fake specifically tests the
    reasoning_content extraction path.
    """

    def build_generation_prompt(self, messages: list[Message]) -> object:
        """Return a fake ModelInput."""

        class FakeModelInput:
            def __init__(self, messages: list[Message]) -> None:
                self.messages = messages

        return FakeModelInput(messages)

    def get_stop_sequences(self) -> list[str]:
        return ["\n\nUser:"]

    def parse_response(self, action: list[int]) -> tuple[Message, bool]:
        return Message(role="assistant", content="4"), True

    def to_openai_message(self, m: Message) -> dict[str, Any]:
        """Convert a Message to OpenAI API format with reasoning_content extraction.

        Mimics thinking-model renderers that extract thinking into reasoning_content,
        not the base Renderer which wraps thinking in <think> tags.
        """
        result: dict[str, Any] = {"role": m["role"]}

        content = m["content"]
        if isinstance(content, str):
            result["content"] = content
        else:
            # Extract thinking into reasoning_content, text into content
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            for p in content:
                if p["type"] == "thinking":
                    thinking_parts.append(p["thinking"])
                elif p["type"] == "text":
                    text_parts.append(p["text"])
            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["reasoning_content"] = "".join(thinking_parts)

        # Handle tool_calls
        if "tool_calls" in m and m["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in m["tool_calls"]
            ]

        # Handle tool response fields
        if "tool_call_id" in m:
            result["tool_call_id"] = m["tool_call_id"]
        if "name" in m:
            result["name"] = m["name"]

        return result


class TestInspectEnv:
    """Tests for InspectEnv class."""

    def test_init_stores_parameters(
        self, sample_info: SampleInfoDict, prompt_messages: list[MessageDict]
    ) -> None:
        """Test that __init__ stores all parameters correctly."""
        renderer = FakeRenderer()
        scorers: list[Scorer] = []

        e = env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=renderer,  # type: ignore[arg-type]
            scorers=scorers,
            env_type="single_turn",
            max_turns=5,
            task_name="test",
        )

        assert e.sample_info == sample_info
        assert e.prompt_messages == prompt_messages
        assert e.answer == "4"
        assert e.env_type == "single_turn"
        assert e.max_turns == 5
        assert e.task_name == "test"
        assert e.current_turn == 0
        assert e.sandbox_instance is None

    @pytest.mark.asyncio
    async def test_initial_observation_builds_prompt(
        self, sample_info: SampleInfoDict, prompt_messages: list[MessageDict]
    ) -> None:
        """Test that initial_observation returns tokenized prompt."""
        renderer = FakeRenderer()

        e = env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=renderer,  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
        )

        _, stop = await e.initial_observation()

        assert stop == ["\n\nUser:"]
        assert len(e.conversation) == 1
        assert e.conversation[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_single_turn_ends_after_one_step(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> None:
        """Test that single-turn env ends after one step."""
        renderer = FakeRenderer()

        # Mock _compute_reward to return a simple reward
        mocker.patch.object(
            env.InspectEnv,
            "_compute_reward",
            return_value=(1.0, {"scorer_0": 1.0}),
        )

        e = env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=renderer,  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
        )

        await e.initial_observation()
        result = await e.step([1, 2, 3])  # Fake token action

        assert result.episode_done is True
        assert result.reward == 1.0
        assert result.metrics["correct"] == 1.0


class TestInspectEnvGroupBuilder:
    """Tests for InspectEnvGroupBuilder class."""

    @pytest.mark.asyncio
    async def test_make_envs_creates_correct_count(self) -> None:
        """Test that make_envs creates the correct number of environments."""
        call_count = 0

        def env_thunk() -> env.InspectEnv:
            nonlocal call_count
            call_count += 1
            return object()  # type: ignore[return-value]

        builder = env.InspectEnvGroupBuilder(
            env_thunk=env_thunk,
            num_envs=4,
            dataset_name="test",
        )

        envs = await builder.make_envs()

        assert len(envs) == 4
        assert call_count == 4

    def test_logging_tags_returns_dataset_name(self) -> None:
        """Test that logging_tags returns the dataset name."""
        builder = env.InspectEnvGroupBuilder(
            env_thunk=lambda: object(),  # type: ignore[return-value]
            num_envs=1,
            dataset_name="my_dataset",
        )

        assert builder.logging_tags() == ["my_dataset"]


class TestInspectRLDataset:
    """Tests for InspectRLDataset class."""

    def test_len_calculates_batches(self) -> None:
        """Test that __len__ returns correct number of batches."""
        # 10 samples with batch_size=3 = 4 batches (3+3+3+1)
        hf_dataset = HFDataset.from_list(
            [{"prompt": [], "answer": "", "info": {}} for _ in range(10)]
        )

        dataset = env.InspectRLDataset(
            hf_dataset=hf_dataset,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            sandbox_config=None,
            batch_size=3,
        )

        assert len(dataset) == 4

    def test_get_batch_returns_correct_count(self) -> None:
        """Test that get_batch returns correct number of builders."""
        hf_dataset = HFDataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": f"q{i}"}],
                    "answer": "",
                    "info": {},
                }
                for i in range(5)
            ]
        )

        dataset = env.InspectRLDataset(
            hf_dataset=hf_dataset,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            sandbox_config=None,
            batch_size=2,
            num_envs_per_group=3,
        )

        batch_0 = dataset.get_batch(0)
        batch_1 = dataset.get_batch(1)
        batch_2 = dataset.get_batch(2)

        assert len(batch_0) == 2  # First 2 problems
        assert len(batch_1) == 2  # Next 2 problems
        assert len(batch_2) == 1  # Last 1 problem

    def test_env_group_builder_has_correct_num_envs(self) -> None:
        """Test that created builders have correct num_envs."""
        hf_dataset = HFDataset.from_list(
            [{"prompt": [{"role": "user", "content": "q"}], "answer": "", "info": {}}]
        )

        dataset = env.InspectRLDataset(
            hf_dataset=hf_dataset,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            sandbox_config=None,
            num_envs_per_group=5,
        )

        builders = dataset.get_batch(0)
        assert len(builders) == 1
        builder = builders[0]
        assert isinstance(builder, env.InspectEnvGroupBuilder)
        assert builder.num_envs == 5


class TestMessageConversion:
    """Tests for message conversion functions."""

    @pytest.mark.parametrize(
        "content,expected_content,expected_reasoning",
        [
            pytest.param(
                "Simple answer",
                "Simple answer",
                None,
                id="string_content",
            ),
            pytest.param(
                [
                    ThinkingPart(type="thinking", thinking="Let me think..."),
                    TextPart(type="text", text="The answer is 4"),
                ],
                "The answer is 4",
                "Let me think...",
                id="thinking_content",
            ),
        ],
    )
    def test_message_to_dict_content_handling(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        content: Any,  # str or list of ContentPart dicts
        expected_content: str,
        expected_reasoning: str | None,
    ) -> None:
        """Test that _message_to_dict handles different content types correctly."""
        renderer = FakeRenderer()
        e = env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=renderer,  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
        )

        msg = Message(role="assistant", content=content)
        result = e._message_to_dict(msg)

        assert result["role"] == "assistant"
        assert result["content"] == expected_content
        if expected_reasoning is not None:
            assert result.get("reasoning_content") == expected_reasoning
        else:
            assert "reasoning_content" not in result

    def test_dict_to_message_basic(self) -> None:
        """Test that _dict_to_message converts basic fields correctly."""
        msg_dict = MessageDict(role="user", content="What is 2+2?")

        result = env.InspectEnv._dict_to_message(msg_dict)

        assert result["role"] == "user"
        assert result["content"] == "What is 2+2?"
