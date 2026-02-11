"""Tests for env module."""

import logging
from typing import Any, Literal

import pytest
from datasets import Dataset as HFDataset
from inspect_ai.scorer import Scorer
from inspect_ai.util import OutputLimitExceededError
from pytest_mock import MockerFixture
from tinker_cookbook.renderers import (
    Message,
    TextPart,
    ThinkingPart,
    ToolCall as TinkerToolCall,
    ToolSpec,
)

from inspect_tinker_bridge import env, sandbox as sandbox_module, truncation
from inspect_tinker_bridge.env import DEFAULT_TOOL_TIMEOUT, MAX_TOOL_TIMEOUT
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

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        tool_names = ", ".join(t["name"] for t in tools)
        content = f"Tools: {tool_names}"
        if system_prompt:
            content = f"{system_prompt}\n{content}"
        return [Message(role="system", content=content)]

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
            task_sandbox_config=None,
            sandbox_init_timeout=120,
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
            task_sandbox_config=None,
            sandbox_init_timeout=120,
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
            task_sandbox_config=None,
            sandbox_init_timeout=120,
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


class TestInspectRLDatasetShuffle:
    """Tests for InspectRLDataset shuffle functionality."""

    @pytest.fixture
    def hf_dataset_10(self) -> HFDataset:
        """Create a 10-sample HF dataset for shuffle tests."""
        return HFDataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": f"q{i}"}],
                    "answer": str(i),
                    "info": {"inspect_sample_id": f"sample_{i}"},
                }
                for i in range(10)
            ]
        )

    def _get_batch_sample_ids(
        self, dataset: env.InspectRLDataset, batch_index: int
    ) -> list[str | int | None]:
        """Extract sample IDs from a batch for comparison."""
        batch = dataset.get_batch(batch_index)
        return [
            builder.env_thunk().sample_info.get("inspect_sample_id")
            for builder in batch
            if isinstance(builder, env.InspectEnvGroupBuilder)
        ]

    @pytest.mark.parametrize(
        ("shuffle", "seed", "expected_different"),
        [
            pytest.param(False, None, False, id="no_shuffle_same_order"),
            pytest.param(True, 42, True, id="shuffle_changes_order"),
        ],
    )
    def test_shuffle_changes_batch_order(
        self,
        hf_dataset_10: HFDataset,
        shuffle: bool,
        seed: int | None,
        expected_different: bool,
    ) -> None:
        """Test that shuffle parameter changes batch order."""
        # Create unshuffled dataset for comparison
        unshuffled = env.InspectRLDataset(
            hf_dataset=hf_dataset_10,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            task_sandbox_config=None,
            sandbox_init_timeout=120,
            batch_size=10,
            shuffle=False,
        )

        # Create test dataset with specified shuffle settings
        shuffled = env.InspectRLDataset(
            hf_dataset=hf_dataset_10,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            task_sandbox_config=None,
            sandbox_init_timeout=120,
            batch_size=10,
            shuffle=shuffle,
            shuffle_seed=seed,
        )

        unshuffled_ids = self._get_batch_sample_ids(unshuffled, 0)
        shuffled_ids = self._get_batch_sample_ids(shuffled, 0)

        if expected_different:
            assert unshuffled_ids != shuffled_ids
        else:
            assert unshuffled_ids == shuffled_ids

    def test_shuffle_reproducible_with_seed(
        self,
        hf_dataset_10: HFDataset,
    ) -> None:
        """Test that shuffle is reproducible with the same seed."""
        dataset1 = env.InspectRLDataset(
            hf_dataset=hf_dataset_10,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            task_sandbox_config=None,
            sandbox_init_timeout=120,
            batch_size=10,
            shuffle=True,
            shuffle_seed=42,
        )

        dataset2 = env.InspectRLDataset(
            hf_dataset=hf_dataset_10,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            task_sandbox_config=None,
            sandbox_init_timeout=120,
            batch_size=10,
            shuffle=True,
            shuffle_seed=42,
        )

        ids1 = self._get_batch_sample_ids(dataset1, 0)
        ids2 = self._get_batch_sample_ids(dataset2, 0)

        assert ids1 == ids2

    def test_shuffle_different_per_epoch(
        self,
        hf_dataset_10: HFDataset,
    ) -> None:
        """Test that different epochs have different shuffle orders."""
        dataset = env.InspectRLDataset(
            hf_dataset=hf_dataset_10,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="single_turn",
            max_turns=1,
            task_sandbox_config=None,
            sandbox_init_timeout=120,
            batch_size=10,
            num_epochs=2,
            shuffle=True,
            shuffle_seed=42,
        )

        # batch_size=10 with 10 samples = 1 batch per epoch
        ids_epoch_0 = self._get_batch_sample_ids(dataset, 0)
        ids_epoch_1 = self._get_batch_sample_ids(dataset, 1)

        assert ids_epoch_0 != ids_epoch_1


def _make_tool_call(name: str, arguments: str, tool_id: str = "tc_0") -> TinkerToolCall:
    """Helper to create a TinkerToolCall for testing."""
    return TinkerToolCall(
        function=TinkerToolCall.FunctionBody(name=name, arguments=arguments),
        id=tool_id,
    )


class TestSetTimeout:
    """Tests for set_timeout tool handler."""

    def _make_env(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        tool_timeout: int = DEFAULT_TOOL_TIMEOUT,
    ) -> env.InspectEnv:
        return env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="multi_turn",
            tool_timeout=tool_timeout,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("timeout_val", "expected_stored"),
        [
            pytest.param(600, 600, id="int_value"),
            pytest.param(600.0, 600, id="float_cast_to_int"),
            pytest.param(1, 1, id="minimum_valid"),
            pytest.param(
                MAX_TOOL_TIMEOUT + 1000, MAX_TOOL_TIMEOUT, id="clamped_to_max"
            ),
        ],
    )
    async def test_set_timeout_valid_values(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        timeout_val: int | float,
        expected_stored: int,
    ) -> None:
        e = self._make_env(sample_info, prompt_messages)
        tc = _make_tool_call("set_timeout", f'{{"timeout": {timeout_val}}}')

        results = await e._execute_tools([tc])

        assert len(results) == 1
        assert results[0]["content"] == f"Timeout set to {expected_stored}"
        assert e._current_tool_timeout == expected_stored

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "args_json",
        [
            pytest.param('{"timeout": 0}', id="zero"),
            pytest.param('{"timeout": -1}', id="negative"),
            pytest.param('{"timeout": "not_a_number"}', id="string_value"),
            pytest.param("{}", id="missing_key"),
            pytest.param('{"timeout": true}', id="bool_true"),
            pytest.param('{"timeout": false}', id="bool_false"),
        ],
    )
    async def test_set_timeout_invalid_values(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        args_json: str,
    ) -> None:
        original = DEFAULT_TOOL_TIMEOUT
        e = self._make_env(sample_info, prompt_messages)
        tc = _make_tool_call("set_timeout", args_json)

        results = await e._execute_tools([tc])

        assert len(results) == 1
        assert (
            f"Invalid set_timeout function call, timeout remains {original} seconds"
            in results[0]["content"]
        )
        assert e._current_tool_timeout == original

    @pytest.mark.asyncio
    async def test_set_timeout_malformed_json(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
    ) -> None:
        """Malformed JSON returns generic JSON error, not set_timeout-specific error."""
        e = self._make_env(sample_info, prompt_messages)
        tc = _make_tool_call("set_timeout", "invalid json")

        results = await e._execute_tools([tc])

        assert len(results) == 1
        assert "Invalid JSON in tool arguments" in results[0]["content"]
        assert e._current_tool_timeout == DEFAULT_TOOL_TIMEOUT

    @pytest.mark.asyncio
    async def test_set_timeout_sub_one_float_floors_to_one(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
    ) -> None:
        """Floats between 0 and 1 are floored to 1, not truncated to 0."""
        e = self._make_env(sample_info, prompt_messages)
        tc = _make_tool_call("set_timeout", '{"timeout": 0.5}')

        results = await e._execute_tools([tc])

        assert results[0]["content"] == "Timeout set to 1"
        assert e._current_tool_timeout == 1

    @pytest.mark.asyncio
    async def test_set_timeout_applied_to_execution(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> None:
        """Verify set_timeout changes the timeout used in sandbox exec calls."""
        e = self._make_env(sample_info, prompt_messages)
        # Give the env a fake sandbox instance
        e.sandbox_instance = mocker.MagicMock()
        e.sandbox_instance.environments = {"default": mocker.MagicMock()}

        mock_exec = mocker.patch.object(
            sandbox_module,
            "exec_in_sandbox",
            return_value=mocker.MagicMock(stdout="ok", stderr=""),
        )

        # Set timeout to 300, then run bash
        set_tc = _make_tool_call("set_timeout", '{"timeout": 300}', tool_id="tc_0")
        bash_tc = _make_tool_call("bash", '{"command": "echo hi"}', tool_id="tc_1")

        await e._execute_tools([set_tc, bash_tc])

        mock_exec.assert_called_once()
        assert mock_exec.call_args[1]["timeout"] == 300

    @pytest.mark.asyncio
    async def test_timeout_lifecycle(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
    ) -> None:
        """Verify per-rollout state is set at init and reset by initial_observation."""
        e = self._make_env(sample_info, prompt_messages, tool_timeout=600)

        # Available immediately after __init__
        assert e._current_tool_timeout == 600
        assert e._submitted is False
        assert e.current_turn == 0

        # Simulate mid-rollout state
        e._current_tool_timeout = 42
        e._submitted = True
        e.current_turn = 5

        # initial_observation resets all per-rollout state
        await e.initial_observation()
        assert e._current_tool_timeout == 600
        assert e._submitted is False
        assert e.current_turn == 0


class FakeRendererNoTools(FakeRenderer):
    """Renderer that does not support tool definitions."""

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        raise NotImplementedError


class TestToolDefinitionInjection:
    """Tests for tool definition injection into multi-turn prompts."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("env_type", "has_system_msg", "expect_prefix"),
        [
            pytest.param("multi_turn", True, True, id="multiturn_with_system_msg"),
            pytest.param("multi_turn", False, True, id="multiturn_no_system_msg"),
            pytest.param("single_turn", True, False, id="single_turn_no_injection"),
        ],
    )
    async def test_tool_definition_injection(
        self,
        sample_info: SampleInfoDict,
        env_type: Literal["single_turn", "multi_turn"],
        has_system_msg: bool,
        expect_prefix: bool,
    ) -> None:
        renderer = FakeRenderer()

        prompt_messages: list[MessageDict] = []
        if has_system_msg:
            prompt_messages.append(
                MessageDict(role="system", content="You are helpful.")
            )
        prompt_messages.append(MessageDict(role="user", content="What is 2 + 2?"))

        e = env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=renderer,  # type: ignore[arg-type]
            scorers=[],
            env_type=env_type,
            max_turns=10,
        )

        await e.initial_observation()

        if expect_prefix:
            assert e.conversation[0]["role"] == "system"
            content = e.conversation[0]["content"]
            assert isinstance(content, str)
            assert "bash" in content
            assert "python" in content
            assert "submit" in content
            assert "set_timeout" in content

            if has_system_msg:
                assert "You are helpful." in content
                assert all(m["role"] != "system" for m in e.conversation[1:])
            assert e.conversation[-1]["role"] == "user"
            assert e.conversation[-1]["content"] == "What is 2 + 2?"
        else:
            if has_system_msg:
                assert e.conversation[0]["role"] == "system"
                content = e.conversation[0]["content"]
                assert isinstance(content, str)
                assert "bash" not in content
            else:
                assert e.conversation[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_unsupported_renderer_logs_warning(
        self,
        sample_info: SampleInfoDict,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        renderer = FakeRendererNoTools()
        prompt_messages = [
            MessageDict(role="system", content="System prompt"),
            MessageDict(role="user", content="Hello"),
        ]

        e = env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=renderer,  # type: ignore[arg-type]
            scorers=[],
            env_type="multi_turn",
            max_turns=10,
        )

        with caplog.at_level(logging.WARNING):
            await e.initial_observation()

        assert "does not support tool definitions" in caplog.text
        # Conversation should be unchanged
        assert len(e.conversation) == 2
        assert e.conversation[0]["role"] == "system"
        assert e.conversation[1]["role"] == "user"


class TestToolOutputTruncation:
    """Tests for tool output truncation in _execute_tools."""

    def _make_env(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        max_tool_output: int = truncation.DEFAULT_MAX_TOOL_OUTPUT,
    ) -> env.InspectEnv:
        return env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="multi_turn",
            max_tool_output=max_tool_output,
        )

    @pytest.mark.asyncio
    async def test_short_output_not_truncated(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> None:
        e = self._make_env(sample_info, prompt_messages, max_tool_output=1000)
        e.sandbox_instance = mocker.MagicMock()
        e.sandbox_instance.environments = {"default": mocker.MagicMock()}

        mocker.patch.object(
            sandbox_module,
            "exec_in_sandbox",
            return_value=mocker.MagicMock(stdout="short output", stderr=""),
        )

        tc = _make_tool_call("bash", '{"command": "echo hi"}')
        results = await e._execute_tools([tc])

        assert results[0]["content"] == "short output"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("bash", id="bash_tool"),
            pytest.param("python", id="python_tool"),
        ],
    )
    async def test_long_output_truncated_with_markers(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
        tool_name: str,
    ) -> None:
        e = self._make_env(sample_info, prompt_messages, max_tool_output=50)
        e.sandbox_instance = mocker.MagicMock()
        e.sandbox_instance.environments = {"default": mocker.MagicMock()}

        mocker.patch.object(
            sandbox_module,
            "exec_in_sandbox",
            return_value=mocker.MagicMock(stdout="x" * 200, stderr=""),
        )

        args_key = "command" if tool_name == "bash" else "code"
        tc = _make_tool_call(tool_name, f'{{"{args_key}": "echo hi"}}')
        results = await e._execute_tools([tc])

        content = results[0]["content"]
        assert f"call to {tool_name} was too long" in content
        assert "<START_TOOL_OUTPUT>" in content
        assert "<END_TOOL_OUTPUT>" in content

    @pytest.mark.asyncio
    async def test_stderr_combined_output_truncated(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> None:
        """Combined stdout+stderr exceeding limit gets truncated."""
        e = self._make_env(sample_info, prompt_messages, max_tool_output=50)
        e.sandbox_instance = mocker.MagicMock()
        e.sandbox_instance.environments = {"default": mocker.MagicMock()}

        mocker.patch.object(
            sandbox_module,
            "exec_in_sandbox",
            return_value=mocker.MagicMock(stdout="o" * 100, stderr="e" * 100),
        )

        tc = _make_tool_call("bash", '{"command": "echo hi"}')
        results = await e._execute_tools([tc])

        content = results[0]["content"]
        assert "call to bash was too long" in content
        assert "<START_TOOL_OUTPUT>" in content

    @pytest.mark.asyncio
    async def test_max_tool_output_parameter_flows(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
    ) -> None:
        """max_tool_output flows from InspectRLDataset through to InspectEnv."""
        hf_dataset = HFDataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "q"}],
                    "answer": "",
                    "info": {},
                }
            ]
        )

        dataset = env.InspectRLDataset(
            hf_dataset=hf_dataset,
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="multi_turn",
            max_turns=10,
            task_sandbox_config=None,
            sandbox_init_timeout=120,
            max_tool_output=8192,
        )

        builders = dataset.get_batch(0)
        builder = builders[0]
        assert isinstance(builder, env.InspectEnvGroupBuilder)
        created_env = builder.env_thunk()
        assert created_env.max_tool_output == 8192


class TestSandboxExecErrorHandling:
    """Tests that sandbox execution errors are returned to the model, not raised."""

    def _make_env(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> env.InspectEnv:
        e = env.InspectEnv(
            sample_info=sample_info,
            prompt_messages=prompt_messages,
            answer="4",
            renderer=FakeRenderer(),  # type: ignore[arg-type]
            scorers=[],
            env_type="multi_turn",
        )
        e.sandbox_instance = mocker.MagicMock()
        e.sandbox_instance.environments = {"default": mocker.MagicMock()}
        return e

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("exception", "expected_fragment"),
        [
            pytest.param(
                TimeoutError(),
                "Command timed out after",
                id="timeout",
            ),
            pytest.param(
                PermissionError("cannot execute"),
                "Permission denied: cannot execute",
                id="permission",
            ),
            pytest.param(
                UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
                "Failed to decode command output",
                id="unicode_decode",
            ),
            pytest.param(
                OutputLimitExceededError("10 MiB", None),
                "Command output exceeded size limit",
                id="output_limit_no_content",
            ),
            pytest.param(
                OutputLimitExceededError("10 MiB", "partial stdout here"),
                "Command output exceeded sandbox limit",
                id="output_limit_with_content",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("bash", id="bash"),
            pytest.param("python", id="python"),
        ],
    )
    async def test_exec_error_returned_as_tool_error(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
        exception: Exception,
        expected_fragment: str,
        tool_name: str,
    ) -> None:
        e = self._make_env(sample_info, prompt_messages, mocker)
        mocker.patch.object(sandbox_module, "exec_in_sandbox", side_effect=exception)

        args_key = "command" if tool_name == "bash" else "code"
        tc = _make_tool_call(tool_name, f'{{"{args_key}": "echo hi"}}')
        results = await e._execute_tools([tc])

        assert len(results) == 1
        assert results[0]["role"] == "tool"
        content = results[0]["content"]
        assert isinstance(content, str)
        assert content.startswith("Error: ")
        assert expected_fragment in content

    @pytest.mark.asyncio
    async def test_timeout_error_suggests_set_timeout(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> None:
        """Timeout message tells the model how to fix it."""
        e = self._make_env(sample_info, prompt_messages, mocker)
        e._current_tool_timeout = 120
        mocker.patch.object(
            sandbox_module, "exec_in_sandbox", side_effect=TimeoutError()
        )

        tc = _make_tool_call("bash", '{"command": "sleep 999"}')
        results = await e._execute_tools([tc])

        content = results[0]["content"]
        assert isinstance(content, str)
        assert "120 seconds" in content
        assert "set_timeout" in content

    @pytest.mark.asyncio
    async def test_output_limit_includes_truncated_content(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> None:
        """OutputLimitExceededError surfaces the truncated output to the model."""
        e = self._make_env(sample_info, prompt_messages, mocker)
        mocker.patch.object(
            sandbox_module,
            "exec_in_sandbox",
            side_effect=OutputLimitExceededError("10 MiB", "first 100 bytes of output"),
        )

        tc = _make_tool_call("bash", '{"command": "cat huge_file"}')
        results = await e._execute_tools([tc])

        content = results[0]["content"]
        assert isinstance(content, str)
        assert "first 100 bytes of output" in content
        assert "exceeded sandbox limit" in content

    @pytest.mark.asyncio
    async def test_exec_error_does_not_block_subsequent_tools(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        mocker: MockerFixture,
    ) -> None:
        """An error on one tool call doesn't prevent processing the next."""
        e = self._make_env(sample_info, prompt_messages, mocker)
        mocker.patch.object(
            sandbox_module,
            "exec_in_sandbox",
            side_effect=[
                TimeoutError(),
                mocker.MagicMock(stdout="ok", stderr=""),
            ],
        )

        tc_fail = _make_tool_call("bash", '{"command": "slow"}', tool_id="tc_0")
        tc_ok = _make_tool_call("bash", '{"command": "echo hi"}', tool_id="tc_1")
        results = await e._execute_tools([tc_fail, tc_ok])

        assert len(results) == 2
        assert "Error: " in results[0]["content"]
        assert results[1]["content"] == "ok"
