"""Tests for model_api module."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
import tinker
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)
from inspect_ai.tool import ToolCall as InspectToolCall, ToolInfo
from inspect_ai.tool._tool_params import ToolParams
from tinker_cookbook.renderers import (
    Message,
    TextPart,
    ThinkingPart,
    ToolCall as TinkerToolCall,
    ToolSpec,
)

from inspect_tinker_bridge import model_api


class TestToolInfoToToolSpec:
    @pytest.mark.parametrize(
        ("tool_info", "expected"),
        [
            pytest.param(
                ToolInfo(
                    name="bash",
                    description="Run a bash command",
                    parameters=ToolParams(
                        type="object",
                        properties={
                            "command": {"type": "string", "description": "The command"}
                        },
                        required=["command"],
                    ),
                ),
                ToolSpec(
                    name="bash",
                    description="Run a bash command",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The command"}
                        },
                        "required": ["command"],
                    },
                ),
                id="basic_tool",
            ),
            pytest.param(
                ToolInfo(name="submit", description="Submit answer"),
                ToolSpec(
                    name="submit",
                    description="Submit answer",
                    parameters={"type": "object", "properties": {}},
                ),
                id="no_params",
            ),
        ],
    )
    def test_conversion(self, tool_info: ToolInfo, expected: ToolSpec) -> None:
        result = model_api._tool_info_to_tool_spec(tool_info)
        assert result["name"] == expected["name"]
        assert result["description"] == expected["description"]
        assert result["parameters"]["type"] == expected["parameters"]["type"]
        assert (
            result["parameters"]["properties"] == expected["parameters"]["properties"]
        )


class TestTinkerMessageToAssistant:
    def test_text_content(self) -> None:
        msg = Message(role="assistant", content="Hello world")
        result = model_api._tinker_message_to_assistant(msg, "test-model")

        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.model == "test-model"

    def test_thinking_content(self) -> None:
        msg = Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="Let me think..."),
                TextPart(type="text", text="The answer is 4"),
            ],
        )
        result = model_api._tinker_message_to_assistant(msg, "test-model")

        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert isinstance(result.content[0], ContentReasoning)
        assert result.content[0].reasoning == "Let me think..."
        assert isinstance(result.content[1], ContentText)
        assert result.content[1].text == "The answer is 4"

    def test_tool_calls(self) -> None:
        msg = Message(
            role="assistant",
            content="Running command",
            tool_calls=[
                TinkerToolCall(
                    id="call_1",
                    function=TinkerToolCall.FunctionBody(
                        name="bash",
                        arguments='{"command": "echo hi"}',
                    ),
                )
            ],
        )
        result = model_api._tinker_message_to_assistant(msg, "test-model")

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_1"
        assert tc.function == "bash"
        assert tc.arguments == {"command": "echo hi"}

    def test_tool_call_id_fallback(self) -> None:
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[
                TinkerToolCall(
                    id=None,
                    function=TinkerToolCall.FunctionBody(
                        name="bash", arguments='{"command": "ls"}'
                    ),
                ),
                TinkerToolCall(
                    id=None,
                    function=TinkerToolCall.FunctionBody(
                        name="python", arguments='{"code": "print(1)"}'
                    ),
                ),
            ],
        )
        result = model_api._tinker_message_to_assistant(msg, "m")

        assert result.tool_calls is not None
        assert result.tool_calls[0].id == "tc_0"
        assert result.tool_calls[1].id == "tc_1"


class TestMapStopReason:
    @pytest.mark.parametrize(
        ("tinker_reason", "tool_calls", "expected"),
        [
            pytest.param("stop", None, "stop", id="stop_no_tools"),
            pytest.param(
                "stop",
                [InspectToolCall(id="1", function="bash", arguments={})],
                "tool_calls",
                id="stop_with_tools",
            ),
            pytest.param("length", None, "max_tokens", id="length_no_tools"),
            pytest.param(
                "length",
                [InspectToolCall(id="1", function="bash", arguments={})],
                "max_tokens",
                id="length_with_tools",
            ),
        ],
    )
    def test_mapping(
        self,
        tinker_reason: str,
        tool_calls: list[InspectToolCall] | None,
        expected: str,
    ) -> None:
        assert model_api._map_stop_reason(tinker_reason, tool_calls) == expected


class FakeModelInput:
    """Minimal fake for tinker.ModelInput."""

    def __init__(self, token_count: int = 10) -> None:
        self.length = token_count


class FakeRenderer:
    """Fake Renderer for testing TinkerSamplingAPI."""

    def __init__(
        self,
        parse_result: Message | None = None,
        input_token_count: int = 10,
    ) -> None:
        self._parse_result = parse_result or Message(
            role="assistant", content="test response"
        )
        self._input_token_count = input_token_count
        self.build_generation_prompt_calls: list[list[Message]] = []
        self.create_tools_calls: list[tuple[list[ToolSpec], str]] = []

    def build_generation_prompt(
        self, messages: list[Message], **kwargs: Any
    ) -> FakeModelInput:
        self.build_generation_prompt_calls.append(messages)
        return FakeModelInput(self._input_token_count)

    def get_stop_sequences(self) -> list[str]:
        return ["<|end|>"]

    def parse_response(self, tokens: list[int]) -> tuple[Message, bool]:
        return self._parse_result, True

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        self.create_tools_calls.append((tools, system_prompt))
        tool_names = ", ".join(t["name"] for t in tools)
        content = f"Tools: {tool_names}"
        if system_prompt:
            content = f"{system_prompt}\n{content}"
        return [Message(role="system", content=content)]


def _make_sample_response(
    tokens: list[int] | None = None, stop_reason: str = "stop"
) -> tinker.SampleResponse:
    return tinker.SampleResponse(
        sequences=[
            tinker.SampledSequence(
                tokens=tokens or [1, 2, 3],
                stop_reason=stop_reason,  # pyright: ignore[reportArgumentType]
            )
        ]
    )


def _make_api(
    renderer: FakeRenderer | None = None,
    sampling_client: Any = None,
    model_name: str = "test/model",
) -> model_api.TinkerSamplingAPI:
    if renderer is None:
        renderer = FakeRenderer()
    if sampling_client is None:
        sampling_client = AsyncMock()
        sampling_client.sample_async.return_value = _make_sample_response()
    return model_api.TinkerSamplingAPI(
        model_name=model_name,
        renderer=renderer,
        sampling_client=sampling_client,
    )


class TestTinkerSamplingAPIGenerate:
    @pytest.mark.asyncio
    async def test_simple_generation(self) -> None:
        renderer = FakeRenderer(
            parse_result=Message(role="assistant", content="The answer is 4"),
            input_token_count=15,
        )
        client = AsyncMock()
        client.sample_async.return_value = _make_sample_response(
            tokens=[10, 20, 30, 40, 50]
        )
        api = _make_api(renderer=renderer, sampling_client=client)

        result = await api.generate(
            input=[ChatMessageUser(content="What is 2+2?")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(max_tokens=100),
        )

        assert result.choices[0].message.content == "The answer is 4"
        assert result.choices[0].stop_reason == "stop"
        assert result.usage is not None
        assert result.usage.input_tokens == 15
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 20
        assert result.model == "test/model"

    @pytest.mark.asyncio
    async def test_tool_injection(self) -> None:
        renderer = FakeRenderer()
        client = AsyncMock()
        client.sample_async.return_value = _make_sample_response()
        api = _make_api(renderer=renderer, sampling_client=client)

        tool_info = ToolInfo(
            name="bash",
            description="Run bash",
            parameters=ToolParams(
                type="object",
                properties={"command": {"type": "string"}},
                required=["command"],
            ),
        )

        await api.generate(
            input=[
                ChatMessageSystem(content="You are helpful"),
                ChatMessageUser(content="Run ls"),
            ],
            tools=[tool_info],
            tool_choice="auto",
            config=GenerateConfig(),
        )

        assert len(renderer.create_tools_calls) == 1
        tools_arg, system_arg = renderer.create_tools_calls[0]
        assert tools_arg[0]["name"] == "bash"
        assert system_arg == "You are helpful"

    @pytest.mark.asyncio
    async def test_tool_choice_none_skips_injection(self) -> None:
        renderer = FakeRenderer()
        client = AsyncMock()
        client.sample_async.return_value = _make_sample_response()
        api = _make_api(renderer=renderer, sampling_client=client)

        tool_info = ToolInfo(name="bash", description="Run bash")

        await api.generate(
            input=[ChatMessageUser(content="hello")],
            tools=[tool_info],
            tool_choice="none",
            config=GenerateConfig(),
        )

        assert len(renderer.create_tools_calls) == 0

    @pytest.mark.asyncio
    async def test_config_mapping(self) -> None:
        client = AsyncMock()
        client.sample_async.return_value = _make_sample_response()
        api = _make_api(sampling_client=client)

        await api.generate(
            input=[ChatMessageUser(content="hi")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                seed=42,
            ),
        )

        call_args = client.sample_async.call_args
        params: tinker.SamplingParams = call_args.kwargs["sampling_params"]
        assert params.max_tokens == 256
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.seed == 42
        assert params.stop == ["<|end|>"]

    @pytest.mark.asyncio
    async def test_config_defaults(self) -> None:
        client = AsyncMock()
        client.sample_async.return_value = _make_sample_response()
        api = _make_api(sampling_client=client)

        await api.generate(
            input=[ChatMessageUser(content="hi")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
        )

        call_args = client.sample_async.call_args
        params: tinker.SamplingParams = call_args.kwargs["sampling_params"]
        assert params.max_tokens is None
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.seed is None

    @pytest.mark.asyncio
    async def test_tool_calls_stop_reason(self) -> None:
        renderer = FakeRenderer(
            parse_result=Message(
                role="assistant",
                content="",
                tool_calls=[
                    TinkerToolCall(
                        id="tc_0",
                        function=TinkerToolCall.FunctionBody(
                            name="bash", arguments='{"command": "ls"}'
                        ),
                    )
                ],
            )
        )
        client = AsyncMock()
        client.sample_async.return_value = _make_sample_response()
        api = _make_api(renderer=renderer, sampling_client=client)

        result = await api.generate(
            input=[ChatMessageUser(content="list files")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
        )

        assert result.choices[0].stop_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None

    @pytest.mark.asyncio
    async def test_renderer_not_implemented_tools(self) -> None:
        """When renderer doesn't support tools, gracefully skip and re-insert system prompt."""

        class NoToolsRenderer(FakeRenderer):
            def create_conversation_prefix_with_tools(
                self, tools: list[ToolSpec], system_prompt: str = ""
            ) -> list[Message]:
                raise NotImplementedError

        renderer = NoToolsRenderer()
        client = AsyncMock()
        client.sample_async.return_value = _make_sample_response()
        api = _make_api(renderer=renderer, sampling_client=client)

        tool_info = ToolInfo(name="bash", description="Run bash")

        await api.generate(
            input=[
                ChatMessageSystem(content="Be helpful"),
                ChatMessageUser(content="hi"),
            ],
            tools=[tool_info],
            tool_choice="auto",
            config=GenerateConfig(),
        )

        messages = renderer.build_generation_prompt_calls[0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"
