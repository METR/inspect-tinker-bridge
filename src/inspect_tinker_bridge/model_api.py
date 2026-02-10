"""Inspect AI ModelAPI provider that routes through Tinker's sampling endpoint."""

import json
import logging
from typing import Any

import tinker
from inspect_ai._util.content import Content, ContentReasoning, ContentText
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatCompletionChoice,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_output import StopReason, as_stop_reason
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolCall as InspectToolCall, ToolChoice, ToolInfo
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import Message, ToolSpec

from inspect_tinker_bridge import utils

logger = logging.getLogger(__name__)


def _tool_info_to_tool_spec(tool_info: ToolInfo) -> ToolSpec:
    """Convert Inspect's ToolInfo to Tinker's ToolSpec."""
    return ToolSpec(
        name=tool_info.name,
        description=tool_info.description,
        parameters=tool_info.parameters.model_dump(exclude_none=True),
    )


def _tinker_message_to_assistant(
    message: Message, model_name: str
) -> ChatMessageAssistant:
    """Convert a parsed Tinker Message to Inspect's ChatMessageAssistant."""
    raw_content = message["content"]

    inspect_content: str | list[Content]
    if isinstance(raw_content, str):
        inspect_content = raw_content
    else:
        parts: list[Content] = []
        for part in raw_content:
            if part["type"] == "thinking":
                parts.append(ContentReasoning(reasoning=part["thinking"]))
            elif part["type"] == "text":
                parts.append(ContentText(text=part["text"]))
        inspect_content = parts

    tool_calls: list[InspectToolCall] | None = None
    raw_tool_calls: list[renderers.ToolCall] | None = message.get("tool_calls")
    if raw_tool_calls:
        # json.loads is safe here: all renderers validate JSON during tool call
        # extraction, routing invalid JSON to unparsed_tool_calls instead.
        tool_calls = [
            InspectToolCall(
                id=tc.id or f"tc_{i}",
                function=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for i, tc in enumerate(raw_tool_calls)
        ]

    return ChatMessageAssistant(
        content=inspect_content,
        tool_calls=tool_calls,
        model=model_name,
    )


def _map_stop_reason(
    tinker_reason: str,
    tool_calls: list[InspectToolCall] | None,
) -> StopReason:
    """Map Tinker stop reason to Inspect stop reason."""
    stop = as_stop_reason(tinker_reason)
    if stop == "stop" and tool_calls:
        return "tool_calls"
    return stop


@modelapi("tinker_sampling")
class TinkerSamplingAPI(ModelAPI):
    """Inspect AI ModelAPI that uses Tinker's sampling endpoint.

    Usage (CLI):
        inspect eval task.py -m tinker_sampling/Qwen/Qwen3-8B
        inspect eval task.py -m tinker_sampling/openai/gpt-oss-120b -M model_path=tinker://run-id/sampler_weights/000100

    Usage (programmatic):
        model = get_model("tinker_sampling/Qwen/Qwen3-8B")
        model = get_model("tinker_sampling/Qwen/Qwen3-8B", model_path="tinker://...")

    model_args:
        model_path: tinker path to finetuned checkpoint. If omitted, uses model_name as base model.
        renderer_name: override auto-detected renderer name. If omitted, auto-detected from model_name.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars, config)

        model_path: str | None = model_args.get("model_path")
        renderer_name: str | None = model_args.get("renderer_name")

        service_client = tinker.ServiceClient()

        self.sampling_client: tinker.SamplingClient = (
            service_client.create_sampling_client(model_path=model_path)
            if model_path
            else service_client.create_sampling_client(base_model=model_name)
        )

        tokenizer = self.sampling_client.get_tokenizer()

        if renderer_name is None:
            renderer_name = model_info.get_recommended_renderer_name(model_name)

        self.renderer: renderers.Renderer = renderers.get_renderer(
            renderer_name, tokenizer
        )

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        messages = utils.chat_messages_to_tinker(input)

        if tools and tool_choice != "none":
            tool_specs = [_tool_info_to_tool_spec(t) for t in tools]
            messages = self._inject_tools(messages, tool_specs)

        model_input = self.renderer.build_generation_prompt(messages)

        sampling_params = tinker.SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature if config.temperature is not None else 1.0,
            top_p=config.top_p if config.top_p is not None else 1.0,
            top_k=config.top_k if config.top_k is not None else -1,
            seed=config.seed,
            stop=self.renderer.get_stop_sequences(),
        )

        response = await self.sampling_client.sample_async(
            model_input, num_samples=1, sampling_params=sampling_params
        )

        sequence = response.sequences[0]
        parsed_message, valid = self.renderer.parse_response(sequence.tokens)
        if not valid:
            logger.warning(
                "Renderer failed to fully parse response (e.g. missing stop token); "
                "using best-effort message"
            )
        assistant_msg = _tinker_message_to_assistant(parsed_message, self.model_name)

        stop_reason = _map_stop_reason(sequence.stop_reason, assistant_msg.tool_calls)

        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(message=assistant_msg, stop_reason=stop_reason)
            ],
            usage=ModelUsage(
                input_tokens=model_input.length,
                output_tokens=len(sequence.tokens),
                total_tokens=model_input.length + len(sequence.tokens),
            ),
        )

    def _inject_tools(
        self, messages: list[Message], tool_specs: list[ToolSpec]
    ) -> list[Message]:
        """Inject tool definitions into the conversation prefix."""
        system_prompt = ""
        remaining = list(messages)

        if remaining and remaining[0]["role"] == "system":
            content = remaining.pop(0)["content"]
            assert isinstance(content, str), (
                f"Expected string system prompt, got {type(content)}"
            )
            system_prompt = content

        try:
            prefix = self.renderer.create_conversation_prefix_with_tools(
                tool_specs, system_prompt
            )
        except NotImplementedError:
            logger.warning(
                "Renderer %s does not support tool definitions, skipping injection",
                type(self.renderer).__name__,
            )
            if system_prompt:
                remaining.insert(0, Message(role="system", content=system_prompt))
            return remaining

        return prefix + remaining
