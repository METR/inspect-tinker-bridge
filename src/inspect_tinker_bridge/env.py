"""
Tinker RL environment implementations for Inspect tasks.

Provides:
- InspectEnv: Single environment instance wrapping an Inspect sample
- InspectEnvGroupBuilder: Creates groups of environments for same problem
- InspectRLDataset: Produces batches of EnvGroupBuilders
"""

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from random import Random
from typing import Literal, cast

import tinker
from datasets import Dataset as HFDataset
from inspect_ai.scorer import Scorer
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl import types
from tinker_cookbook.renderers import Message, Renderer, ToolCall as TinkerToolCall

from inspect_tinker_bridge import sandbox as sandbox_module
from inspect_tinker_bridge import scoring
from inspect_tinker_bridge.types import (
    CustomRewardFn,
    DatasetRowDict,
    MessageDict,
    SampleInfoDict,
)

logger = logging.getLogger(__name__)


def _tool_error_message(tool_id: str, tool_name: str, error: str) -> Message:
    """Create a tool error message."""
    return Message(
        role="tool", content=f"Error: {error}", tool_call_id=tool_id, name=tool_name
    )


class InspectEnv(types.Env):
    """
    Tinker Env wrapping a single Inspect sample.

    Single-turn: Returns reward after one step (with sandbox if needed)
    Multi-turn: Supports tool calls (bash, submit) with sandbox

    Manages sandbox lifecycle internally (no Tinker cleanup hooks).
    """

    def __init__(
        self,
        sample_info: SampleInfoDict,
        prompt_messages: list[MessageDict],
        answer: str | None,
        renderer: Renderer,
        scorers: list[Scorer],
        env_type: Literal["single_turn", "multi_turn"],
        max_turns: int = 1,
        sandbox_config: sandbox_module.SandboxConfig | None = None,
        task_name: str = "inspect",
        custom_reward_fn: CustomRewardFn | None = None,
        custom_reward_fn_timeout: float = scoring.CUSTOM_REWARD_FN_TIMEOUT,
    ):
        self.sample_info = sample_info
        self.prompt_messages = prompt_messages
        self.answer = answer
        self.renderer = renderer
        self.scorers = scorers
        self.env_type = env_type
        self.max_turns = max_turns
        self.sandbox_config = sandbox_config
        self.task_name = task_name
        self.custom_reward_fn = custom_reward_fn
        self.custom_reward_fn_timeout = custom_reward_fn_timeout

        # Per-rollout state
        self.conversation: list[Message] = []
        self.current_turn = 0
        self.sandbox_instance: sandbox_module.SandboxInstance | None = None
        self._submitted = False

    async def initial_observation(
        self,
    ) -> tuple[types.Observation, StopCondition]:
        """Create sandbox if needed, return tokenized prompt."""
        # Create sandbox if configured
        if self.sandbox_config:
            self.sandbox_instance = await sandbox_module.create_sandbox_for_sample(
                self.sample_info, self.task_name, self.sandbox_config
            )

        try:
            # Convert prompt messages to Tinker format
            self.conversation = [self._dict_to_message(m) for m in self.prompt_messages]

            return (
                self.renderer.build_generation_prompt(self.conversation),
                self.renderer.get_stop_sequences(),
            )
        except Exception:
            # Cleanup sandbox on failure to prevent resource leak
            await self._cleanup()
            raise

    async def step(self, action: types.Action) -> types.StepResult:
        """Process action, manage sandbox, return reward when done."""
        try:
            # Parse the response
            message, valid = self.renderer.parse_response(action)
            self.conversation.append(message)
            self.current_turn += 1

            # Check for episode end conditions
            episode_done = self._should_end_episode(message)

            if episode_done:
                reward, individual_rewards = await self._compute_reward()
                await self._cleanup()
                return types.StepResult(
                    reward=reward,
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.renderer.get_stop_sequences(),
                    metrics={
                        "correct": float(reward > 0),
                        "format_valid": float(valid),
                        **individual_rewards,
                    },
                )

            # Multi-turn: handle tool calls
            tool_calls: list[TinkerToolCall] | None = message.get("tool_calls")
            if self.env_type == "multi_turn" and tool_calls:
                tool_results = await self._execute_tools(tool_calls)
                self.conversation.extend(tool_results)

            return types.StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=self.renderer.build_generation_prompt(
                    self.conversation
                ),
                next_stop_condition=self.renderer.get_stop_sequences(),
                metrics={"format_valid": float(valid)},
            )
        except Exception:
            # Cleanup on exception, but don't let cleanup errors mask the original
            try:
                await self._cleanup()
            except Exception:
                logger.exception("Error during cleanup after exception in step()")
            raise

    def _should_end_episode(self, message: Message) -> bool:
        """Determine if the episode should end."""
        # Single-turn always ends after first response
        if self.env_type == "single_turn":
            return True

        # Multi-turn: check for submit or max turns
        if self._submitted:
            return True
        if self.current_turn >= self.max_turns:
            return True

        # Check if message contains submit tool call
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            if hasattr(tc, "function") and tc.function.name == "submit":
                self._submitted = True
                return True

        return False

    async def _compute_reward(self) -> tuple[float, dict[str, float]]:
        """Run Inspect scorers and compute combined reward."""
        if not self.scorers:
            return 0.0, {}

        # Convert conversation to dict format for scoring
        conversation_dicts = [self._message_to_dict(m) for m in self.conversation]

        # Get sandbox environments if available
        sandbox_envs = (
            self.sandbox_instance.environments if self.sandbox_instance else None
        )

        return await scoring.compute_reward(
            conversation=conversation_dicts,
            info=self.sample_info,
            scorers=self.scorers,
            sandbox_envs=sandbox_envs,
            custom_reward_fn=self.custom_reward_fn,
            custom_reward_fn_timeout=self.custom_reward_fn_timeout,
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            answer=self.answer,
        )

    async def _execute_tools(self, tool_calls: list[TinkerToolCall]) -> list[Message]:
        """Execute tool calls and return results."""
        results: list[Message] = []

        for i, tc in enumerate(tool_calls):
            tool_name = tc.function.name
            tool_id = tc.id if tc.id else f"tc_{i}"

            if tool_name == "submit":
                self._submitted = True
                results.append(
                    Message(
                        role="tool",
                        content="Task submitted. Rollout complete.",
                        tool_call_id=tool_id,
                        name="submit",
                    )
                )
            elif tool_name in ("bash", "python"):
                # Execute command in sandbox
                args = tc.function.arguments
                try:
                    args_dict: dict[str, object] = json.loads(args)
                except json.JSONDecodeError as e:
                    results.append(
                        _tool_error_message(
                            tool_id, tool_name, f"Invalid JSON in tool arguments: {e}"
                        )
                    )
                    continue

                cmd: list[str]
                if tool_name == "bash":
                    if "command" not in args_dict:
                        results.append(
                            _tool_error_message(
                                tool_id,
                                tool_name,
                                "missing 'command' argument for bash tool",
                            )
                        )
                        continue
                    cmd = ["bash", "-c", str(args_dict["command"])]
                else:
                    if "code" not in args_dict:
                        results.append(
                            _tool_error_message(
                                tool_id,
                                tool_name,
                                "missing 'code' argument for python tool",
                            )
                        )
                        continue
                    cmd = ["python", "-c", str(args_dict["code"])]

                if self.sandbox_instance:
                    exec_result = await sandbox_module.exec_in_sandbox(
                        self.sandbox_instance.environments,
                        cmd,
                        timeout=60,
                    )
                    output = exec_result.stdout or ""
                    if exec_result.stderr:
                        output = f"{output}\nstderr: {exec_result.stderr}"
                    if not output:
                        output = "(no output)"
                else:
                    output = "Error: No sandbox available"

                results.append(
                    Message(
                        role="tool",
                        content=output,
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
            else:
                # Unknown tool
                results.append(
                    _tool_error_message(
                        tool_id, tool_name, f"Unknown tool: {tool_name}"
                    )
                )

        return results

    async def _cleanup(self) -> None:
        """Cleanup sandbox if present."""
        if self.sandbox_instance:
            await sandbox_module.cleanup_sandbox(self.sandbox_instance)
            self.sandbox_instance = None

    @staticmethod
    def _dict_to_message(d: MessageDict) -> Message:
        """Convert a dict to a Tinker Message."""
        msg = Message(
            role=d["role"],
            content=d["content"],
        )
        # Copy tool_calls if present (assistant messages)
        if "tool_calls" in d:
            msg["tool_calls"] = [
                TinkerToolCall(
                    function=TinkerToolCall.FunctionBody(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                    id=tc["id"],
                )
                for tc in d["tool_calls"]
            ]
        # Copy tool message fields if present
        if "tool_call_id" in d:
            msg["tool_call_id"] = d["tool_call_id"]
        if "name" in d:
            msg["name"] = d["name"]
        return msg

    def _message_to_dict(self, m: Message) -> MessageDict:
        """Convert a Tinker Message to a dict using renderer's to_openai_message.

        Requires renderer to implement to_openai_message (available in tinker-cookbook>=0.1.0).
        """
        if not hasattr(self.renderer, "to_openai_message"):
            raise ValueError(
                f"Renderer {type(self.renderer).__name__} does not implement to_openai_message. "
                "Upgrade tinker-cookbook or use a renderer that supports this method."
            )
        return cast(MessageDict, self.renderer.to_openai_message(m))


@dataclass(frozen=True)
class InspectEnvGroupBuilder(types.EnvGroupBuilder):
    """
    Builds group of InspectEnv instances for same problem.

    Follows Tinker pattern: one HF row = one builder = N envs for parallel rollouts.
    """

    env_thunk: Callable[[], InspectEnv]
    num_envs: int
    dataset_name: str = "inspect"

    async def make_envs(self) -> Sequence[types.Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class InspectRLDataset(types.RLDataset):
    """
    Tinker RLDataset that produces batches of InspectEnvGroupBuilder.

    Maps HF dataset rows to EnvGroupBuilders.
    """

    def __init__(
        self,
        hf_dataset: HFDataset,
        renderer: Renderer,
        scorers: list[Scorer],
        env_type: Literal["single_turn", "multi_turn"],
        max_turns: int,
        sandbox_config: sandbox_module.SandboxConfig | None,
        num_envs_per_group: int = 1,
        batch_size: int = 1,
        task_name: str = "inspect",
        custom_reward_fn: CustomRewardFn | None = None,
        custom_reward_fn_timeout: float = scoring.CUSTOM_REWARD_FN_TIMEOUT,
        num_epochs: int = 1,
        shuffle: bool = False,
        shuffle_seed: int | None = None,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
        self.hf_dataset = hf_dataset
        self.renderer = renderer
        self.scorers = scorers
        self.env_type: Literal["single_turn", "multi_turn"] = env_type
        self.max_turns = max_turns
        self.sandbox_config = sandbox_config
        self.num_envs_per_group = num_envs_per_group
        self.batch_size = batch_size
        self.task_name = task_name
        self.custom_reward_fn = custom_reward_fn
        self.custom_reward_fn_timeout = custom_reward_fn_timeout
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self._batches_per_epoch = math.ceil(len(hf_dataset) / batch_size)

    def _make_env_group_builder(self, row: DatasetRowDict) -> InspectEnvGroupBuilder:
        """Convert one HF row to an EnvGroupBuilder."""
        return InspectEnvGroupBuilder(
            env_thunk=partial(
                InspectEnv,
                sample_info=row["info"],
                prompt_messages=row["prompt"],
                answer=row["answer"],
                renderer=self.renderer,
                scorers=self.scorers,
                env_type=self.env_type,
                max_turns=self.max_turns,
                sandbox_config=self.sandbox_config,
                task_name=self.task_name,
                custom_reward_fn=self.custom_reward_fn,
                custom_reward_fn_timeout=self.custom_reward_fn_timeout,
            ),
            num_envs=self.num_envs_per_group,
            dataset_name=self.task_name,
        )

    def _get_epoch_indices(self, epoch: int) -> list[int]:
        """Get dataset indices for a given epoch, shuffled if enabled."""
        indices = list(range(len(self.hf_dataset)))
        if self.shuffle:
            seed = self.shuffle_seed if self.shuffle_seed is not None else 0
            Random(seed + epoch).shuffle(indices)
        return indices

    def get_batch(self, index: int) -> Sequence[types.EnvGroupBuilder]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Batch index {index} out of range [0, {len(self)})")
        # Map global index to epoch and within-epoch index
        epoch = index // self._batches_per_epoch
        epoch_index = index % self._batches_per_epoch

        indices = self._get_epoch_indices(epoch)
        start = epoch_index * self.batch_size
        end = min(start + self.batch_size, len(indices))

        # HFDataset is untyped; cast rows to our expected type
        return [
            self._make_env_group_builder(
                cast(DatasetRowDict, self.hf_dataset[indices[i]])
            )
            for i in range(start, end)
        ]

    def __len__(self) -> int:
        return self._batches_per_epoch * self.num_epochs
