"""
Tinker RL environment implementations for Inspect tasks.

Provides:
- InspectEnv: Single environment instance wrapping an Inspect sample
- InspectEnvGroupBuilder: Creates groups of environments for same problem
- InspectRLDataset: Produces batches of EnvGroupBuilders
"""

import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Sequence

import tinker
from datasets import Dataset as HFDataset
from inspect_ai.scorer import Scorer
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl import types
from tinker_cookbook.renderers import Message, Renderer

from inspect_tinker_bridge import sandbox as sandbox_module
from inspect_tinker_bridge import scoring

logger = logging.getLogger(__name__)


class InspectEnv(types.Env):
    """
    Tinker Env wrapping a single Inspect sample.

    Single-turn: Returns reward after one step (with sandbox if needed)
    Multi-turn: Supports tool calls (bash, submit) with sandbox

    Manages sandbox lifecycle internally (no Tinker cleanup hooks).
    """

    def __init__(
        self,
        sample_info: dict[str, Any],
        prompt_messages: list[dict[str, Any]],
        answer: str | None,
        renderer: Renderer,
        scorers: list[Scorer],
        env_type: Literal["single_turn", "multi_turn"],
        max_turns: int = 1,
        sandbox_config: sandbox_module.SandboxConfig | None = None,
        task_name: str = "inspect",
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

        # Convert prompt messages to Tinker format
        self.conversation = [self._dict_to_message(m) for m in self.prompt_messages]

        return (
            self.renderer.build_generation_prompt(self.conversation),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: types.Action) -> types.StepResult:
        """Process action, manage sandbox, return reward when done."""
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
                    **{k: v for k, v in individual_rewards.items()},
                },
            )

        # Multi-turn: handle tool calls
        tool_calls = message.get("tool_calls")
        if self.env_type == "multi_turn" and tool_calls:
            tool_results = await self._execute_tools(tool_calls)
            self.conversation.extend(tool_results)

        return types.StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.conversation),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={"format_valid": float(valid)},
        )

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
        )

    async def _execute_tools(self, tool_calls: list[Any]) -> list[Message]:
        """Execute tool calls and return results."""
        results: list[Message] = []

        for tc in tool_calls:
            tool_name = tc.function.name if hasattr(tc, "function") else str(tc)
            tool_id = tc.id if hasattr(tc, "id") else None

            if tool_name == "submit":
                self._submitted = True
                results.append(
                    Message(
                        role="tool",
                        content="Task submitted. Rollout complete.",
                        tool_call_id=tool_id or "",
                        name="submit",
                    )
                )
            elif tool_name == "bash":
                # Execute bash command in sandbox
                args = tc.function.arguments if hasattr(tc, "function") else "{}"
                import json

                args_dict = json.loads(args) if isinstance(args, str) else args
                command = args_dict.get("command", "")

                if self.sandbox_instance:
                    exec_result = await sandbox_module.exec_in_sandbox(
                        self.sandbox_instance.environments,
                        ["bash", "-c", command],
                        timeout=30,
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
                        tool_call_id=tool_id or "",
                        name="bash",
                    )
                )
            else:
                # Unknown tool
                results.append(
                    Message(
                        role="tool",
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_id or "",
                        name=tool_name,
                    )
                )

        return results

    async def _cleanup(self) -> None:
        """Cleanup sandbox if present."""
        if self.sandbox_instance:
            await sandbox_module.cleanup_sandbox(self.sandbox_instance)
            self.sandbox_instance = None

    @staticmethod
    def _dict_to_message(d: dict[str, Any]) -> Message:
        """Convert a dict to a Tinker Message."""
        return Message(
            role=d["role"],
            content=d.get("content", ""),
        )

    @staticmethod
    def _message_to_dict(m: Message) -> dict[str, Any]:
        """Convert a Tinker Message to a dict."""
        result: dict[str, Any] = {
            "role": m["role"],
            "content": m.get("content", ""),
        }
        if "tool_calls" in m:
            # Convert Tinker ToolCall to dict format
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in m["tool_calls"]
            ]
        if "tool_call_id" in m:
            result["tool_call_id"] = m["tool_call_id"]
        if "name" in m:
            result["name"] = m["name"]
        return result


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
    ):
        self.hf_dataset = hf_dataset
        self.renderer = renderer
        self.scorers = scorers
        self.env_type: Literal["single_turn", "multi_turn"] = env_type
        self.max_turns = max_turns
        self.sandbox_config = sandbox_config
        self.num_envs_per_group = num_envs_per_group
        self.batch_size = batch_size
        self.task_name = task_name

    def _make_env_group_builder(self, row: dict[str, Any]) -> InspectEnvGroupBuilder:
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
            ),
            num_envs=self.num_envs_per_group,
            dataset_name=self.task_name,
        )

    def get_batch(self, index: int) -> Sequence[types.EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.hf_dataset))
        return [
            self._make_env_group_builder(self.hf_dataset[i]) for i in range(start, end)
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.hf_dataset) / self.batch_size)
