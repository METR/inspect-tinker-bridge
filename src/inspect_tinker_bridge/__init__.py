"""
Inspect-Tinker Bridge: Convert Inspect AI tasks to Tinker RL environments.
"""

from typing import TYPE_CHECKING

from inspect_tinker_bridge.model_api import TinkerSamplingAPI
from inspect_tinker_bridge.rollout_saving import RolloutRewardFnSig, with_rollout_saving
from inspect_tinker_bridge.sandbox import SandboxConfig, SandboxInstance
from inspect_tinker_bridge.tasks import InspectTaskInfo, load_inspect_task
from inspect_tinker_bridge.tools import BUILT_IN_TOOL_SPECS

if TYPE_CHECKING:
    from inspect_tinker_bridge.env import (
        InspectEnv,
        InspectEnvGroupBuilder,
        InspectRLDataset,
    )
    from inspect_tinker_bridge.loader import load_environment

__all__ = [
    "load_environment",
    "InspectRLDataset",
    "InspectEnvGroupBuilder",
    "InspectEnv",
    "SandboxConfig",
    "SandboxInstance",
    "InspectTaskInfo",
    "load_inspect_task",
    "with_rollout_saving",
    "RolloutRewardFnSig",
    "BUILT_IN_TOOL_SPECS",
    "TinkerSamplingAPI",
]

# Lazy imports for modules that depend on the `rl` optional extra (datasets).
_RL_IMPORTS: dict[str, tuple[str, str]] = {
    "InspectEnv": ("inspect_tinker_bridge.env", "InspectEnv"),
    "InspectEnvGroupBuilder": ("inspect_tinker_bridge.env", "InspectEnvGroupBuilder"),
    "InspectRLDataset": ("inspect_tinker_bridge.env", "InspectRLDataset"),
    "load_environment": ("inspect_tinker_bridge.loader", "load_environment"),
}


def __getattr__(name: str) -> object:
    if name in _RL_IMPORTS:
        module_path, attr = _RL_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
