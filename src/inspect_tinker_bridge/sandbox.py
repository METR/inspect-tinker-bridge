"""
Sandbox bridge: Manage Inspect sandboxes for use in Tinker environments.

This module provides utilities to create and manage sandbox environments
that can be used during reward computation in RL training.
"""

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from inspect_ai._eval.task.sandbox import read_sandboxenv_file, resolve_sample_files
from inspect_ai.util import ExecResult
from inspect_ai.util._sandbox.context import (
    cleanup_sandbox_environments_sample,
    init_sandbox_environments_sample,
    sandbox_default_context_var,
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
)
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.registry import registry_find_sandboxenv

logger = logging.getLogger(__name__)


def _ensure_docker_context() -> None:
    """
    Initialize Docker-specific context variables.

    Must be called in the same context where sandboxes will be created.
    We call this every time because ContextVars don't persist across
    different async contexts in some execution environments.
    """
    try:
        logger.debug("Initializing Docker sandbox context")
        from inspect_ai.util._sandbox.docker.cleanup import project_cleanup_startup

        project_cleanup_startup()
        logger.info("Docker sandbox context initialized successfully")
    except ImportError:
        logger.warning("Docker sandbox not available (import failed)")


@dataclass
class SandboxConfig:
    """Configuration for sandbox creation."""

    sandbox_type: str = "docker"
    config: str | None = None
    timeout: int = 120


@dataclass
class SandboxInstance:
    """Tracks a sandbox instance with its metadata for cleanup."""

    environments: dict[str, SandboxEnvironment]
    sandbox_type: str
    config: str | None
    task_name: str


async def create_sandbox_for_sample(
    sample_info: dict[str, Any],
    task_name: str,
    sandbox_config: SandboxConfig,
) -> SandboxInstance:
    """
    Create sandbox environment(s) for a sample.

    Args:
        sample_info: The info dict from the converted sample
        task_name: Name of the task
        sandbox_config: Sandbox configuration

    Returns:
        SandboxInstance containing environments and metadata for cleanup
    """
    sample_id = sample_info.get("inspect_sample_id", "unknown")
    logger.debug(
        f"Creating sandbox for sample {sample_id}: type={sandbox_config.sandbox_type}, "
        f"task={task_name}"
    )

    # Check for per-sample sandbox configuration (not yet supported)
    per_sample_sandbox = sample_info.get("inspect_sandbox")
    if per_sample_sandbox is not None:
        raise NotImplementedError(
            f"Per-sample sandbox configuration is not yet supported. "
            f"Sample has sandbox={per_sample_sandbox}, but only task-level sandbox config is used."
        )

    # Initialize Docker context if using Docker sandbox
    if sandbox_config.sandbox_type == "docker":
        _ensure_docker_context()

    # Get the sandbox environment class
    logger.debug(f"Looking up sandbox environment class: {sandbox_config.sandbox_type}")
    sandbox_cls = registry_find_sandboxenv(sandbox_config.sandbox_type)

    # Resolve files using Inspect's resolution (handles data URIs, HTTP URLs, file paths)
    files_raw = sample_info.get("inspect_files") or {}
    resolved_files = resolve_sample_files(files_raw)
    files_bytes: dict[str, bytes] = {}
    for path, contents in resolved_files.items():
        files_bytes[path] = await read_sandboxenv_file(contents)

    # Resolve setup script using Inspect's resolution
    setup = sample_info.get("inspect_setup")
    setup_bytes: bytes | None = None
    if setup:
        setup_bytes = await read_sandboxenv_file(setup)

    # Get metadata (JSON-serialized in dataset.py for pyarrow compatibility)
    metadata_raw = sample_info.get("inspect_metadata") or {}
    metadata: dict[str, Any] = (
        json.loads(metadata_raw)
        if isinstance(metadata_raw, str)
        else dict(metadata_raw)
    )

    # Initialize sandbox environments
    logger.debug(f"Initializing sandbox environments for sample {sample_id}")
    sandboxes = await init_sandbox_environments_sample(
        sandboxenv_type=sandbox_cls,
        task_name=task_name,
        config=sandbox_config.config,
        files=files_bytes,
        setup=setup_bytes,
        metadata=metadata,
    )

    logger.info(
        f"Sandbox created for sample {sample_id}: {len(sandboxes)} environment(s) initialized"
    )

    return SandboxInstance(
        environments=sandboxes,
        sandbox_type=sandbox_config.sandbox_type,
        config=sandbox_config.config,
        task_name=task_name,
    )


async def cleanup_sandbox(instance: SandboxInstance) -> None:
    """Clean up sandbox environment(s)."""
    logger.debug(
        f"Cleaning up sandbox: type={instance.sandbox_type}, task={instance.task_name}, "
        f"num_environments={len(instance.environments)}"
    )
    await cleanup_sandbox_environments_sample(
        type=instance.sandbox_type,
        task_name=instance.task_name,
        config=instance.config,
        environments=instance.environments,
        interrupted=False,
    )
    logger.info(f"Sandbox cleanup complete for task={instance.task_name}")


@asynccontextmanager
async def sandbox_context(
    sandboxes: dict[str, SandboxEnvironment],
) -> AsyncIterator[dict[str, SandboxEnvironment]]:
    """
    Context manager that sets up the sandbox context for Inspect scorers.

    This makes sandbox() calls work within the context.
    Sets all three required ContextVars that Inspect expects:
    - sandbox_environments_context_var: The actual sandbox environments
    - sandbox_default_context_var: Name of the default sandbox
    - sandbox_with_environments_context_var: Cache for sandbox_with lookups

    Args:
        sandboxes: Dictionary of sandbox environments to make available

    Yields:
        The sandboxes dict
    """
    # Determine default sandbox name (first key in dict)
    default_name = next(iter(sandboxes.keys())) if sandboxes else "default"

    # Set all three ContextVars that Inspect expects
    token_envs = sandbox_environments_context_var.set(sandboxes)
    token_default = sandbox_default_context_var.set(default_name)
    token_with = sandbox_with_environments_context_var.set({})
    try:
        yield sandboxes
    finally:
        sandbox_environments_context_var.reset(token_envs)
        sandbox_default_context_var.reset(token_default)
        sandbox_with_environments_context_var.reset(token_with)


async def exec_in_sandbox(
    sandboxes: dict[str, SandboxEnvironment],
    cmd: list[str],
    *,
    sandbox_name: str | None = None,
    timeout: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ExecResult[str]:
    """
    Execute a command in a sandbox.

    Args:
        sandboxes: Dictionary of available sandboxes
        cmd: Command to execute
        sandbox_name: Name of sandbox to use (None for default)
        timeout: Execution timeout in seconds
        cwd: Working directory
        env: Environment variables

    Returns:
        ExecResult with stdout, stderr, and success status
    """
    logger.debug(
        f"Executing command in sandbox: {' '.join(cmd)}, sandbox_name={sandbox_name}"
    )

    # Get the appropriate sandbox
    sandbox: SandboxEnvironment
    if sandbox_name and sandbox_name in sandboxes:
        sandbox = sandboxes[sandbox_name]
    elif "default" in sandboxes:
        sandbox = sandboxes["default"]
    elif sandboxes:
        sandbox = next(iter(sandboxes.values()))
    else:
        raise RuntimeError("No sandbox available")

    result = await sandbox.exec(
        cmd=cmd,
        timeout=timeout,
        cwd=cwd,
        env=env or {},
    )

    logger.debug(
        f"Command execution complete: success={result.success}, "
        f"stdout_length={len(result.stdout) if result.stdout else 0}, "
        f"stderr_length={len(result.stderr) if result.stderr else 0}"
    )

    return result
