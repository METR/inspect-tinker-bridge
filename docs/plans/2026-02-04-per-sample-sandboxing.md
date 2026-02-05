# Per-Sample Sandboxing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable samples to specify their own sandbox configuration (e.g., different `compose.yaml` files), overriding task-level sandbox config.

**Architecture:** Per-sample sandbox config stored as `tuple[str, str | None]` in `SampleInfoDict`. During sandbox creation, per-sample config takes precedence over task-level config. Task-level config serves as fallback when sample has no sandbox specified.

**Tech Stack:** Python, pytest, inspect_ai, tinker

---

## Task 1: Update Type Definition

**Files:**
- Modify: `src/inspect_tinker_bridge/types.py:61`

**Step 1: Update the type annotation**

Change `inspect_sandbox` from allowing raw strings to only tuple format:

```python
# In SampleInfoDict (line 61)
# Before:
inspect_sandbox: str | tuple[str, str] | None

# After:
inspect_sandbox: tuple[str, str | None] | None  # (type, config) tuple
```

**Step 2: Verify no type errors**

Run: `uv run basedpyright src/inspect_tinker_bridge/types.py`
Expected: No errors

**Step 3: Commit**

```bash
git add src/inspect_tinker_bridge/types.py
git commit -m "chore: update inspect_sandbox type to tuple format"
```

---

## Task 2: Fix Serialization in dataset.py

**Files:**
- Modify: `src/inspect_tinker_bridge/dataset.py:77-86`
- Test: `tests/test_dataset.py`

**Step 1: Write failing test for SandboxEnvironmentSpec serialization**

Add to `tests/test_dataset.py`:

```python
import pytest
from inspect_ai.dataset import Sample
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_tinker_bridge.dataset import sample_to_row


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
        ],
    )
    async def test_sandbox_serialization(
        self,
        sandbox_input: SandboxEnvironmentSpec | None,
        expected_serialized: tuple[str, str | None] | None,
        minimal_task: "Task",
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
```

Note: This test requires a `minimal_task` fixture. Check if one exists in conftest.py, or create a simple one that returns a Task with minimal solver.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py::TestSampleSandboxSerialization -v`
Expected: FAIL (current code raises ValueError for SandboxEnvironmentSpec)

**Step 3: Implement the fix**

In `src/inspect_tinker_bridge/dataset.py`, replace lines 77-86:

```python
    # Serialize sandbox config to tuple format for pyarrow compatibility
    # SandboxEnvironmentSpec is the normalized form at runtime
    sandbox_serializable: tuple[str, str | None] | None = None
    if sample.sandbox is not None:
        # sample.sandbox is always SandboxEnvironmentSpec (Inspect normalizes on creation)
        config = sample.sandbox.config
        if config is not None and not isinstance(config, str):
            raise ValueError(
                f"Only string sandbox configs (file paths) are supported for serialization, "
                f"got {type(config).__name__}. BaseModel configs are not yet supported."
            )
        sandbox_serializable = (sample.sandbox.type, config)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py::TestSampleSandboxSerialization -v`
Expected: PASS

**Step 5: Run type checker**

Run: `uv run basedpyright src/inspect_tinker_bridge/dataset.py`
Expected: No errors

**Step 6: Commit**

```bash
git add src/inspect_tinker_bridge/dataset.py tests/test_dataset.py
git commit -m "feat: serialize SandboxEnvironmentSpec to tuple format"
```

---

## Task 3: Rename sandbox_config to task_sandbox_config in sandbox.py

**Files:**
- Modify: `src/inspect_tinker_bridge/sandbox.py:67-142`

**Step 1: Rename parameter in create_sandbox_for_sample**

In `src/inspect_tinker_bridge/sandbox.py`, update the function signature and docstring:

```python
async def create_sandbox_for_sample(
    sample_info: SampleInfoDict,
    task_name: str,
    task_sandbox_config: SandboxConfig | None,
) -> SandboxInstance:
    """
    Create sandbox environment(s) for a sample.

    Args:
        sample_info: The info dict from the converted sample
        task_name: Name of the task
        task_sandbox_config: Task-level sandbox configuration (fallback if sample has none)

    Returns:
        SandboxInstance containing environments and metadata for cleanup
    """
```

**Step 2: Update all references inside the function**

Replace all `sandbox_config` with `task_sandbox_config` in the function body (around lines 83-142).

**Step 3: Run type checker**

Run: `uv run basedpyright src/inspect_tinker_bridge/sandbox.py`
Expected: No errors (or errors about callers not updated yet - that's Task 5)

**Step 4: Commit**

```bash
git add src/inspect_tinker_bridge/sandbox.py
git commit -m "refactor: rename sandbox_config to task_sandbox_config in sandbox.py"
```

---

## Task 4: Implement Per-Sample Sandbox Override Logic

**Files:**
- Modify: `src/inspect_tinker_bridge/sandbox.py:67-142`
- Test: `tests/test_sandbox.py`

**Step 1: Write failing test for per-sample override**

Add to `tests/test_sandbox.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_tinker_bridge.sandbox import SandboxConfig, create_sandbox_for_sample
from inspect_tinker_bridge.types import SampleInfoDict


class TestPerSampleSandboxConfig:
    @pytest.mark.asyncio
    async def test_per_sample_sandbox_overrides_task_level(self) -> None:
        """Per-sample sandbox config should override task-level config."""
        sample_info: SampleInfoDict = {
            "inspect_sample_id": "test-1",
            "inspect_sandbox": ("docker", "/sample/compose.yaml"),
            "inspect_files": None,
            "inspect_setup": None,
            "inspect_metadata": "{}",
        }
        task_config = SandboxConfig(
            sandbox_type="docker",
            config="/task/compose.yaml",
            timeout=60,
        )

        with patch(
            "inspect_tinker_bridge.sandbox.registry_find_sandboxenv"
        ) as mock_registry, patch(
            "inspect_tinker_bridge.sandbox.init_sandbox_environments_sample"
        ) as mock_init, patch(
            "inspect_tinker_bridge.sandbox._ensure_docker_context"
        ):
            mock_sandbox_cls = MagicMock()
            mock_registry.return_value = mock_sandbox_cls
            mock_init.return_value = {"default": MagicMock()}

            await create_sandbox_for_sample(sample_info, "test_task", task_config)

            # Verify per-sample config was used, not task-level
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args
            assert call_kwargs[1]["config"] == "/sample/compose.yaml"

    @pytest.mark.asyncio
    async def test_task_level_fallback_when_no_per_sample(self) -> None:
        """Task-level config should be used when sample has no sandbox."""
        sample_info: SampleInfoDict = {
            "inspect_sample_id": "test-2",
            "inspect_sandbox": None,
            "inspect_files": None,
            "inspect_setup": None,
            "inspect_metadata": "{}",
        }
        task_config = SandboxConfig(
            sandbox_type="docker",
            config="/task/compose.yaml",
            timeout=60,
        )

        with patch(
            "inspect_tinker_bridge.sandbox.registry_find_sandboxenv"
        ) as mock_registry, patch(
            "inspect_tinker_bridge.sandbox.init_sandbox_environments_sample"
        ) as mock_init, patch(
            "inspect_tinker_bridge.sandbox._ensure_docker_context"
        ):
            mock_sandbox_cls = MagicMock()
            mock_registry.return_value = mock_sandbox_cls
            mock_init.return_value = {"default": MagicMock()}

            await create_sandbox_for_sample(sample_info, "test_task", task_config)

            # Verify task-level config was used
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args
            assert call_kwargs[1]["config"] == "/task/compose.yaml"

    @pytest.mark.asyncio
    async def test_per_sample_only_no_task_config(self) -> None:
        """Per-sample sandbox should work even without task-level config."""
        sample_info: SampleInfoDict = {
            "inspect_sample_id": "test-3",
            "inspect_sandbox": ("docker", "/sample/compose.yaml"),
            "inspect_files": None,
            "inspect_setup": None,
            "inspect_metadata": "{}",
        }

        with patch(
            "inspect_tinker_bridge.sandbox.registry_find_sandboxenv"
        ) as mock_registry, patch(
            "inspect_tinker_bridge.sandbox.init_sandbox_environments_sample"
        ) as mock_init, patch(
            "inspect_tinker_bridge.sandbox._ensure_docker_context"
        ):
            mock_sandbox_cls = MagicMock()
            mock_registry.return_value = mock_sandbox_cls
            mock_init.return_value = {"default": MagicMock()}

            # task_sandbox_config is None
            await create_sandbox_for_sample(sample_info, "test_task", None)

            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args
            assert call_kwargs[1]["config"] == "/sample/compose.yaml"

    @pytest.mark.asyncio
    async def test_error_when_no_config_available(self) -> None:
        """Should raise error when neither per-sample nor task-level config exists."""
        sample_info: SampleInfoDict = {
            "inspect_sample_id": "test-4",
            "inspect_sandbox": None,
            "inspect_files": None,
            "inspect_setup": None,
            "inspect_metadata": "{}",
        }

        with pytest.raises(ValueError, match="No sandbox config for sample"):
            await create_sandbox_for_sample(sample_info, "test_task", None)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sandbox.py::TestPerSampleSandboxConfig -v`
Expected: FAIL (NotImplementedError is raised)

**Step 3: Implement per-sample override logic**

Replace the body of `create_sandbox_for_sample` in `src/inspect_tinker_bridge/sandbox.py`:

```python
async def create_sandbox_for_sample(
    sample_info: SampleInfoDict,
    task_name: str,
    task_sandbox_config: SandboxConfig | None,
) -> SandboxInstance:
    """
    Create sandbox environment(s) for a sample.

    Args:
        sample_info: The info dict from the converted sample
        task_name: Name of the task
        task_sandbox_config: Task-level sandbox configuration (fallback if sample has none)

    Returns:
        SandboxInstance containing environments and metadata for cleanup
    """
    sample_id = sample_info.get("inspect_sample_id", "unknown")

    # Determine effective sandbox config: per-sample overrides task-level
    per_sample_sandbox = sample_info.get("inspect_sandbox")

    effective_config: SandboxConfig
    if per_sample_sandbox is not None:
        # Per-sample sandbox: (type, config) tuple
        sandbox_type, config_path = per_sample_sandbox
        effective_config = SandboxConfig(
            sandbox_type=sandbox_type,
            config=config_path,
            timeout=task_sandbox_config.timeout if task_sandbox_config else 120,
        )
        logger.debug(
            f"Using per-sample sandbox for {sample_id}: type={sandbox_type}, config={config_path}"
        )
    elif task_sandbox_config is not None:
        effective_config = task_sandbox_config
        logger.debug(f"Using task-level sandbox for {sample_id}")
    else:
        raise ValueError(
            f"No sandbox config for sample {sample_id}: "
            "neither per-sample nor task-level sandbox configured"
        )

    logger.debug(
        f"Creating sandbox for sample {sample_id}: type={effective_config.sandbox_type}, "
        f"task={task_name}"
    )

    # Initialize Docker context if using Docker sandbox
    if effective_config.sandbox_type == "docker":
        _ensure_docker_context()

    # Get the sandbox environment class
    logger.debug(f"Looking up sandbox environment class: {effective_config.sandbox_type}")
    sandbox_cls = registry_find_sandboxenv(effective_config.sandbox_type)

    # Resolve files using Inspect's resolution (handles data URIs, HTTP URLs, file paths)
    files_raw: dict[str, str] = sample_info.get("inspect_files") or {}
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
    metadata_raw = sample_info.get("inspect_metadata", "{}")
    metadata = parse_metadata_json(metadata_raw)

    # Initialize sandbox environments
    logger.debug(f"Initializing sandbox environments for sample {sample_id}")
    sandboxes = await init_sandbox_environments_sample(
        sandboxenv_type=sandbox_cls,
        task_name=task_name,
        config=effective_config.config,
        files=files_bytes,
        setup=setup_bytes,
        metadata=metadata,
    )

    logger.debug(
        f"Sandbox created for sample {sample_id}: {len(sandboxes)} environment(s) initialized"
    )

    return SandboxInstance(
        environments=sandboxes,
        sandbox_type=effective_config.sandbox_type,
        config=effective_config.config,
        task_name=task_name,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sandbox.py::TestPerSampleSandboxConfig -v`
Expected: PASS

**Step 5: Run type checker**

Run: `uv run basedpyright src/inspect_tinker_bridge/sandbox.py`
Expected: No errors

**Step 6: Commit**

```bash
git add src/inspect_tinker_bridge/sandbox.py tests/test_sandbox.py
git commit -m "feat: implement per-sample sandbox override logic"
```

---

## Task 5: Rename sandbox_config in env.py

**Files:**
- Modify: `src/inspect_tinker_bridge/env.py`

**Step 1: Rename in InspectEnv.__init__**

Update the parameter and attribute (around line 64):

```python
# Before:
sandbox_config: sandbox_module.SandboxConfig | None = None,

# After:
task_sandbox_config: sandbox_module.SandboxConfig | None = None,
```

And the attribute assignment (around line 76):

```python
# Before:
self.sandbox_config = sandbox_config

# After:
self.task_sandbox_config = task_sandbox_config
```

**Step 2: Update initial_observation method**

Update the condition and call (around lines 87-95):

```python
async def initial_observation(
    self,
) -> tuple[types.Observation, StopCondition]:
    """Create sandbox if needed, return tokenized prompt."""
    # Create sandbox if task-level config OR per-sample config exists
    has_sample_sandbox = self.sample_info.get("inspect_sandbox") is not None

    if self.task_sandbox_config or has_sample_sandbox:
        self.sandbox_instance = await sandbox_module.create_sandbox_for_sample(
            self.sample_info, self.task_name, self.task_sandbox_config
        )
    # ... rest unchanged
```

**Step 3: Rename in InspectRLDataset.__init__**

Update parameter (around line 370):

```python
# Before:
sandbox_config: sandbox_module.SandboxConfig | None,

# After:
task_sandbox_config: sandbox_module.SandboxConfig | None,
```

And attribute (around line 389):

```python
# Before:
self.sandbox_config = sandbox_config

# After:
self.task_sandbox_config = task_sandbox_config
```

**Step 4: Update _make_env_group_builder**

Update the kwarg passed to InspectEnv (around line 412):

```python
# Before:
sandbox_config=self.sandbox_config,

# After:
task_sandbox_config=self.task_sandbox_config,
```

**Step 5: Run type checker**

Run: `uv run basedpyright src/inspect_tinker_bridge/env.py`
Expected: No errors (or errors about loader.py not updated yet)

**Step 6: Commit**

```bash
git add src/inspect_tinker_bridge/env.py
git commit -m "refactor: rename sandbox_config to task_sandbox_config in env.py"
```

---

## Task 6: Update loader.py

**Files:**
- Modify: `src/inspect_tinker_bridge/loader.py`

**Step 1: Read current loader.py to find sandbox_config usage**

Find where `sandbox_config` is created and passed.

**Step 2: Rename parameter if exposed in public API**

If `sandbox_config` is a parameter to `load_environment()`, rename to `task_sandbox_config` for consistency.

Update all internal references to use the new name.

**Step 3: Run type checker**

Run: `uv run basedpyright src/inspect_tinker_bridge/loader.py`
Expected: No errors

**Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/inspect_tinker_bridge/loader.py
git commit -m "refactor: rename sandbox_config to task_sandbox_config in loader.py"
```

---

## Task 7: Final Verification

**Step 1: Run full linting**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors, files formatted

**Step 2: Run full type checking**

Run: `uv run basedpyright .`
Expected: No errors

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 4: Final commit if any formatting changes**

```bash
git add -A
git commit -m "chore: formatting"
```

---

## Summary

After completing all tasks:

1. **Types** - `inspect_sandbox` is `tuple[str, str | None] | None`
2. **Serialization** - `SandboxEnvironmentSpec` properly serialized to tuple
3. **Override Logic** - Per-sample sandbox overrides task-level, with fallback
4. **Naming** - `task_sandbox_config` throughout for clarity
5. **Condition** - Sandbox created if either task-level OR per-sample config exists
