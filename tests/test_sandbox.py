"""Tests for sandbox module."""

import pytest
from unittest.mock import MagicMock, patch

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
