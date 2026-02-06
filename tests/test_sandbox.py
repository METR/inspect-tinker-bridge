"""Tests for sandbox module."""

import pytest
from pytest_mock import MockerFixture

from inspect_tinker_bridge.sandbox import SandboxConfig, create_sandbox_for_sample
from inspect_tinker_bridge.types import SampleInfoDict


class TestPerSampleSandboxConfig:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("sample_sandbox", "task_config_path", "expected_config"),
        [
            pytest.param(
                ("docker", "/sample/compose.yaml"),
                "/task/compose.yaml",
                "/sample/compose.yaml",
                id="per_sample_overrides_task",
            ),
            pytest.param(
                None,
                "/task/compose.yaml",
                "/task/compose.yaml",
                id="task_fallback_when_no_sample",
            ),
            pytest.param(
                ("docker", "/sample/compose.yaml"),
                None,
                "/sample/compose.yaml",
                id="per_sample_only_no_task_config",
            ),
        ],
    )
    async def test_sandbox_config_resolution(
        self,
        mocker: MockerFixture,
        sample_sandbox: tuple[str, str] | None,
        task_config_path: str | None,
        expected_config: str,
    ) -> None:
        """Test that correct sandbox config is used based on precedence."""
        sample_info: SampleInfoDict = {
            "inspect_sample_id": "test-1",
            "inspect_sandbox": sample_sandbox,
            "inspect_files": None,
            "inspect_setup": None,
            "inspect_metadata": "{}",
            "inspect_eval_metadata": "{}",
        }
        task_config = (
            SandboxConfig(sandbox_type="docker", config=task_config_path, timeout=60)
            if task_config_path
            else None
        )

        mock_registry = mocker.patch(
            "inspect_tinker_bridge.sandbox.registry_find_sandboxenv",
            return_value=mocker.MagicMock(),
        )
        mock_init = mocker.patch(
            "inspect_tinker_bridge.sandbox.init_sandbox_environments_sample",
            return_value={"default": mocker.MagicMock()},
        )
        mocker.patch("inspect_tinker_bridge.sandbox._ensure_docker_context")

        await create_sandbox_for_sample(
            sample_info, "test_task", task_config, sandbox_init_timeout=120
        )

        mock_registry.assert_called_once()
        mock_init.assert_called_once()
        assert mock_init.call_args is not None
        assert mock_init.call_args[1]["config"] == expected_config

    @pytest.mark.asyncio
    async def test_error_when_no_config_available(self) -> None:
        """Should raise error when neither per-sample nor task-level config exists."""
        sample_info: SampleInfoDict = {
            "inspect_sample_id": "test-4",
            "inspect_sandbox": None,
            "inspect_files": None,
            "inspect_setup": None,
            "inspect_metadata": "{}",
            "inspect_eval_metadata": "{}",
        }

        with pytest.raises(ValueError, match="No sandbox config for sample"):
            await create_sandbox_for_sample(
                sample_info, "test_task", None, sandbox_init_timeout=120
            )
