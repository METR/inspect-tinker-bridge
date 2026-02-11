"""Tests for sandbox module."""

import pydantic
import pytest
from pytest_mock import MockerFixture

from inspect_tinker_bridge.sandbox import (
    SandboxConfig,
    _deserialize_config,
    create_sandbox_for_sample,
)
from inspect_tinker_bridge.types import SampleInfoDict


class _StubSandboxConfig(pydantic.BaseModel):
    image: str = "python:3.12"
    replicas: int = 1


class TestDeserializeConfig:
    @pytest.mark.parametrize(
        ("config_input", "expected"),
        [
            pytest.param(None, None, id="none_returns_none"),
            pytest.param(
                "/path/to/compose.yaml",
                "/path/to/compose.yaml",
                id="file_path_passthrough",
            ),
            pytest.param("[1, 2]", "[1, 2]", id="json_non_dict_passthrough"),
            pytest.param("42", "42", id="json_number_passthrough"),
            pytest.param('"a string"', '"a string"', id="json_string_passthrough"),
        ],
    )
    def test_non_dict_configs(
        self, config_input: str | None, expected: str | None
    ) -> None:
        """Non-dict configs (None, file paths, non-dict JSON) pass through unchanged."""
        assert _deserialize_config("docker", config_input) == expected

    def test_json_dict_deserializes_to_basemodel(self, mocker: MockerFixture) -> None:
        """JSON dict config is deserialized to BaseModel via the sandbox plugin."""
        stub = _StubSandboxConfig(image="ubuntu:22.04", replicas=3)
        mocker.patch(
            "inspect_tinker_bridge.sandbox.deserialize_sandbox_specific_config",
            return_value=stub,
        )
        result = _deserialize_config("k8s", '{"image": "ubuntu:22.04", "replicas": 3}')
        assert isinstance(result, _StubSandboxConfig)
        assert result.image == "ubuntu:22.04"

    def test_plugin_returning_dict_raises(self, mocker: MockerFixture) -> None:
        """Raises ValueError when plugin returns dict instead of BaseModel."""
        mocker.patch(
            "inspect_tinker_bridge.sandbox.deserialize_sandbox_specific_config",
            return_value={"image": "python:3.12"},
        )
        with pytest.raises(ValueError, match="Failed to deserialize"):
            _deserialize_config("k8s", '{"image": "python:3.12"}')


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
    async def test_basemodel_config_deserialized(self, mocker: MockerFixture) -> None:
        """JSON-serialized BaseModel config should be deserialized back to BaseModel."""
        stub_config = _StubSandboxConfig(image="ubuntu:22.04", replicas=3)
        sample_info: SampleInfoDict = {
            "inspect_sample_id": "test-k8s",
            "inspect_sandbox": ("k8s", stub_config.model_dump_json()),
            "inspect_files": None,
            "inspect_setup": None,
            "inspect_metadata": "{}",
            "inspect_eval_metadata": "{}",
        }

        mocker.patch(
            "inspect_tinker_bridge.sandbox.registry_find_sandboxenv",
            return_value=mocker.MagicMock(),
        )
        mock_init = mocker.patch(
            "inspect_tinker_bridge.sandbox.init_sandbox_environments_sample",
            return_value={"default": mocker.MagicMock()},
        )
        mocker.patch(
            "inspect_tinker_bridge.sandbox.deserialize_sandbox_specific_config",
            return_value=stub_config,
        )

        await create_sandbox_for_sample(
            sample_info, "test_task", None, sandbox_init_timeout=120
        )

        mock_init.assert_called_once()
        assert mock_init.call_args is not None
        config_arg = mock_init.call_args[1]["config"]
        assert isinstance(config_arg, _StubSandboxConfig)
        assert config_arg.image == "ubuntu:22.04"
        assert config_arg.replicas == 3

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
