"""Tests for rollout_saving module."""

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from inspect_tinker_bridge import rollout_saving
from inspect_tinker_bridge.rollout_saving import RolloutRewardFnSig
from inspect_tinker_bridge.types import MessageDict, SampleInfoDict, ScoringContext


@pytest.fixture
def sample_context() -> ScoringContext:
    """Create a minimal ScoringContext for testing."""
    return ScoringContext(
        conversation=[
            MessageDict(role="user", content="What is 2+2?"),
            MessageDict(role="assistant", content="4"),
        ],
        sample_info=SampleInfoDict(
            inspect_sample_id="test_123",
            inspect_input_raw="What is 2+2?",
            inspect_target_raw="4",
            inspect_task_name="math_task",
        ),
        scores={},
        individual_rewards={"scorer_0": 0.5},
        base_reward=0.5,
        current_turn=1,
        max_turns=10,
        answer="4",
    )


def _dummy_reward_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
    """Dummy reward function for tests."""
    return (0.5, {})


class _CustomClass:
    """Non-serializable custom class for testing."""

    pass


class TestSafeJsonSerializable:
    """Tests for _safe_json_serializable."""

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param({"key": "value", "number": 42, "nested": {"a": 1}}, id="dict"),
            pytest.param([1, 2, "three", {"nested": True}], id="list"),
            pytest.param("simple string", id="string"),
            pytest.param(42, id="int"),
        ],
    )
    def test_serializable_objects_returned_as_is(self, data: Any) -> None:
        """Serializable objects should be returned unchanged."""
        result = rollout_saving._safe_json_serializable(data)
        assert result == data

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param({"timestamp": datetime.now()}, id="datetime"),
            pytest.param({"obj": _CustomClass()}, id="custom_class"),
            pytest.param({"binary": b"hello"}, id="bytes"),
        ],
    )
    def test_non_serializable_objects_return_raw(self, data: Any) -> None:
        """Non-serializable objects should return _raw string."""
        result = rollout_saving._safe_json_serializable(data)
        assert "_raw" in result
        assert isinstance(result["_raw"], str)


class TestWithRolloutSavingValidation:
    """Tests for with_rollout_saving parameter validation."""

    @pytest.mark.parametrize(
        ("samples_per_batch", "save_every", "expected_error"),
        [
            pytest.param(
                0, 10, "samples_per_batch must be positive", id="samples_zero"
            ),
            pytest.param(
                -5, 10, "samples_per_batch must be positive", id="samples_negative"
            ),
            pytest.param(10, 0, "save_every must be positive", id="save_every_zero"),
            pytest.param(
                10, -1, "save_every must be positive", id="save_every_negative"
            ),
        ],
    )
    def test_invalid_parameters_raise_valueerror(
        self, samples_per_batch: int, save_every: int, expected_error: str
    ) -> None:
        """Invalid parameter values should raise ValueError."""
        mock_renderer = MagicMock()
        mock_renderer.tokenizer = MagicMock()

        with pytest.raises(ValueError, match=expected_error):
            rollout_saving.with_rollout_saving(
                inner_fn=_dummy_reward_fn,
                output_path=Path("/tmp/rollouts.jsonl"),
                renderer=mock_renderer,
                samples_per_batch=samples_per_batch,
                save_every=save_every,
            )


@pytest.fixture
def capture_build_record(mocker: MockerFixture) -> Any:
    """Fixture that captures the build_record callback from with_rollout_saving."""
    captured: dict[str, Any] = {"build_record": None}

    def capture_upstream(
        inner_fn: RolloutRewardFnSig,
        output_path: Any,
        renderer: Any,
        *,
        samples_per_batch: int,
        save_every: int,
        build_record: Any,
    ) -> RolloutRewardFnSig:
        captured["build_record"] = build_record
        return inner_fn

    mocker.patch.object(
        rollout_saving, "_with_rollout_saving", side_effect=capture_upstream
    )
    mocker.patch.object(
        rollout_saving, "compute_message_tokens", return_value={"total": 5}
    )

    mock_renderer = MagicMock()
    mock_renderer.tokenizer = MagicMock()

    rollout_saving.with_rollout_saving(
        inner_fn=_dummy_reward_fn,
        output_path=Path("/tmp/rollouts.jsonl"),
        renderer=mock_renderer,
        samples_per_batch=10,
    )

    assert captured["build_record"] is not None
    return captured["build_record"]


class TestWithRolloutSavingWrapper:
    """Tests for with_rollout_saving wrapper behavior."""

    def test_calls_upstream_with_correct_args(self, mocker: MockerFixture) -> None:
        """Wrapper should call upstream with_rollout_saving with correct arguments."""
        mock_upstream = mocker.patch.object(
            rollout_saving,
            "_with_rollout_saving",
            return_value=_dummy_reward_fn,
        )
        mock_renderer = MagicMock()
        mock_renderer.tokenizer = MagicMock()

        def inner_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
            return (0.5, {})

        output_path = Path("/tmp/rollouts.jsonl")

        rollout_saving.with_rollout_saving(
            inner_fn=inner_fn,
            output_path=output_path,
            renderer=mock_renderer,
            samples_per_batch=32,
            save_every=5,
        )

        mock_upstream.assert_called_once()
        call_kwargs = mock_upstream.call_args.kwargs
        assert call_kwargs["samples_per_batch"] == 32
        assert call_kwargs["save_every"] == 5
        assert "build_record" in call_kwargs

    def test_build_record_returns_expected_structure(
        self, capture_build_record: Any, sample_context: ScoringContext
    ) -> None:
        """build_record callback should return dict with expected keys."""
        record = capture_build_record(
            ctx=sample_context,
            total_reward=0.75,
            rewards={"scorer_0": 0.5, "scorer_1": 1.0},
            step=42,
            renderer_name="test_renderer",
        )

        assert "timestamp" in record
        assert record["step"] == 42
        assert record["sample_id"] == "test_123"
        assert record["total_reward"] == 0.75
        assert record["renderer_name"] == "test_renderer"
        assert record["individual_rewards"] == {"scorer_0": 0.5, "scorer_1": 1.0}
        assert len(record["conversation"]) == 2
        assert len(record["token_counts"]) == 2
        assert "sample_info" in record
        assert "scores" in record

    def test_build_record_handles_score_with_metadata(
        self, capture_build_record: Any
    ) -> None:
        """build_record should handle scores with metadata."""
        from inspect_ai.scorer import Score

        ctx = ScoringContext(
            conversation=[MessageDict(role="user", content="test")],
            sample_info=SampleInfoDict(
                inspect_sample_id="test",
                inspect_input_raw="test",
                inspect_target_raw="answer",
                inspect_task_name="test",
            ),
            scores={
                "match_0": Score(
                    value=1.0, answer="correct", metadata={"reason": "exact"}
                ),
                "null_score": None,
            },
            individual_rewards={"match_0": 1.0, "null_score": 0.0},
            base_reward=0.5,
            current_turn=1,
            max_turns=10,
            answer="answer",
        )

        record = capture_build_record(
            ctx=ctx,
            total_reward=0.5,
            rewards={},
            step=1,
            renderer_name="test",
        )

        assert record["scores"]["match_0"]["value"] == 1.0
        assert record["scores"]["match_0"]["answer"] == "correct"
        assert record["scores"]["match_0"]["metadata"] == {"reason": "exact"}
        assert record["scores"]["null_score"]["value"] == 0.0
        assert record["scores"]["null_score"]["answer"] is None

    def test_build_record_handles_non_serializable_metadata(
        self, capture_build_record: Any
    ) -> None:
        """build_record should safely handle non-serializable score metadata."""
        from inspect_ai.scorer import Score

        ctx = ScoringContext(
            conversation=[MessageDict(role="user", content="test")],
            sample_info=SampleInfoDict(
                inspect_sample_id="test",
                inspect_input_raw="test",
                inspect_target_raw="answer",
                inspect_task_name="test",
            ),
            scores={
                "scorer_0": Score(
                    value=1.0,
                    answer="ok",
                    metadata={"timestamp": datetime.now()},
                ),
            },
            individual_rewards={"scorer_0": 1.0},
            base_reward=1.0,
            current_turn=1,
            max_turns=10,
            answer="answer",
        )

        record = capture_build_record(
            ctx=ctx,
            total_reward=1.0,
            rewards={},
            step=1,
            renderer_name="test",
        )

        metadata = record["scores"]["scorer_0"]["metadata"]
        assert "_raw" in metadata

    def test_build_record_missing_inspect_sample_id(
        self, capture_build_record: Any
    ) -> None:
        """build_record should handle missing inspect_sample_id gracefully."""
        ctx = ScoringContext(
            conversation=[MessageDict(role="user", content="test")],
            sample_info=SampleInfoDict(
                inspect_input_raw="test",
                inspect_target_raw="answer",
                inspect_task_name="test",
            ),
            scores={},
            individual_rewards={},
            base_reward=0.0,
            current_turn=1,
            max_turns=10,
            answer="answer",
        )

        record = capture_build_record(
            ctx=ctx,
            total_reward=0.0,
            rewards={},
            step=1,
            renderer_name="test",
        )

        assert record["sample_id"] is None
