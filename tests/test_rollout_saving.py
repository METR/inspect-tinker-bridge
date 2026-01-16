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


class TestSafeJsonSerializable:
    """Tests for _safe_json_serializable."""

    def test_serializable_dict_returned_as_is(self) -> None:
        """Serializable objects should be returned unchanged."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        result = rollout_saving._safe_json_serializable(data)
        assert result == data

    def test_serializable_list_returned_as_is(self) -> None:
        """Serializable lists should be returned unchanged."""
        data = [1, 2, "three", {"nested": True}]
        result = rollout_saving._safe_json_serializable(data)
        assert result == data

    def test_non_serializable_datetime_returns_raw(self) -> None:
        """Non-serializable datetime should return _raw string."""
        data = {"timestamp": datetime.now()}
        result = rollout_saving._safe_json_serializable(data)
        assert "_raw" in result
        assert isinstance(result["_raw"], str)

    def test_non_serializable_custom_class_returns_raw(self) -> None:
        """Non-serializable custom class should return _raw string."""

        class CustomClass:
            pass

        data = {"obj": CustomClass()}
        result = rollout_saving._safe_json_serializable(data)
        assert "_raw" in result
        assert isinstance(result["_raw"], str)

    def test_non_serializable_bytes_returns_raw(self) -> None:
        """Non-serializable bytes should return _raw string."""
        data = {"binary": b"hello"}
        result = rollout_saving._safe_json_serializable(data)
        assert "_raw" in result


class TestWithRolloutSavingValidation:
    """Tests for with_rollout_saving parameter validation."""

    def test_samples_per_batch_zero_raises_valueerror(self) -> None:
        """samples_per_batch=0 should raise ValueError."""
        mock_renderer = MagicMock()
        mock_renderer.tokenizer = MagicMock()

        with pytest.raises(ValueError, match="samples_per_batch must be positive"):
            rollout_saving.with_rollout_saving(
                inner_fn=_dummy_reward_fn,
                output_path=Path("/tmp/rollouts.jsonl"),
                renderer=mock_renderer,
                samples_per_batch=0,
            )

    def test_samples_per_batch_negative_raises_valueerror(self) -> None:
        """Negative samples_per_batch should raise ValueError."""
        mock_renderer = MagicMock()
        mock_renderer.tokenizer = MagicMock()

        with pytest.raises(ValueError, match="samples_per_batch must be positive"):
            rollout_saving.with_rollout_saving(
                inner_fn=_dummy_reward_fn,
                output_path=Path("/tmp/rollouts.jsonl"),
                renderer=mock_renderer,
                samples_per_batch=-5,
            )

    def test_save_every_zero_raises_valueerror(self) -> None:
        """save_every=0 should raise ValueError."""
        mock_renderer = MagicMock()
        mock_renderer.tokenizer = MagicMock()

        with pytest.raises(ValueError, match="save_every must be positive"):
            rollout_saving.with_rollout_saving(
                inner_fn=_dummy_reward_fn,
                output_path=Path("/tmp/rollouts.jsonl"),
                renderer=mock_renderer,
                samples_per_batch=10,
                save_every=0,
            )

    def test_save_every_negative_raises_valueerror(self) -> None:
        """Negative save_every should raise ValueError."""
        mock_renderer = MagicMock()
        mock_renderer.tokenizer = MagicMock()

        with pytest.raises(ValueError, match="save_every must be positive"):
            rollout_saving.with_rollout_saving(
                inner_fn=_dummy_reward_fn,
                output_path=Path("/tmp/rollouts.jsonl"),
                renderer=mock_renderer,
                samples_per_batch=10,
                save_every=-1,
            )


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
        self, mocker: MockerFixture, sample_context: ScoringContext
    ) -> None:
        """build_record callback should return dict with expected keys."""
        captured_build_record: Any = None

        def capture_upstream(
            inner_fn: RolloutRewardFnSig,
            output_path: Any,
            renderer: Any,
            *,
            samples_per_batch: int,
            save_every: int,
            build_record: Any,
        ) -> RolloutRewardFnSig:
            nonlocal captured_build_record
            captured_build_record = build_record
            return inner_fn

        mocker.patch.object(
            rollout_saving, "_with_rollout_saving", side_effect=capture_upstream
        )
        mocker.patch.object(
            rollout_saving,
            "compute_message_tokens",
            return_value={"total": 10, "content": 8, "reasoning_content": 2},
        )

        mock_renderer = MagicMock()
        mock_renderer.tokenizer = MagicMock()

        rollout_saving.with_rollout_saving(
            inner_fn=_dummy_reward_fn,
            output_path=Path("/tmp/rollouts.jsonl"),
            renderer=mock_renderer,
            samples_per_batch=10,
        )

        assert captured_build_record is not None

        # Call build_record with sample context
        record = captured_build_record(
            ctx=sample_context,
            total_reward=0.75,
            rewards={"scorer_0": 0.5, "scorer_1": 1.0},
            step=42,
            renderer_name="test_renderer",
        )

        # Verify structure
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
        self, mocker: MockerFixture
    ) -> None:
        """build_record should handle scores with metadata."""
        from inspect_ai.scorer import Score

        captured_build_record: Any = None

        def capture_upstream(
            inner_fn: RolloutRewardFnSig,
            output_path: Any,
            renderer: Any,
            *,
            samples_per_batch: int,
            save_every: int,
            build_record: Any,
        ) -> RolloutRewardFnSig:
            nonlocal captured_build_record
            captured_build_record = build_record
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

        assert captured_build_record is not None

        # Create context with scores
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

        record = captured_build_record(
            ctx=ctx,
            total_reward=0.5,
            rewards={},
            step=1,
            renderer_name="test",
        )

        # Verify score details
        assert "match_0" in record["scores"]
        assert record["scores"]["match_0"]["value"] == 1.0
        assert record["scores"]["match_0"]["answer"] == "correct"
        assert record["scores"]["match_0"]["metadata"] == {"reason": "exact"}

        assert "null_score" in record["scores"]
        assert record["scores"]["null_score"]["value"] == 0.0
        assert record["scores"]["null_score"]["answer"] is None

    def test_build_record_handles_non_serializable_metadata(
        self, mocker: MockerFixture
    ) -> None:
        """build_record should safely handle non-serializable score metadata."""
        from inspect_ai.scorer import Score

        captured_build_record: Any = None

        def capture_upstream(
            inner_fn: RolloutRewardFnSig,
            output_path: Any,
            renderer: Any,
            *,
            samples_per_batch: int,
            save_every: int,
            build_record: Any,
        ) -> RolloutRewardFnSig:
            nonlocal captured_build_record
            captured_build_record = build_record
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

        assert captured_build_record is not None

        # Create context with non-serializable metadata
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
                    metadata={"timestamp": datetime.now()},  # Not JSON serializable
                ),
            },
            individual_rewards={"scorer_0": 1.0},
            base_reward=1.0,
            current_turn=1,
            max_turns=10,
            answer="answer",
        )

        # Should not raise - metadata is safely converted
        record = captured_build_record(
            ctx=ctx,
            total_reward=1.0,
            rewards={},
            step=1,
            renderer_name="test",
        )

        # Verify metadata was safely serialized
        metadata = record["scores"]["scorer_0"]["metadata"]
        assert "_raw" in metadata

    def test_build_record_missing_inspect_sample_id(
        self, mocker: MockerFixture
    ) -> None:
        """build_record should handle missing inspect_sample_id gracefully."""
        captured_build_record: Any = None

        def capture_upstream(
            inner_fn: RolloutRewardFnSig,
            output_path: Any,
            renderer: Any,
            *,
            samples_per_batch: int,
            save_every: int,
            build_record: Any,
        ) -> RolloutRewardFnSig:
            nonlocal captured_build_record
            captured_build_record = build_record
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

        assert captured_build_record is not None

        # Create context without inspect_sample_id
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

        record = captured_build_record(
            ctx=ctx,
            total_reward=0.0,
            rewards={},
            step=1,
            renderer_name="test",
        )

        # sample_id should be None when inspect_sample_id is missing
        assert record["sample_id"] is None
