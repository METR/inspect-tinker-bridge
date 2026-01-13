"""Tests for scoring module, focusing on custom_reward_fn."""

import asyncio
import math
from collections.abc import Callable

import pytest
from pytest_mock import MockerFixture

from inspect_tinker_bridge import scoring
from inspect_tinker_bridge.types import (
    CustomRewardFn,
    MessageDict,
    SampleInfoDict,
    ScoringContext,
)


@pytest.fixture
def sample_context() -> ScoringContext:
    """Create a minimal ScoringContext for testing."""
    return ScoringContext(
        conversation=[MessageDict(role="user", content="test")],
        sample_info=SampleInfoDict(
            inspect_sample_id="test",
            inspect_input_raw="test",
            inspect_target_raw="answer",
            inspect_task_name="test",
        ),
        scores={},
        individual_rewards={"scorer_0": 0.5},
        base_reward=0.5,
        current_turn=1,
        max_turns=10,
        answer="answer",
    )


# Factory functions for parameterized tests
def _sync_float_fn(ctx: ScoringContext) -> float:
    return ctx.base_reward * 2


async def _async_float_fn(ctx: ScoringContext) -> float:
    await asyncio.sleep(0.001)
    return ctx.base_reward * 2


def _sync_tuple_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
    return 0.8, {"bonus": 0.3}


async def _async_tuple_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
    await asyncio.sleep(0.001)
    return 0.8, {"bonus": 0.3}


class TestInvokeCustomRewardFn:
    """Tests for _invoke_custom_reward_fn."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reward_fn", "expected_reward", "expected_metrics"),
        [
            pytest.param(_sync_float_fn, 1.0, {}, id="sync_float"),
            pytest.param(_async_float_fn, 1.0, {}, id="async_float"),
            pytest.param(_sync_tuple_fn, 0.8, {"bonus": 0.3}, id="sync_tuple"),
            pytest.param(_async_tuple_fn, 0.8, {"bonus": 0.3}, id="async_tuple"),
        ],
    )
    async def test_valid_return_types(
        self,
        sample_context: ScoringContext,
        reward_fn: CustomRewardFn,
        expected_reward: float,
        expected_metrics: dict[str, float],
    ) -> None:
        """Test sync/async functions returning float or tuple."""
        reward, metrics = await scoring._invoke_custom_reward_fn(
            reward_fn, sample_context, timeout=5.0
        )
        assert reward == expected_reward
        assert metrics == expected_metrics

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self, sample_context: ScoringContext) -> None:
        """Test that timeout raises TimeoutError."""

        async def slow_fn(ctx: ScoringContext) -> float:
            await asyncio.sleep(10)
            return 0.5

        with pytest.raises(asyncio.TimeoutError):
            await scoring._invoke_custom_reward_fn(slow_fn, sample_context, timeout=0.1)

    @pytest.mark.asyncio
    async def test_invalid_return_type_raises_typeerror(
        self, sample_context: ScoringContext
    ) -> None:
        """Test that invalid return type raises TypeError."""

        def bad_fn(ctx: ScoringContext) -> str:  # type: ignore[return-value]
            return "not a number"

        with pytest.raises(TypeError, match="must return float or"):
            await scoring._invoke_custom_reward_fn(
                bad_fn,  # pyright: ignore[reportArgumentType]
                sample_context,
                timeout=5.0,
            )

    @pytest.mark.asyncio
    async def test_wrong_tuple_length_shows_actual_value(
        self, sample_context: ScoringContext
    ) -> None:
        """Test that wrong tuple length error shows actual returned value."""

        def bad_fn(ctx: ScoringContext) -> tuple[float, dict[str, float], str]:  # type: ignore[return-value]
            return (0.5, {}, "extra")

        with pytest.raises(TypeError, match=r"got \(0\.5, \{\}, 'extra'\)"):
            await scoring._invoke_custom_reward_fn(
                bad_fn,  # pyright: ignore[reportArgumentType]
                sample_context,
                timeout=5.0,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="neg_inf"),
        ],
    )
    async def test_non_finite_reward_raises_valueerror(
        self, sample_context: ScoringContext, value: float
    ) -> None:
        """Test that NaN/Inf rewards raise ValueError."""

        def bad_fn(ctx: ScoringContext) -> float:
            return value

        with pytest.raises(ValueError, match="must be finite"):
            await scoring._invoke_custom_reward_fn(bad_fn, sample_context, timeout=5.0)

    @pytest.mark.asyncio
    async def test_invalid_metric_value_raises_typeerror(
        self, sample_context: ScoringContext
    ) -> None:
        """Test that non-numeric metric value raises TypeError."""

        def bad_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
            return 0.5, {"bad": "string"}  # type: ignore[dict-item]

        with pytest.raises(TypeError, match="metric 'bad' must be float"):
            await scoring._invoke_custom_reward_fn(bad_fn, sample_context, timeout=5.0)

    @pytest.mark.asyncio
    async def test_nan_metric_logs_warning(
        self, sample_context: ScoringContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that NaN metric value logs a warning but doesn't raise."""

        def nan_metric_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
            return 0.5, {"nan_metric": float("nan")}

        reward, metrics = await scoring._invoke_custom_reward_fn(
            nan_metric_fn, sample_context, timeout=5.0
        )

        assert reward == 0.5
        assert math.isnan(metrics["nan_metric"])
        assert "nan_metric" in caplog.text
        assert "not finite" in caplog.text


@pytest.fixture
def mock_scorer_setup(
    mocker: MockerFixture,
) -> Callable[[float], None]:
    """Factory fixture to set up mocked scorer with given value."""

    def setup(score_value: float) -> None:
        mock_score = mocker.MagicMock()
        mock_score.value = score_value
        mocker.patch.object(scoring, "run_inspect_scorer", return_value=mock_score)

    return setup


@pytest.fixture
def sample_conversation() -> list[MessageDict]:
    return [MessageDict(role="user", content="test")]


@pytest.fixture
def sample_info() -> SampleInfoDict:
    return SampleInfoDict(
        inspect_sample_id="test",
        inspect_input_raw="test",
        inspect_target_raw="answer",
        inspect_task_name="test",
    )


class TestComputeRewardWithCustomFn:
    """Tests for compute_reward with custom_reward_fn integration."""

    @pytest.mark.asyncio
    async def test_custom_reward_fn_receives_context(
        self,
        mock_scorer_setup: Callable[[float], None],
        sample_conversation: list[MessageDict],
        sample_info: SampleInfoDict,
        mocker: MockerFixture,
    ) -> None:
        """Test that custom_reward_fn receives proper context."""
        mock_scorer_setup(0.75)
        received_ctx: ScoringContext | None = None

        def capture_fn(ctx: ScoringContext) -> float:
            nonlocal received_ctx
            received_ctx = ctx
            return ctx.base_reward

        fake_scorer = mocker.MagicMock()
        fake_scorer.__name__ = "fake_scorer"

        await scoring.compute_reward(
            conversation=sample_conversation,
            info=sample_info,
            scorers=[fake_scorer],
            custom_reward_fn=capture_fn,
            current_turn=3,
            max_turns=10,
            answer="the answer",
        )

        assert received_ctx is not None
        assert received_ctx.current_turn == 3
        assert received_ctx.max_turns == 10
        assert received_ctx.answer == "the answer"
        assert received_ctx.base_reward == 0.75
        assert "fake_scorer_0" in received_ctx.individual_rewards

    @pytest.mark.asyncio
    async def test_metric_collision_prefixing(
        self,
        mock_scorer_setup: Callable[[float], None],
        sample_conversation: list[MessageDict],
        sample_info: SampleInfoDict,
        mocker: MockerFixture,
    ) -> None:
        """Test that colliding metric keys get prefixed."""
        mock_scorer_setup(0.75)

        def collision_fn(ctx: ScoringContext) -> tuple[float, dict[str, float]]:
            return 0.5, {"fake_scorer_0": 0.99}

        fake_scorer = mocker.MagicMock()
        fake_scorer.__name__ = "fake_scorer"

        _reward, metrics = await scoring.compute_reward(
            conversation=sample_conversation,
            info=sample_info,
            scorers=[fake_scorer],
            custom_reward_fn=collision_fn,
        )

        assert metrics["fake_scorer_0"] == 0.75
        assert metrics["custom_reward_fn/fake_scorer_0"] == 0.99

    @pytest.mark.asyncio
    async def test_without_custom_fn_returns_base_reward(
        self,
        mock_scorer_setup: Callable[[float], None],
        sample_conversation: list[MessageDict],
        sample_info: SampleInfoDict,
        mocker: MockerFixture,
    ) -> None:
        """Test that without custom_reward_fn, base reward is returned."""
        mock_scorer_setup(0.8)

        fake_scorer = mocker.MagicMock()
        fake_scorer.__name__ = "fake_scorer"

        reward, metrics = await scoring.compute_reward(
            conversation=sample_conversation,
            info=sample_info,
            scorers=[fake_scorer],
        )

        assert reward == 0.8
        assert metrics == {"fake_scorer_0": 0.8}
