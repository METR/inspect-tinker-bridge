"""Tests for truncation module."""

import logging

import pytest

from inspect_tinker_bridge import truncation


class TestTruncateStringToBytes:
    """Tests for truncate_string_to_bytes utility."""

    @pytest.mark.parametrize(
        ("input_str", "max_bytes"),
        [
            pytest.param("short", 100, id="well_under_limit"),
            pytest.param("exact", 5, id="exact_limit"),
            pytest.param("", 10, id="empty_string"),
        ],
    )
    def test_no_truncation_needed(self, input_str: str, max_bytes: int) -> None:
        assert truncation.truncate_string_to_bytes(input_str, max_bytes) is None

    def test_ascii_truncation(self) -> None:
        s = "a" * 100
        result = truncation.truncate_string_to_bytes(s, 20)
        assert result is not None
        truncated, original = result
        assert original == 100
        assert len(truncated.encode("utf-8")) <= 20
        # First 10 + last 10
        assert truncated == "a" * 20

    def test_ascii_middle_truncation_preserves_edges(self) -> None:
        s = "AAAAAAAAAA" + "BBBBBBBBBB"  # 10 A's + 10 B's = 20 bytes
        result = truncation.truncate_string_to_bytes(s, 10)
        assert result is not None
        truncated, original = result
        assert original == 20
        # First 5 bytes + last 5 bytes
        assert truncated == "AAAAABBBBB"

    def test_multibyte_safety(self) -> None:
        # Each emoji is 4 bytes in UTF-8
        s = "\U0001f600" * 10  # 40 bytes total
        result = truncation.truncate_string_to_bytes(s, 16)
        assert result is not None
        truncated, original = result
        assert original == 40
        # Verify round-trip: decoded string re-encodes cleanly (no broken sequences)
        assert truncated == truncated.encode("utf-8").decode("utf-8")
        assert len(truncated.encode("utf-8")) <= 16

    def test_max_bytes_zero(self) -> None:
        result = truncation.truncate_string_to_bytes("hello", 0)
        assert result is not None
        truncated, original = result
        assert original == 5
        assert truncated == ""

    def test_odd_max_bytes(self) -> None:
        s = "abcdefghij"  # 10 bytes
        result = truncation.truncate_string_to_bytes(s, 5)
        assert result is not None
        truncated, _ = result
        # half=2, remainder=3 -> first 2 + last 3
        assert truncated == "abhij"


class TestTruncateToolOutput:
    """Tests for truncate_tool_output wrapper."""

    def test_short_output_passthrough(self) -> None:
        output = "hello world"
        result = truncation.truncate_tool_output(
            output, "bash", truncation.DEFAULT_MAX_TOOL_OUTPUT
        )
        assert result == output

    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("bash", id="bash"),
            pytest.param("python", id="python"),
        ],
    )
    def test_long_output_wrapped(self, tool_name: str) -> None:
        output = "x" * 20000
        result = truncation.truncate_tool_output(output, tool_name, 100)
        assert f"The output of your call to {tool_name} was too long" in result
        assert "<START_TOOL_OUTPUT>" in result
        assert "<END_TOOL_OUTPUT>" in result

    def test_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        output = "x" * 200
        with caplog.at_level(logging.WARNING):
            truncation.truncate_tool_output(output, "bash", 50)
        assert "Truncated bash output from 200 to 50 bytes" in caplog.text
