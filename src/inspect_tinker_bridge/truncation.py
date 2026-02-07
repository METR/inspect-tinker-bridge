"""Tool output truncation for sandbox execution results.

Middle-truncates long tool outputs to stay within byte limits,
mirroring Inspect's approach: keep first half + last half with
an explanatory wrapper message.
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_OUTPUT = 16 * 1024  # 16 KB, matching Inspect


def truncate_string_to_bytes(s: str, max_bytes: int) -> tuple[str, int] | None:
    """Middle-truncate a string to fit within max_bytes of UTF-8.

    Returns (truncated_output, original_bytes) if truncation was needed,
    or None if the string already fits.
    """
    encoded = s.encode("utf-8", errors="replace")
    original_bytes = len(encoded)
    if original_bytes <= max_bytes:
        return None

    half = max_bytes // 2
    remainder = max_bytes - half

    # Split encoded bytes, decode with errors="ignore" to handle splits landing
    # mid-character. May lose a few bytes vs budget; in pathological cases
    # (all wide chars, tiny budget) this can yield empty output.
    first = encoded[:half].decode("utf-8", errors="ignore")
    last = encoded[-remainder:].decode("utf-8", errors="ignore") if remainder else ""
    return first + last, original_bytes


def truncate_tool_output(output: str, tool_name: str, max_bytes: int) -> str:
    """Truncate tool output if it exceeds max_bytes, wrapping in a template.

    The max_bytes limit applies to the raw content; the wrapper adds
    a small constant overhead (~130 bytes) on top.
    Returns the original output unchanged if within limits.
    """
    result = truncate_string_to_bytes(output, max_bytes)
    if result is None:
        return output

    truncated, original_bytes = result
    logger.warning(
        "Truncated %s output from %d to %d bytes",
        tool_name,
        original_bytes,
        max_bytes,
    )
    return (
        f"The output of your call to {tool_name} was too long to be displayed.\n"
        f"Here is a truncated version:\n"
        f"<START_TOOL_OUTPUT>\n"
        f"{truncated}\n"
        f"<END_TOOL_OUTPUT>"
    )
