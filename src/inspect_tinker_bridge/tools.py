from tinker_cookbook.renderers import ToolSpec

BUILT_IN_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="bash",
        description="Execute a bash command in the sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
            },
            "required": ["command"],
        },
    ),
    ToolSpec(
        name="python",
        description="Execute Python code in the sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute",
                },
            },
            "required": ["code"],
        },
    ),
    ToolSpec(
        name="submit",
        description="Submit the task. Call this when you have completed the task.",
        parameters={
            "type": "object",
            "properties": {},
        },
    ),
    ToolSpec(
        name="set_timeout",
        description="Set the timeout in seconds for subsequent tool calls.",
        parameters={
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (positive, max 3600)",
                },
            },
            "required": ["timeout"],
        },
    ),
]
