# Inject Tool Definitions Into Multi-Turn Prompts

## Problem

Multi-turn RL environments have four built-in tools (bash, python, submit, set_timeout) that the model can call. The `InspectEnv._execute_tools()` method handles execution, but the model never sees the tool definitions/schemas in its prompt. Each renderer has `create_conversation_prefix_with_tools()` which formats tool definitions in the model's native format (Harmony developer message for GptOss, `<tools>` XML for Qwen3, `tool_declare` for Kimi, etc.), but nobody calls it.

This means the model must guess the tool interface from context alone, rather than receiving explicit tool definitions as the model format expects.

## Design

### New file: `src/inspect_tinker_bridge/tools.py`

Define `BUILT_IN_TOOL_SPECS: list[ToolSpec]` with four tools matching what `_execute_tools()` accepts:

- **bash**: `command` (string, required) — execute a bash command in the sandbox
- **python**: `code` (string, required) — execute python code in the sandbox
- **submit**: no parameters — signal task completion
- **set_timeout**: `timeout` (number, required) — set tool execution timeout in seconds

### Modified: `InspectEnv.initial_observation()` in `env.py`

After converting prompt messages to Tinker format (existing line 118), when `env_type == "multi_turn"`:

1. Extract the first `system` role message from `self.conversation` (if any), capturing its text content.
2. Remove that system message from the conversation list.
3. Call `self.renderer.create_conversation_prefix_with_tools(BUILT_IN_TOOL_SPECS, system_prompt_content)` to get prefix messages formatted for the model.
4. Prepend the prefix messages to the remaining conversation.
5. If the renderer raises `NotImplementedError` (e.g., Llama3, RoleColon), log a warning and keep the original conversation unchanged.

### No other changes needed

`load_environment()`, `InspectRLDataset`, and `InspectEnvGroupBuilder` remain unchanged — no new parameters are threaded through.

## Files Changed

| File | Change |
|------|--------|
| `src/inspect_tinker_bridge/tools.py` | New — `BUILT_IN_TOOL_SPECS` constant |
| `src/inspect_tinker_bridge/env.py` | Modify `initial_observation()` to inject tool prefix for multi-turn |
| `tests/test_env.py` | Add tests for tool prefix injection |
