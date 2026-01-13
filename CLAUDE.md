# Project Guidelines

## Type Safety

This project uses TypedDicts (defined in `src/inspect_tinker_bridge/types.py`) for structured data.

**Direct dict access for required keys:** Use `dict[key]` instead of `dict.get(key, fallback)` for keys marked as `Required` in TypedDicts. This ensures malformed data causes immediate failures rather than silent bugs.

```python
# Good - fails fast on malformed data
content = msg["content"]  # MessageDict.content is Required

# Bad - hides bugs with silent fallback
content = msg.get("content", "")
```

Reserve `.get()` for truly optional fields (those without `Required` in the TypedDict).
