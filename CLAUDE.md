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

## Upstream Contracts

**Trust upstream library contracts; don't add defensive wrappers.** When a library (e.g., tinker_cookbook renderers) documents that a field "must be valid JSON" or follows a specific format, trust that contract. Don't wrap calls in try/except to handle violations — failing fast with a clear traceback pointing at the contract violation is better than silently producing garbage data. If invalid data appears in practice, fix it at the source (the upstream library), not by adding error-swallowing wrappers here.

## Test Style

**Parameterize similar tests:** Always use `@pytest.mark.parametrize` with `pytest.param(..., id="descriptive_name")` when testing multiple inputs with the same logic. Extract shared setup into fixtures to reduce duplication.

```python
# Good - parameterized
@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(0, "zero", id="zero_case"),
        pytest.param(-1, "negative", id="negative_case"),
    ],
)
def test_something(self, input: int, expected: str) -> None:
    assert func(input) == expected

# Bad - duplicated test functions
def test_something_zero(self) -> None:
    assert func(0) == "zero"

def test_something_negative(self) -> None:
    assert func(-1) == "negative"
```
