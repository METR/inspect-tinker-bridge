# Preserve Task Metadata (eval_metadata) Through the Bridge

## Problem

In Tinker RL mode, the monitor scanner crashes with `KeyError: 'eval_metadata'` because task-level metadata (like `side_task_correctness_scorer_name`) is lost during dataset conversion.

Task metadata is set in `shushcast.py` but only sample-level metadata is serialized into the HuggingFace dataset. When the scorer reconstructs `TaskState`, eval-level metadata is missing.

## Fix (bridge-only)

### 1. `types.py` — New field on `SampleInfoDict`

```python
inspect_eval_metadata: Required[str]  # JSON-serialized task.metadata
```

### 2. `dataset.py` — Serialize task metadata in `sample_to_row()`

```python
"inspect_eval_metadata": json.dumps(task.metadata or {}),
```

### 3. `scoring.py` — Inject into `TaskState.metadata`

```python
eval_metadata_raw = info.get("inspect_eval_metadata", "{}")
metadata["eval_metadata"] = eval_metadata_raw
```

This flows through `as_scorer` to `transcript.metadata["sample_metadata"]["eval_metadata"]`.

### Companion change (monitoring_horizons)

Scanner needs to read from `transcript.metadata["sample_metadata"]["eval_metadata"]` instead of `transcript.metadata["eval_metadata"]`.

## Backward Compatibility

`.get("inspect_eval_metadata", "{}")` in scoring.py handles old datasets missing the field.
