# inspect-tinker-bridge

Convert [Inspect AI](https://inspect.ai/) tasks into [Tinker](https://github.com/anthropics/tinker) RL environments for reinforcement learning training.

## Overview

This bridge enables training language models with RL using Inspect's rich ecosystem of evaluation tasks (datasets, scorers, solvers, sandboxes) as the source of prompts and reward signals.

## Control Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INITIALIZATION PHASE                               │
└─────────────────────────────────────────────────────────────────────────────┘

    Inspect Task Function                      Tinker Renderer
    (e.g., gsm8k, coding_task)                (e.g., LlamaRenderer)
              │                                       │
              └───────────────┬───────────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  load_environment() │  ← Main Entry Point
                    │     loader.py       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌─────────────────┐ ┌───────────────┐ ┌────────────────────┐
    │load_inspect_task│ │   Validate    │ │inspect_dataset_to_ │
    │    tasks.py     │ │ Configuration │ │   hf() dataset.py  │
    └────────┬────────┘ └───────────────┘ └─────────┬──────────┘
             │                                      │
             ▼                                      │
    ┌─────────────────┐                             │
    │ InspectTaskInfo │                             │
    │  - task         │                             │
    │  - scorers      │                             │
    │  - sandbox_type │                             │
    │  - dataset      │                             │
    └─────────────────┘                             │
                                                    ▼
                              ┌──────────────────────────────────────┐
                              │  For each Sample in Inspect Dataset  │
                              └──────────────────────┬───────────────┘
                                                     │
                                                     ▼
                              ┌──────────────────────────────────────┐
                              │   get_ground_truth_messages()        │
                              │         ground_truth.py              │
                              │                                      │
                              │  Run solver chain WITHOUT model      │
                              │  inference to get transformed prompt │
                              │  (system message, few-shot, etc.)    │
                              └──────────────────────┬───────────────┘
                                                     │
                                                     ▼
                              ┌──────────────────────────────────────┐
                              │        HuggingFace Dataset           │
                              │  ┌─────────────────────────────────┐ │
                              │  │ Row: {prompt, answer, info, id} │ │
                              │  │                                 │ │
                              │  │ prompt: List[message dicts]     │ │
                              │  │ answer: target answer           │ │
                              │  │ info: Inspect metadata (JSON)   │ │
                              │  └─────────────────────────────────┘ │
                              └──────────────────────┬───────────────┘
                                                     │
                                                     ▼
                              ┌──────────────────────────────────────┐
                              │          InspectRLDataset            │
                              │             env.py                   │
                              │                                      │
                              │  Wraps HF dataset for Tinker training│
                              └──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PHASE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                              InspectRLDataset
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │    get_batch(batch_index)      │
                    └────────────────┬───────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
    ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
    │InspectEnvGroup    │ │InspectEnvGroup    │ │InspectEnvGroup    │
    │Builder (problem 1)│ │Builder (problem 2)│ │Builder (problem N)│
    └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
              │                     │                     │
              ▼                     ▼                     ▼
         make_envs()           make_envs()           make_envs()
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │[InspectEnv × N] │   │[InspectEnv × N] │   │[InspectEnv × N] │
    │ (parallel       │   │ (parallel       │   │ (parallel       │
    │  rollouts)      │   │  rollouts)      │   │  rollouts)      │
    └─────────────────┘   └─────────────────┘   └─────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         EPISODE EXECUTION                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                              InspectEnv
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   initial_observation() │
                    └────────────┬────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                                         ▼
   (if multi-turn with sandbox)              Convert prompt messages
            │                                 to Tinker format
            ▼                                         │
   ┌────────────────────┐                             │
   │create_sandbox_for_ │                             │
   │sample() sandbox.py │                             │
   │                    │                             │
   │ - Init Docker/local│                             │
   │ - Resolve files    │                             │
   │ - Run setup script │                             │
   └────────────────────┘                             │
                                                      ▼
                                        ┌─────────────────────────┐
                                        │   Return: Observation   │
                                        │  (tokenized prompt +    │
                                        │   stop sequences)       │
                                        └─────────────────────────┘
                                                      │
                                                      │
                          ┌───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STEP LOOP                                       │
└─────────────────────────────────────────────────────────────────────────────┘

         Model generates tokens
                   │
                   ▼
         ┌─────────────────┐
         │   step(action)  │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────────────┐
         │ renderer.parse_response │
         │ (tokens → Message)      │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ Append to conversation  │
         │ Increment turn counter  │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ _should_end_episode()?  │
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
    [Episode Done]           [Continue]
         │                         │
         │                         ▼
         │            ┌─────────────────────────┐
         │            │   _execute_tools()      │
         │            │                         │
         │            │ Tool calls supported:   │
         │            │ - submit(): end episode │
         │            │ - bash(): run command   │
         │            │ - python(): run code    │
         │            │                         │
         │            │ Executes in sandbox via │
         │            │ exec_in_sandbox()       │
         │            └────────────┬────────────┘
         │                         │
         │                         ▼
         │            ┌─────────────────────────┐
         │            │ Return: next observation│
         │            │ (with tool results)     │
         │            └─────────────────────────┘
         │                         │
         │                         └──────┐
         │                                │
         │         ┌──────────────────────┘
         │         │
         │         ▼
         │    (Back to Model)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EPISODE TERMINATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

         ┌─────────────────────────┐
         │  _compute_reward()      │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────────────────────────┐
         │              scoring.py                     │
         │                                             │
         │  ┌───────────────────────────────────────┐  │
         │  │ 1. Convert conversation to Inspect    │  │
         │  │    ChatMessage format                 │  │
         │  │                                       │  │
         │  │ 2. Build TaskState with:              │  │
         │  │    - Original sample input            │  │
         │  │    - Conversation history             │  │
         │  │    - Model output (last assistant)    │  │
         │  │                                       │  │
         │  │ 3. Set up sandbox context (if needed) │  │
         │  │    via sandbox_context()              │  │
         │  │                                       │  │
         │  │ 4. Run Inspect scorers                │  │
         │  │    (exact_match, model_graded, etc.)  │  │
         │  │                                       │  │
         │  │ 5. Convert Score → float reward       │  │
         │  │    (combine multiple with weights)    │  │
         │  └───────────────────────────────────────┘  │
         └─────────────────────────────────────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │   cleanup_sandbox()     │
         │   (if sandbox exists)   │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │  Return: StepResult     │
         │   - reward              │
         │   - done=True           │
         │   - info (scorer        │
         │     breakdown)          │
         └─────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         EPISODE TYPES COMPARISON                             │
└─────────────────────────────────────────────────────────────────────────────┘

    SINGLE-TURN                          MULTI-TURN
    ───────────                          ──────────
    initial_observation()                initial_observation()
           │                                    │
           ▼                                    ▼
    step(action)                         step(action)
           │                                    │
           ▼                                    ▼
    [Always terminates]                  [Check: submit() called?
           │                              max_turns reached?]
           │                                    │
           ▼                             ┌──────┴──────┐
    compute_reward()                     ▼             ▼
           │                           [No]         [Yes]
           ▼                             │             │
    Return result                        ▼             ▼
                                  execute_tools()  compute_reward()
                                        │             │
                                        ▼             ▼
                                  Return obs      Return result
                                        │
                                        └───→ (loop back to step)
```

## Architecture Components

| Component | File | Purpose |
|-----------|------|---------|
| `load_environment` | `loader.py` | Main entry point - orchestrates initialization |
| `InspectTaskInfo` | `tasks.py` | Task introspection - extracts scorers, sandbox config |
| `get_ground_truth_messages` | `ground_truth.py` | Runs solver chain without model to get prompts |
| `inspect_dataset_to_hf` | `dataset.py` | Converts Inspect Dataset → HuggingFace Dataset |
| `InspectRLDataset` | `env.py` | Tinker RLDataset wrapper for batching |
| `InspectEnvGroupBuilder` | `env.py` | Creates parallel environments per problem |
| `InspectEnv` | `env.py` | Core Tinker Env - handles observations, steps, rewards |
| `compute_reward` | `scoring.py` | Runs Inspect scorers on conversation |
| `SandboxConfig/Instance` | `sandbox.py` | Manages Docker/local sandbox lifecycle |

## Usage

### Single-Turn (e.g., Math Problems)

```python
from inspect_evals.gsm8k import gsm8k
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from inspect_tinker_bridge import load_environment

tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
renderer = get_renderer("LlamaRenderer", tokenizer=tokenizer)

dataset = load_environment(
    gsm8k,
    renderer=renderer,
    env_type="single_turn",
    max_samples=100,
    batch_size=8,
    num_envs_per_group=4,
)
```

### Multi-Turn with Sandbox (e.g., Coding Tasks)

```python
from examples.coding_task import coding_task
from inspect_tinker_bridge import load_environment

dataset = load_environment(
    coding_task,
    renderer=renderer,
    env_type="multi_turn",
    max_turns=10,
    batch_size=4,
    sandbox_type="docker",
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `Callable` | Required | Inspect task factory function |
| `renderer` | `Renderer` | Required | Tinker message renderer/tokenizer |
| `env_type` | `str` | `"single_turn"` | `"single_turn"` or `"multi_turn"` |
| `max_samples` | `int \| None` | `None` | Limit dataset size |
| `max_turns` | `int` | `10` | Max turns per episode (multi-turn only) |
| `num_envs_per_group` | `int` | `1` | Parallel rollouts per problem |
| `batch_size` | `int` | `1` | Problems per training batch |
| `sandbox_type` | `str \| None` | `None` | `"docker"` or `"local"` |
| `sandbox_config` | `str \| None` | `None` | Path to sandbox config file |
| `submit_instruction` | `str \| None` | (default msg) | System instruction for submit tool |
