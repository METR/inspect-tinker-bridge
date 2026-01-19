"""
Evaluate original vs trained model on AIME 2025 using Inspect AI.

This script runs the AIME 2025 evaluation on:
1. Original Qwen3-4B-Instruct via OpenRouter
2. Trained checkpoint via Tinker

Run with:
    cd examples && uv run --env-file ../.env python eval_aime2025.py

Requires:
    - OPENROUTER_API_KEY in .env for baseline evaluation
    - TINKER_API_KEY in .env for trained model evaluation
"""

import subprocess

# Model configuration
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OPENROUTER_MODEL = "openai/openrouter/qwen/qwen3-4b-04-28"  # OpenRouter model ID

# Trained checkpoint (update this with your actual checkpoint path)
TRAINED_CHECKPOINT = (
    "tinker://6af7890c-64bc-521e-9bbf-134620892aae:train:0/sampler_weights/final"
)

# Renderer for Qwen3 instruct models
RENDERER_NAME = "qwen3_instruct"


def run_baseline_eval() -> None:
    """Run evaluation on the original model via OpenRouter."""
    print("\n" + "=" * 60)
    print("Evaluating BASELINE model (Qwen3-4B-Instruct via OpenRouter)")
    print("=" * 60 + "\n")

    cmd = [
        "uv",
        "run",
        "inspect",
        "eval",
        "inspect_evals/aime2025",
        "--model",
        f"{OPENROUTER_MODEL}",
        "--log-dir",
        "/tmp/inspect-tinker-bridge/evals/baseline",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def run_trained_eval() -> None:
    """Run evaluation on the trained checkpoint via Tinker."""
    print("\n" + "=" * 60)
    print("Evaluating TRAINED model (checkpoint via Tinker)")
    print("=" * 60 + "\n")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "tinker_cookbook.eval.run_inspect_evals",
        f"model_path={TRAINED_CHECKPOINT}",
        f"model_name={BASE_MODEL}",
        "tasks=inspect_evals/aime2025",
        f"renderer_name={RENDERER_NAME}",
        "log_dir=/tmp/inspect-tinker-bridge/evals/trained",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run both evaluations."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AIME 2025 models")
    parser.add_argument(
        "--baseline-only", action="store_true", help="Only run baseline evaluation"
    )
    parser.add_argument(
        "--trained-only", action="store_true", help="Only run trained model evaluation"
    )
    args = parser.parse_args()

    if args.baseline_only:
        run_baseline_eval()
    elif args.trained_only:
        run_trained_eval()
    else:
        # Run both
        run_baseline_eval()
        run_trained_eval()

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print("\nResults saved to:")
    print("  Baseline: /tmp/inspect-tinker-bridge/evals/baseline/")
    print("  Trained:  /tmp/inspect-tinker-bridge/evals/trained/")
    print("\nView results with:")
    print("  uv run inspect view /tmp/inspect-tinker-bridge/evals/")


if __name__ == "__main__":
    main()
