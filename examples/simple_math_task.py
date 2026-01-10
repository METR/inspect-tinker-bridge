"""
Simple math task for testing the Inspect-Tinker bridge.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import match


def create_math_samples() -> list[Sample]:
    """Create simple arithmetic problems."""
    problems = [
        ("What is 2 + 3?", "5"),
        ("What is 7 - 4?", "3"),
        ("What is 5 * 6?", "30"),
        ("What is 12 / 4?", "3"),
        ("What is 8 + 9?", "17"),
        ("What is 15 - 7?", "8"),
        ("What is 4 * 7?", "28"),
        ("What is 20 / 5?", "4"),
        ("What is 11 + 13?", "24"),
        ("What is 25 - 9?", "16"),
        ("What is 6 * 8?", "48"),
        ("What is 36 / 6?", "6"),
        ("What is 14 + 18?", "32"),
        ("What is 50 - 23?", "27"),
        ("What is 9 * 9?", "81"),
        ("What is 100 / 10?", "10"),
    ]
    return [
        Sample(
            input=question,
            target=answer,
            id=f"math_{i}",
        )
        for i, (question, answer) in enumerate(problems)
    ]


@task
def simple_math() -> Task:
    """
    Simple arithmetic task for testing RL training.

    The model must answer basic arithmetic questions.
    Scoring uses exact string match on the numeric answer.
    """
    return Task(
        dataset=MemoryDataset(create_math_samples()),
        scorer=match(),
    )
