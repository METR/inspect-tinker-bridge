"""
Multi-turn coding task with Python execution for testing the Inspect-Tinker bridge.

The model must write code and can use the python tool to test it.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import python


def create_coding_samples() -> list[Sample]:
    """Create simple coding problems that benefit from iterative testing."""
    problems = [
        {
            "input": "Write a Python function called `is_palindrome(s)` that returns True if the string s is a palindrome (ignoring case and spaces), False otherwise. Test it and when you're confident it works, call submit() with your final implementation.",
            "target": "def is_palindrome",
            "id": "palindrome",
        },
        {
            "input": "Write a Python function called `fizzbuzz(n)` that returns a list of strings from 1 to n where: multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', multiples of both are 'FizzBuzz', and other numbers are their string representation. Test it and call submit() when done.",
            "target": "def fizzbuzz",
            "id": "fizzbuzz",
        },
        {
            "input": "Write a Python function called `count_vowels(s)` that counts the number of vowels (a,e,i,o,u) in a string, case-insensitive. Test it and call submit() when done.",
            "target": "def count_vowels",
            "id": "count_vowels",
        },
        {
            "input": "Write a Python function called `reverse_words(s)` that reverses the order of words in a string. For example, 'hello world' becomes 'world hello'. Test it and call submit() when done.",
            "target": "def reverse_words",
            "id": "reverse_words",
        },
        {
            "input": "Write a Python function called `sum_digits(n)` that returns the sum of all digits in a non-negative integer. Test it and call submit() when done.",
            "target": "def sum_digits",
            "id": "sum_digits",
        },
        {
            "input": "Write a Python function called `find_max(lst)` that finds the maximum value in a list without using the built-in max() function. Test it and call submit() when done.",
            "target": "def find_max",
            "id": "find_max",
        },
        {
            "input": "Write a Python function called `is_prime(n)` that returns True if n is a prime number, False otherwise. Test it and call submit() when done.",
            "target": "def is_prime",
            "id": "is_prime",
        },
        {
            "input": "Write a Python function called `factorial(n)` that computes the factorial of a non-negative integer n. Test it and call submit() when done.",
            "target": "def factorial",
            "id": "factorial",
        },
    ]
    return [
        Sample(
            input=p["input"],
            target=p["target"],
            id=p["id"],
        )
        for p in problems
    ]


SYSTEM_PROMPT = """You are a Python programming assistant. Your task is to write correct Python code.

You have access to a Python interpreter via the python() tool. Use it to test your code before submitting.

When you are confident your solution is correct, call submit() with your final implementation.

Always test your code with multiple test cases before submitting."""


@task
def coding_task() -> Task:
    """
    Multi-turn coding task with Python execution sandbox.

    The model writes code iteratively, testing with the python tool,
    and submits when confident.
    """
    return Task(
        dataset=MemoryDataset(create_coding_samples()),
        solver=[
            system_message(SYSTEM_PROMPT),
            basic_agent(tools=[python()], max_attempts=5),
        ],
        scorer=includes(),
        sandbox="docker",
    )
