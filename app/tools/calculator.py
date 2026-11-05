
def calculate_average(numbers: list[float]) -> float:
    """Calculate the average of a list of numbers.

    Args:
        numbers: The complete list of numbers given by the user, without omitting or changing any of them.

    Returns:
        The arithmetic mean of the numbers.
    """

    if not numbers:
        raise ValueError("The list cannot be empty")

    return sum(numbers) / len(numbers)


def increase_percentage(
    value: float,
    percentage: float
) -> float:
    """Increase a value by a given percentage.

    Args:
        value: The original value to increase.
        percentage: The percentage to add, as a plain number (use 10 for 10%, not 0.1).

    Returns:
        The value after applying the percentage increase.
    """

    return value * (1 + percentage / 100)
