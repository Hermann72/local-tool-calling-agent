
import pytest

from app.tools.calculator import (
    calculate_average,
    increase_percentage
)


def test_calculate_average():

    result = calculate_average([125, 250, 375])

    assert result == 250.0


def test_calculate_average_empty_list():

    with pytest.raises(ValueError):
        calculate_average([])


def test_increase_percentage():

    result = increase_percentage(250, 15)

    assert result == 287.5


def test_increase_percentage_zero():

    result = increase_percentage(250, 0)

    assert result == 250
