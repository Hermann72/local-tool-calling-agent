
from types import SimpleNamespace

import pytest

from app import agent as agent_module
from app.tools.calculator import calculate_average, increase_percentage
from app.tools.validation import (
    ToolArgumentError,
    build_args_model,
    validate_arguments,
)


AVERAGE = build_args_model(calculate_average)
INCREASE = build_args_model(increase_percentage)


def test_coerces_numeric_strings():

    kwargs = validate_arguments(
        "increase_percentage", INCREASE, {"value": "10", "percentage": 15}
    )

    assert kwargs == {"value": 10.0, "percentage": 15.0}


def test_coerces_list_items():

    kwargs = validate_arguments(
        "calculate_average", AVERAGE, {"numbers": ["1", 2, "3.5"]}
    )

    assert kwargs == {"numbers": [1.0, 2.0, 3.5]}


def test_parses_list_sent_as_json_string():

    kwargs = validate_arguments(
        "calculate_average", AVERAGE, {"numbers": "[1, 2, 3]"}
    )

    assert kwargs == {"numbers": [1.0, 2.0, 3.0]}


def test_parses_arguments_sent_as_json_string():

    kwargs = validate_arguments(
        "increase_percentage", INCREASE, '{"value": 100, "percentage": "5"}'
    )

    assert kwargs == {"value": 100.0, "percentage": 5.0}


def test_rejects_non_numeric_value():

    with pytest.raises(ToolArgumentError) as info:
        validate_arguments(
            "increase_percentage", INCREASE, {"value": "abc", "percentage": 5}
        )

    message = str(info.value)
    assert "Invalid arguments for tool 'increase_percentage'" in message
    assert "- value:" in message
    assert "'abc'" in message
    assert "Call 'increase_percentage' again" in message


def test_reports_index_of_bad_list_item():

    with pytest.raises(ToolArgumentError) as info:
        validate_arguments("calculate_average", AVERAGE, {"numbers": ["1", "x"]})

    assert "- numbers[1]:" in str(info.value)


def test_reports_missing_argument():

    with pytest.raises(ToolArgumentError) as info:
        validate_arguments("increase_percentage", INCREASE, {"value": 10})

    assert "- percentage: required argument is missing" in str(info.value)


def test_reports_unexpected_argument():

    with pytest.raises(ToolArgumentError) as info:
        validate_arguments(
            "calculate_average", AVERAGE, {"numbers": [1], "unit": "kg"}
        )

    assert "- unit: unexpected argument (allowed: numbers)" in str(info.value)


@pytest.mark.parametrize(
    "arguments",
    [
        {"value": True, "percentage": 5},
        {"value": 10, "percentage": False},
    ],
)
def test_rejects_booleans_for_numbers(arguments):

    with pytest.raises(ToolArgumentError):
        validate_arguments("increase_percentage", INCREASE, arguments)


def test_rejects_boolean_inside_list():

    with pytest.raises(ToolArgumentError) as info:
        validate_arguments("calculate_average", AVERAGE, {"numbers": [1, True]})

    assert "- numbers[1]:" in str(info.value)


def _response(content="", tool_calls=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=message, total_duration=None, eval_count=None)


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_agent_sends_validation_error_back_to_model(monkeypatch):

    sent = []
    responses = iter([
        _response(tool_calls=[_tool_call("increase_percentage", {"value": "ten", "percentage": 5})]),
        _response(tool_calls=[_tool_call("increase_percentage", {"value": "10", "percentage": 5})]),
        _response(content="The result is 10.5"),
    ])

    def fake_send_message(messages, tools):
        sent.append(list(messages))
        return next(responses)

    monkeypatch.setattr(agent_module, "send_message", fake_send_message)

    answer = agent_module.run_agent("increase ten by 5%")

    assert answer == "The result is 10.5"

    first_tool_reply = sent[1][-1]
    assert first_tool_reply["role"] == "tool"
    assert "Invalid arguments for tool 'increase_percentage'" in first_tool_reply["content"]

    second_tool_reply = sent[2][-1]
    assert second_tool_reply["content"] == "10.5"
