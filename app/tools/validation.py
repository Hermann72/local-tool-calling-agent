
"""Validate and coerce tool arguments produced by the LLM.

Argument models are derived from each tool's type hints, so the schema the
model is shown and the checks applied here never drift apart.
"""

import inspect
import json
import typing
from typing import Annotated, Any, Callable, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    ValidationError,
    create_model,
)


class ToolArgumentError(Exception):
    """Arguments for a tool call are invalid; the message is meant for the LLM."""


def _reject_bool(value: Any) -> Any:
    # Lax mode would turn True into 1.0; for a number that is almost
    # always a model mistake, so refuse it explicitly.
    if isinstance(value, bool):
        raise ValueError("expected a number, not a boolean")
    return value


def _strict_numbers(annotation: Any) -> Any:
    """Wrap int/float (also inside list[...] / Optional[...]) to reject bools."""

    if annotation in (int, float):
        return Annotated[annotation, BeforeValidator(_reject_bool)]

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list and args:
        return list[_strict_numbers(args[0])]

    if origin is Union:
        return Union[tuple(_strict_numbers(arg) for arg in args)]

    return annotation


def _expects_list(annotation: Any) -> bool:
    return get_origin(annotation) is list or annotation is list


def build_args_model(func: Callable) -> type[BaseModel]:
    """Build a Pydantic model describing the parameters of ``func``."""

    hints = typing.get_type_hints(func)
    fields = {}

    for name, param in inspect.signature(func).parameters.items():

        if name not in hints:
            raise TypeError(
                f"Tool '{func.__name__}' parameter '{name}' needs a type hint"
            )

        annotation = _strict_numbers(hints[name])
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)

    model = create_model(
        f"{func.__name__}Args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    # Keep the original hints to detect list fields during pre-processing.
    model.__tool_hints__ = {name: hints[name] for name in fields}
    return model


def _format_location(loc: tuple) -> str:
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text or "(arguments)"


def _format_error(err: dict, model: type[BaseModel]) -> str:
    location = _format_location(err["loc"])
    kind = err["type"]

    if kind == "missing":
        return f"- {location}: required argument is missing"

    if kind == "extra_forbidden":
        allowed = ", ".join(model.model_fields) or "(none)"
        return f"- {location}: unexpected argument (allowed: {allowed})"

    message = err["msg"].removeprefix("Value error, ")
    received = err.get("input")
    return f"- {location}: {message}; got {received!r} ({type(received).__name__})"


def validate_arguments(
    func_name: str,
    model: type[BaseModel],
    arguments: Any,
) -> dict:
    """Return coerced keyword arguments, or raise ToolArgumentError."""

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            raise ToolArgumentError(
                f"Invalid arguments for tool '{func_name}': arguments must be "
                f"a JSON object, got {arguments!r}. "
                f"Call '{func_name}' again with corrected arguments."
            ) from None

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, typing.Mapping):
        raise ToolArgumentError(
            f"Invalid arguments for tool '{func_name}': arguments must be an "
            f"object, got {type(arguments).__name__}. "
            f"Call '{func_name}' again with corrected arguments."
        )

    arguments = dict(arguments)
    hints = getattr(model, "__tool_hints__", {})

    # Models sometimes send lists as JSON strings: "numbers": "[1, 2, 3]".
    for name, value in arguments.items():
        if isinstance(value, str) and _expects_list(hints.get(name)):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                arguments[name] = parsed

    try:
        return model.model_validate(arguments).model_dump()

    except ValidationError as error:
        lines = [f"Invalid arguments for tool '{func_name}':"]
        lines += [_format_error(err, model) for err in error.errors()]
        lines.append(f"Call '{func_name}' again with corrected arguments.")
        raise ToolArgumentError("\n".join(lines)) from None
