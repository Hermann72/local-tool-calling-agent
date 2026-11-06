
from types import SimpleNamespace

import httpx
import pytest
from ollama import ResponseError

from app import main as main_module
from app.core import ollama_client
from app.core.ollama_client import (
    MODEL,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    check_ollama,
    send_message,
)


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc
    return _raise


@pytest.mark.parametrize(
    "raised, expected",
    [
        (ConnectionError("down"), OllamaUnavailableError),
        (httpx.ReadError("dropped"), OllamaUnavailableError),
        (httpx.RemoteProtocolError("dropped"), OllamaUnavailableError),
        (ResponseError("not found", 404), OllamaModelNotFoundError),
    ],
)
def test_send_message_maps_errors(monkeypatch, raised, expected):

    monkeypatch.setattr(ollama_client._client, "chat", _raiser(raised))

    with pytest.raises(expected):
        send_message(messages=[], tools=[])


def test_send_message_generic_response_error(monkeypatch):

    monkeypatch.setattr(
        ollama_client._client, "chat", _raiser(ResponseError("boom", 500))
    )

    with pytest.raises(OllamaError) as info:
        send_message(messages=[], tools=[])

    assert type(info.value) is OllamaError
    assert "500" in str(info.value)


def test_check_ollama_unreachable(monkeypatch):

    monkeypatch.setattr(
        ollama_client._client, "list", _raiser(ConnectionError("down"))
    )

    with pytest.raises(OllamaUnavailableError):
        check_ollama()


def test_check_ollama_model_missing(monkeypatch):

    response = SimpleNamespace(models=[SimpleNamespace(model="other:1b")])
    monkeypatch.setattr(ollama_client._client, "list", lambda: response)

    with pytest.raises(OllamaModelNotFoundError):
        check_ollama()


def test_check_ollama_ok(monkeypatch):

    response = SimpleNamespace(models=[SimpleNamespace(model=MODEL)])
    monkeypatch.setattr(ollama_client._client, "list", lambda: response)

    check_ollama()


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
def test_main_exits_cleanly_on_interrupt_at_prompt(monkeypatch, capsys, exc):

    monkeypatch.setattr(main_module, "check_ollama", lambda: None)
    monkeypatch.setattr("builtins.input", _raiser(exc()))

    main_module.main()

    assert "Agent terminated." in capsys.readouterr().out


def test_main_continues_after_ollama_error_and_cancel(monkeypatch, capsys):

    monkeypatch.setattr(main_module, "check_ollama", lambda: None)

    inputs = iter(["first", "second", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    outcomes = iter([OllamaUnavailableError("Ollama is down"), KeyboardInterrupt()])
    monkeypatch.setattr(main_module, "run_agent", _raiser_iter(outcomes))

    main_module.main()

    out = capsys.readouterr().out
    assert "[!] Ollama is down" in out
    assert "Request cancelled." in out
    assert "Agent terminated." in out


def _raiser_iter(excs):
    def _raise(*args, **kwargs):
        raise next(excs)
    return _raise
