
import os

import httpx
from ollama import Client, ResponseError

MODEL = "qwen3:4b-instruct"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

_client = Client(host=OLLAMA_HOST)


class OllamaError(Exception):
    """Base error with a user-facing message."""


class OllamaUnavailableError(OllamaError):
    """Ollama server is not reachable or the connection dropped."""


class OllamaModelNotFoundError(OllamaError):
    """The configured model is not available in Ollama."""


def _unreachable_message() -> str:
    return (
        f"Cannot reach Ollama at {OLLAMA_HOST}. Start it with "
        "'ollama serve' (or open the Ollama app) and try again."
    )


def _model_missing_message() -> str:
    return f"Model '{MODEL}' not found. Run: ollama pull {MODEL}"


def send_message(messages: list, tools: list):
    """Send a conversation request to Ollama."""

    try:
        return _client.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            keep_alive="10m",
            options={
                "temperature": 0
            }
        )

    except ConnectionError as error:
        # The ollama library raises this on httpx.ConnectError.
        raise OllamaUnavailableError(_unreachable_message()) from error

    except httpx.TransportError as error:
        # ReadError, RemoteProtocolError, timeouts: not wrapped by ollama.
        raise OllamaUnavailableError(
            "Lost connection to Ollama while processing the request."
        ) from error

    except ResponseError as error:
        if error.status_code == 404:
            raise OllamaModelNotFoundError(_model_missing_message()) from error

        raise OllamaError(
            f"Ollama returned an error ({error.status_code}): {error.error}"
        ) from error


def check_ollama() -> None:
    """Verify Ollama is reachable and the configured model is installed."""

    try:
        response = _client.list()

    except (ConnectionError, httpx.TransportError) as error:
        raise OllamaUnavailableError(_unreachable_message()) from error

    except ResponseError as error:
        raise OllamaError(
            f"Ollama returned an error ({error.status_code}): {error.error}"
        ) from error

    installed = {model.model for model in response.models}
    wanted = {MODEL} if ":" in MODEL else {MODEL, f"{MODEL}:latest"}

    if not installed & wanted:
        raise OllamaModelNotFoundError(_model_missing_message())
