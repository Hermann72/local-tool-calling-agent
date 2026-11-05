
from ollama import chat

MODEL = "qwen3:4b-instruct"

def send_message(messages: list, tools: list):
    """Send a conversation request to Ollama."""

    return chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        keep_alive="10m",
        options={
            "temperature": 0
        }
    )
