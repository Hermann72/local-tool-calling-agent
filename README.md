# Local Tool-Calling Agent

A small command-line agent that runs fully on your machine. A local LLM served by [Ollama](https://ollama.com) answers maths questions by calling Python functions ("tools") rather than doing the arithmetic itself.

```
You: What is the average of 125, 250 and 375, then increase it by 15%?

--- Agent iteration 1 ---
Tool requested: calculate_average
Arguments: {'numbers': [125, 250, 375]}
Tool result: 250.0

--- Agent iteration 2 ---
Tool requested: increase_percentage
Arguments: {'value': 250.0, 'percentage': 15}
Tool result: 287.5
...
Agent:
The average is 250, and increasing it by 15% gives 287.5.
```

## Features

- **Local only.** No API keys, and nothing leaves your machine.
- **Tool-calling loop.** The model calls tools, reads their results, and repeats until it can answer. The loop is capped at 5 iterations per question.
- **Argument validation.** Before a tool runs, the arguments the model sent are checked against the tool's type hints and converted where that's safe (`"10"` → `10.0`, `"[1, 2]"` → `[1.0, 2.0]`). Invalid arguments go back to the model as a specific error so it can correct itself.
- **Clear errors.** If Ollama isn't running, drops the connection, or the model isn't installed, you get a one-line message and the session keeps going. Ctrl+C cancels the current request; Ctrl+C at the prompt exits cleanly.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- The model used by the agent:
  ```bash
  ollama pull qwen3:4b-instruct
  ```

## Installation

```bash
git clone <repository-url>
cd local-tool-calling-agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # runtime + pytest
```

## Usage

```bash
python -m app.main
```

Type a question at the `You:` prompt. To leave, type `exit` or `quit`, or press Ctrl+C.

### Configuration

| Setting | Where | Default |
|---|---|---|
| Ollama server address | `OLLAMA_HOST` environment variable | `http://localhost:11434` |
| Model | `MODEL` in [app/core/ollama_client.py](app/core/ollama_client.py) | `qwen3:4b-instruct` |
| Max iterations per question | `max_iterations` in [app/agent.py](app/agent.py) | `5` |

### Error handling

| Situation | What happens |
|---|---|
| Ollama not running | `[!] Cannot reach Ollama at ...`, then you're back at the prompt. Start Ollama and retry. |
| Connection lost mid-request | `[!] Lost connection to Ollama ...`, then back at the prompt |
| Model not installed | `[!] Model '...' not found. Run: ollama pull ...` |
| Ctrl+C while the agent works | `Request cancelled.`, then back at the prompt |
| Ctrl+C or EOF at the prompt | `Agent terminated.` and a clean exit with code 0 |

## Project structure

```
app/
├── main.py              # CLI loop, Ctrl+C and Ollama error handling
├── agent.py             # Tool-calling loop
├── core/
│   └── ollama_client.py # Ollama client, error mapping, startup check
└── tools/
    ├── calculator.py    # The tools: calculate_average, increase_percentage
    ├── registry.py      # Tools exposed to the model + validation models
    └── validation.py    # Argument coercion and error messages for the model
tests/
├── test_calculator.py
├── test_ollama_client.py
└── test_validation.py
```

## Adding a tool

1. Write a function with **type hints** and a **docstring** in `app/tools/`. Ollama builds the tool schema from them, and the validator uses the type hints:
   ```python
   def multiply(a: float, b: float) -> float:
       """Multiply two numbers.

       Args:
           a: The first number.
           b: The second number.
       """
       return a * b
   ```
2. Import it in [app/tools/registry.py](app/tools/registry.py) and add it to `TOOLS`. Execution and validation pick it up automatically.

## Running tests

```bash
python -m pytest
```

The tests don't need Ollama. All model and server calls are mocked.
