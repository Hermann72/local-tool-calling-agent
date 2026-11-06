
from app.tools.calculator import (
    calculate_average,
    increase_percentage
)
from app.tools.validation import build_args_model


# Tools exposed to the LLM.
# To add a new tool, import it above and add it to this list.
TOOLS = [
    calculate_average,
    increase_percentage
]


# Functions available for execution, derived from TOOLS
# so that both always stay in sync.
AVAILABLE_FUNCTIONS = {tool.__name__: tool for tool in TOOLS}


# Argument validation models, built from each tool's type hints.
ARGUMENT_MODELS = {tool.__name__: build_args_model(tool) for tool in TOOLS}

