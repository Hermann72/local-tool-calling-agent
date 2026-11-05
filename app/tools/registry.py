
from app.tools.calculator import (
    calculate_average,
    increase_percentage
)


# Tools exposed to the LLM.
# To add a new tool, import it above and add it to this list.
TOOLS = [
    calculate_average,
    increase_percentage
]


# Functions available for execution, derived from TOOLS
# so that both always stay in sync.
AVAILABLE_FUNCTIONS = {tool.__name__: tool for tool in TOOLS}

