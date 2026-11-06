
import time

from app.core.ollama_client import send_message

from app.tools.registry import (
    TOOLS,
    AVAILABLE_FUNCTIONS,
    ARGUMENT_MODELS
)
from app.tools.validation import ToolArgumentError, validate_arguments


def run_agent(user_input: str, max_iterations: int = 5) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a mathematical assistant that uses external tools.\n\n"

                "1. You MUST use the available tools for every "
                "mathematical calculation.\n"
                "2. Never calculate numerical results by yourself.\n"
                "3. Never invent, omit, or modify numbers provided "
                "by the user.\n"
                "4. For an average calculation, call "
                "calculate_average with ALL the numbers provided.\n"
                "5. For a percentage increase, call "
                "increase_percentage with the original value and "
                "the requested percentage.\n"
                "6. When one calculation depends on another, use "
                "the actual result returned by the first tool.\n"
                "7. Do not provide a final answer until all requested "
                "calculations have been executed successfully.\n"
                "8. Explain the results using the values returned "
                "by the tools."                
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    for iteration in range(max_iterations):

        print(f"\n--- Agent iteration {iteration + 1} ---")

        start_time = time.perf_counter()

        response = send_message(
            messages=messages,
            tools=TOOLS
        )

        elapsed = time.perf_counter() - start_time
        
        print(f"LLM request time: {elapsed:.2f} seconds")

        if response.total_duration:
            print(
                f"Ollama processing time: "
                f"{response.total_duration / 1_000_000_000:.2f} seconds"
            )

            print(f"Tokens generated: {response.eval_count}")

        messages.append(response.message)

        tool_calls = response.message.tool_calls

        print(f"Tool calls returned: {len(tool_calls or [])}")

        if not tool_calls:
            return response.message.content

        for call in tool_calls:

            function_name = call.function.name
            arguments = call.function.arguments

            print(f"\nTool requested: {function_name}")
            print(f"Arguments: {arguments}")

            function = AVAILABLE_FUNCTIONS.get(function_name)

            if function is None:
                result = (
                    f"Unknown tool: {function_name}. "
                    f"Available tools: {', '.join(AVAILABLE_FUNCTIONS)}"
                )

            else:
                try:
                    kwargs = validate_arguments(
                        function_name,
                        ARGUMENT_MODELS[function_name],
                        arguments
                    )

                    if kwargs != arguments:
                        print(f"Coerced arguments: {kwargs}")

                    result = function(**kwargs)

                except ToolArgumentError as error:
                    result = str(error)

                except Exception as error:
                    result = f"Execution error in '{function_name}': {error}"

            print(f"Tool result: {result}")

            messages.append({
                "role": "tool",
                "tool_name": function_name,
                "content": str(result)
            })

    return "Maximum agent iterations reached."
