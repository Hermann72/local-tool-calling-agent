
import sys

from app.agent import run_agent
from app.core.ollama_client import MODEL, OllamaError, check_ollama


def main():

    print("=" * 45)
    print("       LOCAL TOOL-CALLING AGENT")
    print("=" * 45)

    try:
        check_ollama()
        print(f"Connected to Ollama (model: {MODEL})")

    except OllamaError as error:
        # Warn but keep going: the user can start Ollama and retry.
        print(f"\n[!] {error}")

    while True:

        try:
            user_input = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nAgent terminated.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("\nAgent terminated.")
            break

        if not user_input:
            continue

        try:
            response = run_agent(user_input)

        except KeyboardInterrupt:
            print("\n\nRequest cancelled.")
            continue

        except OllamaError as error:
            print(f"\n[!] {error}")
            continue

        print("\nAgent:")
        print(response)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        # Safety net, e.g. Ctrl+C during the startup check.
        print("\nAgent terminated.")

    sys.exit(0)
