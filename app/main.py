
from app.agent import run_agent


def main():

    print("=" * 45)
    print("       LOCAL TOOL-CALLING AGENT")
    print("=" * 45)

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nAgent terminated.")
            break

        if not user_input:
            continue

        response = run_agent(user_input)

        print("\nAgent:")
        print(response)


if __name__ == "__main__":
    main()
