"""
Entry point for the e-commerce support agent backend.

Run from apps/backend/:
    python main.py

For module execution:
    python -m backend.agent
"""

from backend.agent import run_agent_with_cache

if __name__ == "__main__":
    print("Smart E-Commerce Support Agent")
    print("Type 'quit' to exit\n")
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == "quit":
            break
        response = run_agent_with_cache(user_input)
        print(response)
        print()
