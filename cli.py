from __future__ import annotations

import argparse
import sys
import uuid

from app.agent import Agent
from app.config import KNOWLEDGE_BASE_DIR, ORDERS_PATH
from app.llm_client import OpenRouterLLMClient
from app.orders import OrderStore
from app.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser(description="Aster & Row reliable support agent")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-llm", action="store_true", help="disable OpenRouter fallback")
    args = parser.parse_args()

    llm = None
    if not args.no_llm:
        try:
            llm = OpenRouterLLMClient()
        except Exception as exc:
            print(f"OpenRouter fallback disabled: {exc}", file=sys.stderr)
            print("Deterministic policy/order paths remain available.\n")

    agent = Agent(
        Retriever(KNOWLEDGE_BASE_DIR),
        OrderStore(ORDERS_PATH),
        llm,
    )

    session_id = str(uuid.uuid4())
    print("Aster & Row support agent. Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        result = agent.handle_message(session_id, user_text)
        print(f"\nAgent: {result['response']}")
        if result["handoff"]:
            print("[Human assistance recommended]")
        print()

        if args.debug:
            result["trace"].print_debug()


if __name__ == "__main__":
    main()
