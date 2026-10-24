"""STEP 7 - Multi-turn chat: making follow-up questions work.

Run:  python steps/07_chat_with_memory.py

The problem: you ask "What is the mobile policy?" then "What does it cover?".
Sent to the retriever as-is, "What does it cover?" contains no searchable
words - "it" means nothing to a vector search.

The fix: use the LLM to rewrite the follow-up into a standalone question
FIRST, then run ordinary RAG on the rewrite. Watch the rewrites print below.
"""

import _bootstrap  # noqa: F401

from rag.chain import ChatRAG, format_history
from rag.llm import check_ollama

CONVERSATION = [
    "What is the mobile phone policy?",
    "What does it cover?",          # "it" only resolves via history
    "Why does the company have it?",
]


def main() -> None:
    check_ollama()
    chat = ChatRAG()

    for question in CONVERSATION:
        print("=" * 70)
        print(f"YOU: {question}")

        # Show the rewrite so the mechanism is visible, not magic.
        if chat.history:
            rewritten = chat.condenser.invoke(
                {"chat_history": format_history(chat.history), "question": question}
            ).strip()
            print(f"     (searched as: {rewritten})")

        print("=" * 70)
        print(f"BOT: {chat.ask(question).strip()}\n")

    print("Next: python steps/08_interactive.py  - ask your own questions")


if __name__ == "__main__":
    main()
