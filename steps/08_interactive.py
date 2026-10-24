"""STEP 8 - Your turn. An interactive chat over the document.

Run:  python steps/08_interactive.py

Type questions; 'reset' clears the conversation, 'quit' exits.
Everything runs locally: the document never leaves your machine.

Uses GuardedChatRAG (see step 10), so small talk like 'hi' or 'ok thanks'
is answered directly instead of being turned into an invented policy
question. Pass --naive to use the unguarded version and see the
difference. --trace shows which path each message took.
"""

import _bootstrap  # noqa: F401

import sys

from rag import config
from rag.chain import ChatRAG
from rag.guarded import GuardedChatRAG
from rag.llm import check_ollama


def main() -> None:
    check_ollama()
    naive = "--naive" in sys.argv
    trace = "--trace" in sys.argv
    chat = ChatRAG() if naive else GuardedChatRAG()

    mode = "NAIVE - no guards" if naive else "guarded"
    print(f"RAG chat over {config.DOCUMENT_PATH.name}  "
          f"(model: {config.LLM_MODEL}, {mode})")
    print("Commands: 'reset' to clear history, 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not question:
            continue
        if question.lower() in {"quit", "exit", "bye"}:
            print("Bye.")
            return
        if question.lower() == "reset":
            chat.reset()
            print("(history cleared)\n")
            continue

        answer = chat.ask(question)
        if trace and not naive:
            print(f"      [{chat.trace.get('path')}]")
        print(f"\nBot: {answer.strip()}\n")


if __name__ == "__main__":
    main()
