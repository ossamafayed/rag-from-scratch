"""STEP 5 - Full RAG: retrieve, then generate.

Run:  python steps/05_ask_with_llm.py

Now we add the LLM. The chain is:

    question -> retriever -> chunks -> prompt -> LLM -> answer

Compare the answers with the raw chunks from step 4 and you can see exactly
which text the model is paraphrasing.

The last part of this step shows something RAG is genuinely BAD at, so you
learn the shape of the tool and not just its happy path.
"""

import _bootstrap  # noqa: F401

from rag import config
from rag.chain import build_qa_chain_with_sources
from rag.llm import check_ollama

QUESTIONS = [
    "What is the mobile phone policy?",
    "What should I do if my company phone is lost or stolen?",
    "What does the code of conduct say about integrity?",
]


def ask(chain, question: str) -> None:
    print("=" * 70)
    print(f"Q: {question}")
    print("=" * 70)

    result = chain.invoke(question)

    print(f"\nA: {result['answer'].strip()}\n")
    print("  sources used:")
    for rank, doc in enumerate(result["documents"], start=1):
        preview = doc.page_content[:90].replace("\n", " ")
        print(f"    [{rank}] {preview}...")
    print()


def main() -> None:
    check_ollama()
    chain = build_qa_chain_with_sources()

    for question in QUESTIONS:
        ask(chain, question)

    # --- the limitation ----------------------------------------------------
    print("#" * 70)
    print("# NOW A QUESTION RAG HANDLES BADLY")
    print("#" * 70)
    ask(chain, "Summarize the whole document in three sentences.")

    print("Read that last answer next to its sources.")
    print()
    print(f"'Summarize the whole document' is not a retrieval question. The")
    print(f"retriever can only fetch TOP_K={config.TOP_K} chunks out of 26, and it picks them")
    print("by similarity to the word 'summarize' - which matches nothing in")
    print("particular. So the model is asked to summarize a document it was")
    print("only shown a random-ish 15% of, and a well-behaved model will")
    print("often refuse outright.")
    print()
    print("The fix is not a better prompt. It is a different technique:")
    print("  - map-reduce summarization: summarize EVERY chunk, then summarize")
    print("    the summaries (no retrieval step at all)")
    print("  - or just feed the whole document in, if it fits the context window")
    print()
    print("RAG answers 'what does the document say about X'.")
    print("It does not answer 'what does the document say'.")
    print("\nNext: python steps/06_prompt_matters.py")


if __name__ == "__main__":
    main()
