"""STEP 4 - Search WITHOUT an LLM.

Run:  python steps/04_retrieve_only.py

The most useful debugging habit in RAG: look at what the retriever returns
before blaming the model. If the right chunk isn't in this list, no LLM can
give you a correct answer - it never saw the information.

Note the scores are DISTANCES: lower is more similar.
"""

import _bootstrap  # noqa: F401

from rag import config
from rag.store import load_store

QUESTIONS = [
    "What is the mobile phone policy?",
    "Can I eat in company vehicles?",
    "What is the recipe for chocolate cake?",   # deliberately not in the doc
]


def main() -> None:
    store = load_store()

    for question in QUESTIONS:
        print("=" * 70)
        print(f"QUERY: {question}")
        print("=" * 70)

        results = store.similarity_search_with_score(question, k=config.TOP_K)
        for rank, (doc, score) in enumerate(results, start=1):
            preview = doc.page_content[:180].replace("\n", " ")
            print(f"\n  [{rank}] distance={score:.4f}")
            print(f"      {preview}...")
        print()

    print("The last query has no good answer in the document, but the store")
    print("still returns its 4 closest chunks - a vector search ALWAYS returns")
    print("something. Look at the distances: they are noticeably worse.")
    print("That is why step 6's prompt must tell the model to say 'I don't know'.")
    print("\nNext: python steps/05_ask_with_llm.py")


if __name__ == "__main__":
    main()
