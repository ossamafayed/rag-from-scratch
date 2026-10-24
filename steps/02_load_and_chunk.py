"""STEP 2 - Load a document and cut it into chunks.

Run:  python steps/02_load_and_chunk.py

No embeddings, no LLM yet. This step is pure text handling, and it is the
step that most determines whether your RAG works: bad chunks cannot be
rescued by a better model.

Try it: change CHUNK_SIZE in rag/config.py to 300, run again, and watch the
chunk count rise and each chunk get narrower.
"""

import _bootstrap  # noqa: F401

from rag import config
from rag.loader import download_document, load_document, split_documents


def main() -> None:
    download_document()

    documents = load_document()
    full_text = documents[0].page_content
    print(f"\nloaded 1 document, {len(full_text):,} characters")
    print(f"metadata: {documents[0].metadata}")

    chunks = split_documents(documents)
    print(
        f"\nsplit into {len(chunks)} chunks "
        f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})"
    )

    sizes = [len(c.page_content) for c in chunks]
    print(f"chunk sizes: min={min(sizes)} max={max(sizes)} avg={sum(sizes)//len(sizes)}")

    print("\n--- first chunk ---")
    print(chunks[0].page_content[:400])
    print("\n--- second chunk ---")
    print(chunks[1].page_content[:400])

    print("\nNotice the chunks break at paragraph boundaries, not mid-word.")
    print("Next: python steps/03_embed_and_store.py")


if __name__ == "__main__":
    main()
