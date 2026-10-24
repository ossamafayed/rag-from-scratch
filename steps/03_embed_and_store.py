"""STEP 3 - Turn chunks into vectors and save them.

Run:  python steps/03_embed_and_store.py

The first run downloads the embedding model (~90 MB) and is slow. After that
the index lives in chroma_db/ and later steps just open it.

An embedding is a list of numbers describing what a piece of text MEANS.
Texts about similar things end up close together in that number space, which
is what makes meaning-based search possible.
"""

import _bootstrap  # noqa: F401

import shutil

from rag import config
from rag.loader import load_and_split
from rag.store import build_store, get_embeddings


def main() -> None:
    # Rebuild from scratch so re-running never silently duplicates documents.
    if config.CHROMA_DIR.exists():
        print(f"removing old index at {config.CHROMA_DIR}")
        shutil.rmtree(config.CHROMA_DIR)

    chunks = load_and_split()
    print(f"{len(chunks)} chunks to embed")

    # Show what one embedding actually looks like, before hiding it in a DB.
    embeddings = get_embeddings()
    sample = embeddings.embed_query("What is the mobile phone policy?")
    print(f"\nembedding model: {config.EMBEDDING_MODEL}")
    print(f"one question -> a vector of {len(sample)} numbers")
    print(f"first 8 of them: {[round(v, 4) for v in sample[:8]]}")

    print(f"\nembedding {len(chunks)} chunks and writing the index ...")
    build_store(chunks)
    print(f"done -> {config.CHROMA_DIR}")

    print("\nNext: python steps/04_retrieve_only.py")


if __name__ == "__main__":
    main()
