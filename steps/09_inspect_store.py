"""STEP 9 - Look inside the vector store: the chunks AND the vectors.

Run:  python steps/09_inspect_store.py            # overview of all chunks
      python steps/09_inspect_store.py 7          # everything about chunk 7
      python steps/09_inspect_store.py --vectors  # dump raw vectors
      python steps/09_inspect_store.py --compare  # which chunks resemble which

The store is not a black box. Chroma keeps two things in chroma_db/:

  chroma.sqlite3   - the ids, the chunk TEXT, and metadata. Openable in any
                     SQLite viewer (DB Browser for SQLite, the VS Code SQLite
                     extension, `sqlite3` on the command line).
  *.bin files      - the HNSW index: the actual VECTORS, in a binary format
                     no SQLite viewer can read.

So a SQLite browser shows you the chunks but never the numbers. To see the
vectors you have to go through the API - which is what this script does.
"""

import _bootstrap  # noqa: F401

import sys

from rag import config
from rag.store import load_store


def get_collection():
    """The raw Chroma collection underneath the LangChain wrapper.

    LangChain hides this to keep its interface generic; for inspection we
    want the real thing, because it can return embeddings.
    """
    return load_store()._collection


def overview() -> None:
    col = get_collection()
    data = col.get(include=["documents", "metadatas"])
    ids = data["ids"]

    print(f"collection : {config.COLLECTION_NAME}")
    print(f"location   : {config.CHROMA_DIR}")
    print(f"chunks     : {len(ids)}")
    print(f"dimensions : {col.get(limit=1, include=['embeddings'])['embeddings'].shape[1]}")
    print()
    print(f"{'#':>3}  {'chars':>5}  text")
    print("-" * 100)
    for i, (doc, _meta) in enumerate(zip(data["documents"], data["metadatas"])):
        preview = doc[:80].replace("\n", " ")
        print(f"{i:>3}  {len(doc):>5}  {preview}")
    print()
    print("Chunks with very few characters are usually bare headings - they")
    print("carry almost no meaning, yet still occupy a slot in every search.")
    print()
    print("Run with a number (e.g. `09_inspect_store.py 7`) to inspect one chunk.")


def show_chunk(index: int) -> None:
    col = get_collection()
    data = col.get(include=["documents", "metadatas", "embeddings"])

    if not 0 <= index < len(data["ids"]):
        print(f"No chunk {index}. There are {len(data['ids'])} (0-{len(data['ids']) - 1}).")
        return

    doc = data["documents"][index]
    vec = data["embeddings"][index]

    print(f"CHUNK {index}")
    print(f"id       : {data['ids'][index]}")
    print(f"metadata : {data['metadatas'][index]}")
    print(f"length   : {len(doc)} characters")
    print("=" * 70)
    print(doc)
    print("=" * 70)
    print(f"\nVECTOR ({len(vec)} dimensions)")
    print(f"  first 12 : {[round(float(v), 4) for v in vec[:12]]}")
    print(f"  min      : {float(vec.min()):.4f}")
    print(f"  max      : {float(vec.max()):.4f}")
    print(f"  length   : {float((vec ** 2).sum() ** 0.5):.4f}  (normalised vectors are 1.0)")
    print("\nThat list of numbers IS the chunk, as far as search is concerned.")
    print("Nothing in it maps to a word - each dimension is a learned feature")
    print("with no human-readable name.")


def dump_vectors(limit: int = 5, width: int = 8) -> None:
    col = get_collection()
    data = col.get(limit=limit, include=["documents", "embeddings"])

    print(f"First {limit} chunks as raw vectors (showing {width} of "
          f"{data['embeddings'].shape[1]} dimensions)\n")
    for i, (doc, vec) in enumerate(zip(data["documents"], data["embeddings"])):
        preview = doc[:45].replace("\n", " ")
        numbers = "  ".join(f"{float(v):+.3f}" for v in vec[:width])
        print(f"[{i}] {numbers}  ...   {preview}")
    print("\nEyeballing the numbers tells you nothing - which is the point.")
    print("Meaning lives in the DISTANCE between vectors, never in one vector.")
    print("Run with --compare to see that distance do something useful.")


def compare() -> None:
    """Show the chunk-to-chunk similarity that makes retrieval work."""
    col = get_collection()
    data = col.get(include=["documents", "embeddings"])
    vectors = data["embeddings"]
    docs = data["documents"]

    # Cosine similarity: normalise each vector, then take the dot product.
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    unit = vectors / norms[:, None]
    similarity = unit @ unit.T

    print("For each chunk, its most similar OTHER chunk:\n")
    for i in range(len(docs)):
        row = similarity[i].copy()
        row[i] = -1                      # exclude the chunk itself
        j = int(row.argmax())
        print(f"[{i:>2}] {docs[i][:42]!r:46} ~ [{j:>2}] {docs[j][:42]!r}")
        print(f"     cosine similarity {row[j]:.3f}")
    print("\nHigh similarity between neighbouring chunks is the overlap doing")
    print("its job. High similarity between DISTANT chunks means they discuss")
    print("the same topic - exactly what a search is meant to surface.")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        overview()
    elif args[0] == "--vectors":
        dump_vectors()
    elif args[0] == "--compare":
        compare()
    elif args[0].lstrip("-").isdigit():
        show_chunk(int(args[0]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
