"""Step 2 of RAG: turn chunks into vectors and store them so we can search.

A "vector store" is a database whose lookup key is meaning, not exact text.
Each chunk becomes a list of numbers (an embedding); searching means finding
the stored vectors closest to the vector of your question.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag import config


def get_embeddings() -> HuggingFaceEmbeddings:
    """The model that converts text -> vector.

    The SAME model must be used for indexing and for querying, otherwise the
    two sets of vectors live in different spaces and distances are meaningless.
    That is why this is a function everything else calls, not a copy-paste.
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def build_store(chunks: list[Document]) -> Chroma:
    """Embed every chunk and write the index to disk.

    `persist_directory` is what makes this survive between runs - without it
    Chroma would be in-memory and you would pay the embedding cost every time.
    """
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )


def load_store() -> Chroma:
    """Open the index that build_store() already wrote. No re-embedding."""
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No vector store at {config.CHROMA_DIR}.\n"
            "Run steps/03_embed_and_store.py first."
        )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def get_retriever(k: int | None = None):
    """Wrap the store in a `Retriever`.

    A Retriever is LangChain's generic "given a question, give me documents"
    interface. Chains accept any retriever, so you could swap Chroma for a
    keyword search or a web search without touching the chain code.
    """
    return load_store().as_retriever(search_kwargs={"k": k or config.TOP_K})
