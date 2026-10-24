"""Step 1 of RAG: get raw text in, and cut it into chunks.

This is the "Indexing" half of RAG. Nothing here talks to an LLM.
"""

from urllib.request import urlretrieve

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config


def download_document() -> None:
    """Fetch the sample document if we don't already have it."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.DOCUMENT_PATH.exists():
        print(f"already downloaded: {config.DOCUMENT_PATH.name}")
        return
    print(f"downloading {config.DOCUMENT_URL} ...")
    urlretrieve(config.DOCUMENT_URL, config.DOCUMENT_PATH)
    print(f"saved to {config.DOCUMENT_PATH}")


def load_document() -> list[Document]:
    """Read the file into LangChain `Document` objects.

    A Document is just `page_content` (the text) plus `metadata` (a dict).
    A whole .txt file loads as a single Document.
    """
    loader = TextLoader(str(config.DOCUMENT_PATH), encoding="utf-8")
    return loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    """Cut long Documents into overlapping chunks.

    Why chunk at all? Two reasons:
      1. An embedding vector for a whole 20-page document is a blurry average
         of everything in it, so it matches nothing well.
      2. The LLM has a limited context window - we can only paste a few
         chunks into the prompt, so they need to be small and on-topic.

    RecursiveCharacterTextSplitter tries to break on paragraph boundaries
    first, then lines, then words - so chunks tend to end at natural seams
    rather than mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def load_and_split() -> list[Document]:
    """The whole ingest path in one call: download -> load -> chunk."""
    download_document()
    return split_documents(load_document())
