"""All tunable settings in one place.

Every value here is something you will want to change while experimenting,
so nothing else in the project hard-codes these numbers.
"""

from pathlib import Path

# --- paths -----------------------------------------------------------------
# Path(__file__) is this file; .parent is rag/; .parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENT_PATH = DATA_DIR / "companyPolicies.txt"
DOCUMENT_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "6JDbUb_L3egv_eOkouY71A.txt"
)
# Where Chroma writes the vector index so we don't re-embed on every run.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "company_policies"

# --- chunking --------------------------------------------------------------
# Documents are split before embedding: one vector per chunk, not per document.
# Smaller chunks = more precise retrieval but less surrounding context.
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 100     # characters repeated between neighbours, so a sentence
                        # split across a boundary still appears whole somewhere

# --- embeddings ------------------------------------------------------------
# Downloaded once from HuggingFace (~90 MB) and then run locally on CPU.
# It turns text into a 384-dimension vector. Small and fast; good enough here.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- retrieval -------------------------------------------------------------
TOP_K = 4               # how many chunks to fetch and feed to the LLM

# --- LLM (local, via Ollama) -----------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:8b"      # any model listed by `ollama list`
# Qwen3 "thinks" before answering unless told not to. Measured on this laptop:
# a one-word reply took 8.1s with thinking and 0.5s without. RAG answers come
# from the retrieved chunks, so the hidden reasoning buys little here.
THINKING = False
TEMPERATURE = 0             # 0 = deterministic. Raise for more creative answers.
NUM_PREDICT = 256           # max tokens to generate
NUM_CTX = 4096              # context window: must fit TOP_K chunks + the question
