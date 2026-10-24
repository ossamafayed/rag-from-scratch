# RAG from scratch (local, no cloud)

A step-by-step RAG project built from the "Summarize Private Documents"
notebook, rewritten as a real Python project. Everything runs on your machine:
the embedding model is local, the vector store is a local folder, and the LLM
is served by your local [Ollama](https://ollama.com/). No API keys, and the
document never leaves your laptop.

**Stack:** LangChain · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · Ollama (`qwen3:8b`)

## What it covers

| Concept | Step |
|---|---|
| Loading and chunking (size vs overlap trade-off) | 02 |
| Embedding into a persistent Chroma store | 03 |
| Retrieval on its own, before any LLM is involved | 04 |
| Grounded answering with sources | 05 |
| How much the prompt actually matters | 06 |
| Conversational memory and question condensing | 07–08 |
| Inspecting stored chunks and raw vectors | 09 |
| Stopping hallucination: small-talk routing + a distance threshold | 10 |

Step 05 also explains why *"summarize the whole document"* is the wrong job
for RAG: retrieval only ever sees `TOP_K` chunks, so a whole-document summary
needs map-reduce instead.

## Prerequisites

Ollama running with a model pulled:

```bash
ollama serve          # usually already running as a service
ollama pull qwen3:8b  # any model works: change LLM_MODEL in rag/config.py
```

Qwen3's "thinking" mode is switched off in [`rag/config.py`](rag/config.py)
(`THINKING = False`). I measured a one-word reply at 8.1 s with thinking on
and 0.5 s with it off, and in RAG the facts come from the retrieved chunks anyway.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

## Run the steps in order

```bash
.venv/Scripts/python.exe steps/01_check_setup.py
.venv/Scripts/python.exe steps/02_load_and_chunk.py
.venv/Scripts/python.exe steps/03_embed_and_store.py
.venv/Scripts/python.exe steps/04_retrieve_only.py
.venv/Scripts/python.exe steps/05_ask_with_llm.py
.venv/Scripts/python.exe steps/06_prompt_matters.py
.venv/Scripts/python.exe steps/07_chat_with_memory.py
.venv/Scripts/python.exe steps/08_interactive.py
.venv/Scripts/python.exe steps/10_stop_hallucinating.py
```

Step 8 accepts `--trace` (show which path each message takes) and `--naive`
(use the unguarded chat, to reproduce the hallucination step 10 fixes).

Plus an inspector you can run at any time to see what is actually stored:

```bash
.venv/Scripts/python.exe steps/09_inspect_store.py            # all chunks
.venv/Scripts/python.exe steps/09_inspect_store.py 14         # one chunk + its vector
.venv/Scripts/python.exe steps/09_inspect_store.py --vectors  # raw numbers
.venv/Scripts/python.exe steps/09_inspect_store.py --compare  # chunk similarity
```

Read `WALKTHROUGH.md` alongside them - it explains what each step teaches and
what to change to see the concept move.

## Layout

```
rag/                 the reusable library
  config.py          every tunable setting
  loader.py          download, load, chunk
  store.py           embed + Chroma vector store
  llm.py             local Ollama client
  chain.py           retriever + prompt + LLM wiring
  router.py          is this a question or small talk?
  guarded.py         chat with routing, safe condensing, relevance gate
steps/               numbered lessons, run in order
data/                the source document (downloaded by step 02)
chroma_db/           the vector index (generated)
```

---

Built while studying *Build RAG Applications* in IBM's RAG and Agentic AI
Professional Certificate.
