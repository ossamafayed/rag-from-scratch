# RAG, step by step

Written for someone comfortable with React/Next and Java who is newer to
Python. Each section maps to one file in `steps/`. Run the file, read the
section, then change the thing it suggests and run it again.

---

## The one-paragraph version of RAG

An LLM only knows what was in its training data. It has never seen your
company's policy document. **Retrieval-Augmented Generation** fixes that
without retraining anything: when a question arrives, you *search* your own
documents for the few paragraphs most likely to contain the answer, paste
those paragraphs into the prompt, and ask the model to answer using only
them.

Two phases, and it's worth keeping them separate in your head:

| Phase | When it runs | What it does | Files |
|---|---|---|---|
| **Indexing** | Once, up front | document to chunks to vectors to stored | `loader.py`, `store.py` |
| **Retrieval** | On every question | question to vector to nearest chunks to prompt to LLM | `chain.py`, `llm.py` |

---

## Python notes for a Java/JS developer

A few things in this code that don't look like what you're used to:

- **`rag/` is a package** because it contains `__init__.py`. `from rag import config` is roughly Java's `import com.example.rag.Config`.
- **No classes required.** `loader.py` is just functions in a file. A module *is* the unit of organisation — you don't need a class to hold static methods the way Java does.
- **`if __name__ == "__main__":`** is `public static void main`. It runs only when the file is executed directly, not when it's imported.
- **Type hints (`-> list[Document]`) are documentation, not enforcement.** Nothing checks them at runtime. They exist for your editor.
- **`steps/_bootstrap.py`** exists only to let the scripts see the `rag` package. Python searches the *script's own folder* for imports, not the project root — unlike Node resolving up through `node_modules`.
- **The venv** (`.venv/`) is `node_modules` for Python: a per-project dependency folder. `.venv/Scripts/python.exe` is that project's own interpreter, isolated from your system Python.

---

## Step 1 — `01_check_setup.py`

Verifies packages are importable and Ollama is reachable with your model
pulled. Nothing conceptual, but it turns two very common and very confusing
failures into a clear message.

**The lesson underneath it:** the notebook version of this lab failed with
`ModuleNotFoundError` because a `%%capture` hid a failed install. Check your
environment explicitly instead of assuming.

---

## Step 2 — `02_load_and_chunk.py`

Downloads the policy document and splits it into chunks. No AI yet.

**Why chunk?** Two hard constraints:

1. One embedding for a whole document is a blurry average of every topic in it, so it matches no specific question well.
2. The model's context window is finite. You can only paste a few thousand characters into a prompt, so the pieces must be small and on-topic.

`RecursiveCharacterTextSplitter` tries paragraph breaks first, then line
breaks, then words — so chunks end at natural seams. `CHUNK_OVERLAP` repeats
a little text between neighbours so a sentence straddling a boundary still
appears intact in at least one chunk.

> **Try it:** set `CHUNK_SIZE = 300` in `rag/config.py` and re-run. More
> chunks, each narrower. Then try `3000`. Later, notice how this changes the
> answers in step 5 — chunking quality caps everything downstream.

---

## Step 3 — `03_embed_and_store.py`

The heart of it. An **embedding** converts text into a list of numbers (here,
384 of them) that describes what the text *means*. Text about similar things
lands close together in that 384-dimensional space.

That's what makes this different from `Ctrl+F`. A question about "cell phone
rules" can match a chunk that says "mobile device usage" — no shared keywords,
similar meaning, nearby vectors.

The step prints an actual vector so it stops being abstract, then embeds every
chunk into a **Chroma** store persisted in `chroma_db/`. Think of it as a
database whose index is meaning rather than exact values.

**One rule you must not break:** the same embedding model has to be used for
indexing and for querying. Different models produce different number spaces,
and distances between them are meaningless. That's why `get_embeddings()` is
one shared function.

---

## Step 4 — `04_retrieve_only.py`

Search with **no LLM at all**. This is the most valuable debugging habit in
RAG: when an answer is wrong, look at the retrieved chunks first. If the
needed text isn't in that list, the model never saw it and no amount of prompt
tuning will help.

The third query is deliberately about something the document doesn't contain.
Watch what happens: **the store still returns four chunks.** A vector search
always returns its nearest neighbours, however far away they are. There's no
"no results" — only worse distances. (Scores here are distances: lower is
closer.)

That fact is exactly why step 6 matters.

> **Try it:** change `TOP_K` in config. Too low and you miss the answer; too
> high and you flood the prompt with noise and hit the context limit.

---

## Step 5 — `05_ask_with_llm.py`

Now the full pipeline. The chain, in `rag/chain.py`:

```python
RunnableParallel(
    context=retriever | format_docs,   # question -> chunks -> one string
    question=RunnablePassthrough(),    # question, untouched
)
| ANSWER_PROMPT                        # fill {context} and {question}
| get_llm()                            # send to the local model
| StrOutputParser()                    # pull the plain string out
```

The `|` is LangChain's LCEL, and it reads like a shell pipe: output of each
step feeds the next. `RunnableParallel` runs both branches on the same input
and produces a dict whose keys line up with the prompt's placeholders.

This version also returns the source chunks. Compare each answer against them
and you can see precisely which sentences the model paraphrased. Being able
to show sources is most of the point of RAG.

**Then the step deliberately fails.** It asks *"Summarize the whole document
in three sentences"* — and gets `I don't know`.

That is not a bug, it's the shape of the tool. Summarizing a whole document
is not a retrieval question: the retriever can only fetch `TOP_K` (4) chunks
out of 26, chosen by similarity to the word "summarize", which resembles
nothing in particular. The model is handed a random-ish 15% of the document
and asked to summarize all of it, so a well-behaved model refuses.

No prompt fixes this. A different technique does — map-reduce summarization
(summarize every chunk, then summarize the summaries, with no retrieval step
at all), or just feeding the whole document in if it fits the context window.

> RAG answers *"what does the document say about X"*.
> It does not answer *"what does the document say"*.

The original notebook asked exactly this question and treated whatever came
back as a success. Worth knowing which of your questions RAG is even the
right tool for.

---

## Step 6 — `06_prompt_matters.py`

Same retriever, same model, same question — three prompts: **naive** (no
instruction about grounding), **reckless** (explicitly allowed to fall back on
general knowledge), and **grounded** (use only the context, or say "I don't
know").

The question asks about parental leave, which isn't in the document. Step 4
already showed the retriever will hand over four irrelevant chunks regardless.

**Be honest about what you'll see:** on `llama3.2:3b` all three prompts
usually behave — even the reckless one tends to say it can't find the
information. Modern instruction-tuned models are cautious by default. That is
*luck*, and it varies with the model, the question, and the temperature. Try
the same step with a different `LLM_MODEL` and you may well see a confident
invented number.

The point stands regardless: the grounded prompt is what turns good behaviour
from a hope into a property of your system. In production you'd add a distance
threshold too — step 4 showed the off-topic query sitting around 1.85, so if
the nearest chunk is beyond some cutoff, skip the LLM call entirely.

---

## Step 7 — `07_chat_with_memory.py`

Follow-up questions break naive RAG. Ask "What is the mobile policy?" then
"What does it cover?" — that second question sent to the retriever contains no
searchable content. "It" means nothing to a vector search.

The fix, in `ChatRAG`: use the LLM to **rewrite the follow-up into a
standalone question** using the conversation so far, then run ordinary RAG on
the rewrite. The step prints each rewrite so you can watch it happen.

So chat RAG costs two model calls per turn: one to condense, one to answer.
I wrote this as a small class rather than a LangChain memory object precisely
because the logic is short enough to read — that *is* the lesson.

---

## Step 8 — `08_interactive.py`

Ask your own questions. Try:

- something answered directly in the document
- something related but absent (does it correctly say "I don't know"?)
- a follow-up using "it" or "that"
- `reset`, then the same follow-up — it should now fail, because the history it depended on is gone

---

## Step 9 — `09_inspect_store.py`

Not part of the pipeline — a microscope for the index. Run it any time.

```bash
python steps/09_inspect_store.py            # all 26 chunks, with lengths
python steps/09_inspect_store.py 14         # one chunk: text + its vector
python steps/09_inspect_store.py --vectors  # raw numbers for the first 5
python steps/09_inspect_store.py --compare  # chunk-to-chunk similarity
```

**Where the data physically lives.** `chroma_db/` holds two different things:

| File | Contains | Viewable with |
|---|---|---|
| `chroma.sqlite3` | ids, chunk **text**, metadata | any SQLite viewer |
| `*/data_level0.bin` | the **vectors** (HNSW index) | nothing but the API |

So a SQLite browser shows you the chunks and never the numbers. The
`embeddings` table has columns `id, segment_id, embedding_id, seq_id,
created_at` — note there is no vector column. That's why this script exists.

**What the overview reveals about our own index:** 8 of the 26 chunks are bare
section headings of 17–36 characters — `1. Code of Conduct`, `5. Smoking
Policy`. They carry almost no meaning but still compete for the `TOP_K` slots
in every single search. Look back at step 5: chunk `4. Mobile Phone Policy`
was retrieved as source [1] for the mobile phone question, contributing
nothing but the title. Roughly a quarter of our retrieval budget is spent on
headings.

That's a real chunking defect, and it's the most valuable thing in this whole
project to have noticed yourself. Fixing it — filtering out short chunks, or
attaching each heading to the section beneath it — would measurably improve
every answer.

`--compare` shows cosine similarity (1.0 = identical direction, 0 =
unrelated). Neighbouring chunks score high partly because `CHUNK_OVERLAP`
duplicates text between them. Distant chunks scoring high means genuine topic
overlap — which is exactly what retrieval exploits.

### The official browser

Chroma ships a terminal UI:

```bash
.venv/Scripts/chroma.exe browse company_policies --path chroma_db
```

Honest status: it launched for me but reported *"Failed to load records"*, both
against the path directly and against a running `chroma run` server. The data
itself is fine — the same collection reads perfectly through the Python API
(26 records, 384 dimensions). It's a full-screen TUI, so it may simply need a
real interactive terminal rather than the piped shell I tested it in. Worth
trying in your own terminal; step 9 is the reliable option either way.

---

## Step 10 — `10_stop_hallucinating.py`

**The bug you hit.** Type `ok thanks` or `hi how u doin` into step 8's naive
chat and it answers a policy question you never asked.

**Where it actually comes from — and it is not where you'd guess.** Not the
answering model; step 6's grounded prompt behaves perfectly. It's the
**condenser**, upstream. Real trace from part 1:

| You typed | What it searched for | Distance |
|---|---|---|
| `What is the smoking policy?` | *(unchanged)* | 0.292 |
| `ok thanks` | "What is the purpose of the Smoking Policy?" | 0.469 |
| `hi how u doin` | "How can I find more information about the Smoking Policy?" | 0.514 |
| `no more` | "What is the purpose of the Smoking Policy?" | 0.469 |

Step 7's condenser is instructed to *"rewrite the follow-up as a standalone
question"* and given **no way to decline**. Handed a greeting, it invents a
plausible question. Everything downstream then answers that invented question
perfectly — the damage happened before retrieval ever ran.

Note the distances are **low**. The invented questions really are about the
document, so a relevance threshold alone never catches this. You have to stop
the question being invented.

### The fix, in order of impact

**1. Route first.** Classify the message before doing RAG on it. Greetings and
thanks never reach the retriever. See [`rag/router.py`](rag/router.py) —
deterministic sets for the obvious cases, an LLM classifier for the rest.

**2. Let every LLM step say no.** The condenser now has an explicit
`NOT_A_QUESTION` output. *Any prompt that must always produce output will
invent output* — this applies to every call in a pipeline, not just the last.

**3. Gate on distance.** If the nearest chunk is beyond `DISTANCE_THRESHOLD`
(1.2 here), refuse without calling the LLM. Calibrate on your own corpus with
step 4; on ours, on-topic ≈0.3–0.6, absent ≈1.0–1.5, off-topic ≈1.8+.

**4. Keep small talk out of history.** Anything in history is material the
condenser can misread next turn.

**5. Answer from context only.** Step 6's prompt — the *last* line of defence,
not the first.

### Two bugs the guards themselves introduced

Part 3 exists because **a guard that refuses everything is not a fix**, and
building this caught exactly that, twice:

- The LLM classifier labelled *"And what happens if I lose the phone?"* as
  CHITCHAT. Fixed with a deterministic rule — a message containing `?` that
  isn't a known greeting is always a question — checked *before* the LLM, plus
  few-shot examples and an explicit "when in doubt, say DOCUMENT".
- The condenser returned `NOT_A_QUESTION` for *"Can I smoke in a company
  car?"*, refusing a perfectly good question. Fixed by precedence: the router
  judged the message *without* history to confuse it, so it overrules the
  condenser, and we search with the message unchanged.

Both errors were in the expensive direction — silently refusing real
questions. When two guards disagree, prefer the one that fails toward
answering, and let the distance gate catch what slips through.

> **The general principle:** in a multi-step LLM pipeline, an error at step 1
> is faithfully executed by every step after it. Debug from the **front**, and
> print the intermediate values. That's how the condenser was caught here —
> the answer looked reasonable; only the rewritten query revealed the lie.

Step 8 now uses the guarded version by default. Compare them yourself:

```bash
python steps/08_interactive.py --trace    # guarded, shows the path taken
python steps/08_interactive.py --naive    # the original, to reproduce the bug
```

---

## Where to go next

1. **Swap the model.** `LLM_MODEL = "mistral:7b"` in config. Bigger model, better answers, slower. You have `mistral:7b` and `granite3.3:2b` pulled.
2. **Use your own documents.** Point `DOCUMENT_PATH` at your own file, delete `chroma_db/`, re-run step 3. For PDFs, `PyPDFLoader` from `langchain_community.document_loaders` replaces `TextLoader`.
3. **Add a distance threshold** so genuinely irrelevant matches never reach the LLM.
4. **Try a better embedding model** — `all-mpnet-base-v2` is larger and more accurate than the MiniLM used here. Remember to delete `chroma_db/` and rebuild: you cannot mix embedding models in one index.
