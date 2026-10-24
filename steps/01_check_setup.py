"""STEP 1 - Is everything actually installed and running?

Run:  python steps/01_check_setup.py

Checks the two things that break most often: missing packages, and Ollama
not running. Fix anything reported here before moving on.
"""

import _bootstrap  # noqa: F401  (imported for its sys.path side effect)

import importlib.util
import sys

from rag import config

PACKAGES = [
    "langchain", "langchain_core", "langchain_community",
    "langchain_text_splitters", "langchain_classic", "langchain_chroma",
    "chromadb", "langchain_huggingface", "sentence_transformers",
    "langchain_ollama",
]


def main() -> int:
    print(f"python: {sys.executable}\n")

    print("packages:")
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    for package in PACKAGES:
        status = "MISSING" if package in missing else "ok"
        print(f"  {package:30} {status}")
    if missing:
        print("\n  -> pip install -r requirements.txt")
        return 1

    print("\nollama:")
    from rag.llm import check_ollama  # imported late so a missing package
                                      # reports as MISSING above, not a crash
    try:
        models = check_ollama()
    except RuntimeError as exc:
        print(f"  {exc}")
        return 1
    print(f"  running at {config.OLLAMA_BASE_URL}")
    print(f"  models: {models}")
    print(f"  using: {config.LLM_MODEL}")

    print("\nAll good. Next: python steps/02_load_and_chunk.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
