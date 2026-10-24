"""Lets `python steps/0X_....py` import the sibling `rag` package.

Python only searches the script's own folder (steps/), not the project root,
so we add the root to sys.path. In a packaged app you'd `pip install -e .`
instead; this keeps the project dependency-free to run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
