"""Pytest smoke tests for notebook execution in CI."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "0_exploration.ipynb"


def test_notebook_0_exploration_executes() -> None:
    """Execute the exploration notebook and fail on any cell error."""
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=900,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
