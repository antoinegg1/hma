"""Publicly buildable preparation images; no cluster-specific dependencies."""

import os
from pathlib import Path

BASE_RUNTIME_IMAGE = os.environ.get("HMA_EVALUATOR_IMAGE", "hma-evaluator:local")
FULL_EVALUATOR_IMAGE = BASE_RUNTIME_IMAGE


def credential_home() -> Path:
    """Return the user's home for Kaggle's standard credential lookup."""
    return Path.home()
