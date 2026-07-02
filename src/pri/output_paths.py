"""Default output directory helpers."""

from __future__ import annotations

import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def _topic_dir_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_") or "topic"


def default_output_dir(from_date: str, label: str) -> str:
    year = from_date[:4]
    return os.path.join(RESULTS_DIR, year, _topic_dir_slug(label))
