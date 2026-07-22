"""Shared LLM fallback helpers for PRI paper text prompts."""

from __future__ import annotations

import re
from typing import NamedTuple

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def sanitize_paper_text(text: str) -> str:
    """Strip simple HTML tags from titles/abstracts before LLM prompting."""
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def abstract_sentence_fallbacks(abstract: str) -> list[str]:
    """Return progressively shorter abstract snippets for provider fallbacks."""
    text = abstract.strip()
    if not text:
        return []
    sentences = [part.strip() for part in text.split(".") if part.strip()]
    if not sentences:
        return []
    return [sentences[0] + "."]


class PaperTextAttempt(NamedTuple):
    mode: str
    title: str
    abstract: str
    use_json_format: bool


def build_paper_text_attempts(title: str, abstract: str) -> list[PaperTextAttempt]:
    """Ordered fallbacks when the provider returns empty or unparseable content."""
    title = sanitize_paper_text(title)
    abstract = sanitize_paper_text(abstract)
    attempts = [
        PaperTextAttempt("full", title, abstract, True),
        PaperTextAttempt("full", title, abstract, False),
    ]
    for snippet in abstract_sentence_fallbacks(abstract):
        attempts.append(PaperTextAttempt("abstract_snippet", title, snippet, True))
    attempts.append(PaperTextAttempt("title_only", title, "", True))
    return attempts
