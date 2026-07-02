#!/usr/bin/env python3
"""
Find top papers by 18m citations, filtered by LLM topic-relevance check.

Fetches an oversized candidate pool from OpenAlex, classifies each paper with an LLM,
and returns the top N papers that are substantively about the target topic.

Example:
    uv run python src/pri/find_top_by_18m_citations_llm_filtered.py \\
        --topic "T11287,Cancer Genomics and Diagnostics" -n 20 \\
        --from-date 2024-01-01 --to-date 2024-12-31
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pri.output_paths import default_output_dir
from pri.openalex_client import (
    CITATION_WINDOW_MONTHS,
    find_top_n_by_citation_window,
    parse_topic_spec,
    resolve_topic_id,
    validate_topic,
)
from utils.llm_processor import LLMProcessor, LLMRetryExhausted

DEFAULT_LIMIT = 20
DEFAULT_INITIAL_MULTIPLIER = 1.5
DEFAULT_MAX_MULTIPLIER = 5.0
DEFAULT_CITATION_INITIAL_MULTIPLIER = DEFAULT_INITIAL_MULTIPLIER
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_MULTIPLIER_LADDER = [1.5, 2.0, 3.0, 4.0, 5.0]

TOPIC_RELEVANCE_PROMPT_TEMPLATE = """\
You are a scientific literature classifier. Your task is to decide whether a scholarly paper's research is about a given topic.

## Topic
{topic}

## Paper
**Title:** {title}

**Abstract:**
{abstract}

## Classification rules
- Return `true` if the paper studies, investigates, analyzes, develops methods for, or applies methods to the given topic.
- Return `true` if the topic is a part of the paper, even if the paper also covers broader or additional subjects.
- Return `false` if the topic is unrelated to the research.
- Review papers count as `true` when the review substantially covers the topic.
- If the abstract is missing or very short, use the title to make your best judgment.

## Output
Respond with a single JSON object only. No markdown, no explanation, and no extra keys.

```json
{{"is_about_the_topic": true}}
```

or

```json
{{"is_about_the_topic": false}}
```
"""

TOPIC_RELEVANCE_SYSTEM_PROMPT = (
    "You are a scientific literature classifier. "
    "Respond with a single valid JSON object containing only the key "
    '"is_about_the_topic" with a boolean value. '
    "No markdown, no explanation, no extra keys."
)


def default_topic_filter_model() -> str:
    """Model for topic relevance checks from OPENROUTER_PRI_TOPIC_CHECK_MODEL."""
    model = (os.getenv("OPENROUTER_PRI_TOPIC_CHECK_MODEL") or "").strip()
    if not model:
        raise ValueError(
            "OPENROUTER_PRI_TOPIC_CHECK_MODEL is required for topic relevance checks. "
            "Set it in .env."
        )
    return model


def build_multiplier_ladder(
    initial_multiplier: float,
    max_multiplier: float,
    *,
    ladder: list[float] | None = None,
) -> list[float]:
    """Return multipliers from initial up to max, using default ladder steps."""
    steps = ladder or DEFAULT_MULTIPLIER_LADDER
    result = [m for m in steps if initial_multiplier <= m <= max_multiplier]
    if not result or result[0] != initial_multiplier:
        if initial_multiplier <= max_multiplier:
            result = [initial_multiplier] + [m for m in result if m > initial_multiplier]
    if not result:
        result = [min(initial_multiplier, max_multiplier)]
    if result[-1] < max_multiplier and max_multiplier not in result:
        result.append(max_multiplier)
    return sorted(set(result))


def build_relevance_prompt(
    topic: str,
    title: str,
    abstract: str,
    *,
    template: str | None = None,
) -> str:
    text = template if template is not None else TOPIC_RELEVANCE_PROMPT_TEMPLATE
    abstract_text = abstract.strip() if abstract and abstract.strip() else "(not available)"
    return text.format(
        topic=topic.strip(),
        title=(title or "").strip() or "(not available)",
        abstract=abstract_text,
    )


def coerce_is_about_the_topic(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _validate_topic_relevance_fields(fields: dict[str, Any]) -> bool:
    if "is_about_the_topic" not in fields:
        raise ValueError("LLM response missing 'is_about_the_topic'")
    return coerce_is_about_the_topic(fields["is_about_the_topic"])


def _work_sort_key(work: dict[str, Any]) -> tuple[int, int]:
    return (
        int(work.get("citations_within_window") or 0),
        int(work.get("cited_by_count") or 0),
    )


async def filter_works_by_topic_llm(
    works: list[dict[str, Any]],
    topic_label: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    progress_callback: Callable[[int, int], None] | None = None,
    llm_enricher: Callable[[list[dict[str, Any]], str], Any] | None = None,
) -> list[dict[str, Any]]:
    """LLM-classify each work; merge is_about_the_topic into work dicts."""
    if not works:
        return []

    if llm_enricher is not None:
        return await llm_enricher(works, topic_label)

    processor = LLMProcessor(
        model=model or default_topic_filter_model(),
        base_url=base_url,
        max_tokens=64,
    )
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(works)

    async def classify_one(work: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        async with semaphore:
            prompt = build_relevance_prompt(
                topic_label,
                work.get("title") or "",
                work.get("abstract") or "",
            )
            work_id = work.get("openalex_id") or "unknown"
            title_preview = (work.get("title") or "")[:80]
            is_about_the_topic: bool | None = None
            last_error: BaseException | None = None

            for attempt in range(2):
                try:
                    response_text = await processor._call_llm(
                        prompt, TOPIC_RELEVANCE_SYSTEM_PROMPT
                    )
                    fields = processor._parse_json_response(response_text or "")
                    is_about_the_topic = _validate_topic_relevance_fields(fields)
                    break
                except (
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                    LLMRetryExhausted,
                ) as exc:
                    last_error = exc
                    if attempt == 0:
                        continue
                    raise RuntimeError(
                        f"Topic relevance LLM failed for {work_id!r} "
                        f"({title_preview!r}) after retry: {exc}"
                    ) from exc

            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return {
                **work,
                "is_about_the_topic": is_about_the_topic,
            }

    return list(await asyncio.gather(*[classify_one(work) for work in works]))


def fetch_and_filter_top_n(
    *,
    query: str,
    n: int,
    topic_label: str,
    from_publication_date: str,
    to_publication_date: str,
    topic_id: str | None = None,
    articles_only: bool = True,
    months: int = CITATION_WINDOW_MONTHS,
    use_cache: bool = True,
    initial_multiplier: float = DEFAULT_INITIAL_MULTIPLIER,
    max_multiplier: float = DEFAULT_MAX_MULTIPLIER,
    citation_initial_multiplier: float = DEFAULT_CITATION_INITIAL_MULTIPLIER,
    model: str | None = None,
    base_url: str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    fetch_fn: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    llm_enricher: Callable[[list[dict[str, Any]], str], Any] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    multiplier_ladder: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Fetch candidates with escalating pool multipliers until >= n papers pass LLM filter.

    Returns (top_n_works, evaluation_pool, combined_metadata).
    """
    fetch = fetch_fn or find_top_n_by_citation_window
    multipliers = build_multiplier_ladder(
        initial_multiplier, max_multiplier, ladder=multiplier_ladder
    )
    evaluated_by_id: dict[str, dict[str, Any]] = {}
    last_openalex_metadata: dict[str, Any] = {}
    multiplier_used = multipliers[0]
    pool_size_fetched = 0

    for multiplier in multipliers:
        pool_size = max(n, math.ceil(multiplier * n))
        works, openalex_metadata = fetch(
            query=query,
            n=pool_size,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
            initial_multiplier=citation_initial_multiplier,
            topic_id=topic_id,
            articles_only=articles_only,
            months=months,
            use_cache=use_cache,
        )
        last_openalex_metadata = openalex_metadata
        multiplier_used = multiplier
        pool_size_fetched = pool_size

        new_works = [
            work for work in works if work.get("openalex_id") not in evaluated_by_id
        ]
        if new_works:
            enriched = asyncio.run(
                filter_works_by_topic_llm(
                    new_works,
                    topic_label,
                    model=model,
                    base_url=base_url,
                    max_concurrent=max_concurrent,
                    progress_callback=progress_callback,
                    llm_enricher=llm_enricher,
                )
            )
            for work in enriched:
                evaluated_by_id[work["openalex_id"]] = work

        passed = [
            work
            for work in sorted(evaluated_by_id.values(), key=_work_sort_key, reverse=True)
            if work.get("is_about_the_topic")
        ]
        if len(passed) >= n:
            break

    evaluation_pool = sorted(evaluated_by_id.values(), key=_work_sort_key, reverse=True)
    passed_works = [work for work in evaluation_pool if work.get("is_about_the_topic")]
    top_works = passed_works[:n]
    sufficient = len(passed_works) >= n

    llm_filter_meta = {
        "topic_label": topic_label,
        "target_n": n,
        "initial_multiplier": initial_multiplier,
        "max_multiplier": max_multiplier,
        "llm_pool_initial_multiplier": initial_multiplier,
        "llm_pool_max_multiplier": max_multiplier,
        "multiplier_ladder": multipliers,
        "multiplier_used": multiplier_used,
        "llm_pool_multiplier_ladder": multipliers,
        "llm_pool_multiplier_used": multiplier_used,
        "citation_initial_multiplier": citation_initial_multiplier,
        "pool_size_fetched": pool_size_fetched,
        "evaluated_total": len(evaluation_pool),
        "passed_count": len(passed_works),
        "rejected_count": len(evaluation_pool) - len(passed_works),
        "sufficient": sufficient,
        "model": model or default_topic_filter_model(),
    }

    metadata = {
        **last_openalex_metadata,
        "n": n,
        "returned": len(top_works),
        "llm_filter": llm_filter_meta,
    }
    return top_works, evaluation_pool, metadata


def _write_markdown_filtered(path: str, works: list[dict], metadata: dict) -> None:
    """Write markdown with LLM filter metadata and relevance column."""
    window_months = metadata.get("citation_window_months", CITATION_WINDOW_MONTHS)
    llm_filter = metadata.get("llm_filter") or {}
    label = (
        metadata.get("topic_spec")
        or metadata.get("query")
        or metadata.get("topic_id")
        or "all"
    )
    lines = [
        f"# LLM-filtered most cited papers ({window_months} months): {label}",
        "",
        f"- **Source:** {metadata.get('source', 'OpenAlex')}",
        f"- **Algorithm:** {metadata.get('algorithm', 'n/a')} + LLM topic filter",
        f"- **Topic label (LLM):** {llm_filter.get('topic_label', 'n/a')}",
        f"- **Target N:** {llm_filter.get('target_n', metadata.get('n', 'n/a'))}",
        "- **LLM pool multiplier used:** "
        f"{llm_filter.get('llm_pool_multiplier_used', llm_filter.get('multiplier_used', 'n/a'))}",
        f"- **LLM pool size fetched:** {llm_filter.get('pool_size_fetched', 'n/a')}",
        "- **Citation algorithm initial multiplier:** "
        f"{llm_filter.get('citation_initial_multiplier', metadata.get('initial_multiplier', 'n/a'))}",
        "- **Citation algorithm initial pool size |S|:** "
        f"{metadata.get('initial_pool_size', metadata.get('initial_set_size', 'n/a'))}",
        f"- **Evaluated / passed / rejected:** "
        f"{llm_filter.get('evaluated_total', 'n/a')} / "
        f"{llm_filter.get('passed_count', 'n/a')} / "
        f"{llm_filter.get('rejected_count', 'n/a')}",
        f"- **Sufficient:** {llm_filter.get('sufficient', 'n/a')}",
        f"- **LLM model:** {llm_filter.get('model', 'n/a')}",
        "- **Publication date range:** "
        f"{metadata.get('from_publication_date')} to {metadata.get('to_publication_date')}",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Rank | 18m Citations | On Topic | Title | Journal | Published | DOI |",
        "| ---: | ---: | :---: | --- | --- | --- | --- |",
    ]
    for idx, work in enumerate(works, start=1):
        title = (work.get("title") or "").replace("|", "\\|")
        journal = (work.get("journal") or "").replace("|", "\\|")
        doi = work.get("doi") or ""
        doi_cell = f"[{doi}](https://doi.org/{doi})" if doi else ""
        on_topic = "yes" if work.get("is_about_the_topic") else "no"
        lines.append(
            f"| {idx} | {work.get('citations_within_window', 0)} | {on_topic} | "
            f"{title} | {journal} | {work.get('publication_date', '')} | {doi_cell} |"
        )

    lines.extend(["", "## Details", ""])
    for idx, work in enumerate(works, start=1):
        lines.extend(
            [
                f"### {idx}. {work.get('title', '')}",
                "",
                f"- **On topic (LLM):** {work.get('is_about_the_topic')}",
                f"- **Citations within {window_months} months:** {work.get('citations_within_window')}",
                f"- **Total citations (OpenAlex):** {work.get('cited_by_count')}",
                f"- **Authors:** {work.get('authors')}",
                f"- **Journal:** {work.get('journal')}",
                f"- **DOI:** {work.get('doi') or 'n/a'}",
                f"- **OpenAlex:** {work.get('openalex_url')}",
                "",
            ]
        )
        if work.get("abstract"):
            lines.extend([work["abstract"], ""])

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find top papers by 18m citations after LLM topic-relevance filtering (OpenAlex)."
        )
    )
    parser.add_argument(
        "--topic",
        help='Topic spec as "T{id},{display_name}" (exact match); topic-only, no text search',
    )
    parser.add_argument(
        "--query",
        "-q",
        default="",
        help="Text search query (ignored when --topic is set)",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        help="Start of publication date range, YYYY-MM-DD",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        help="End of publication date range, YYYY-MM-DD",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of top on-topic papers to return (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--initial-multiplier",
        type=float,
        default=DEFAULT_INITIAL_MULTIPLIER,
        help=f"First LLM candidate pool size multiplier (default: {DEFAULT_INITIAL_MULTIPLIER})",
    )
    parser.add_argument(
        "--max-multiplier",
        type=float,
        default=DEFAULT_MAX_MULTIPLIER,
        help=f"Maximum LLM pool multiplier before giving up (default: {DEFAULT_MAX_MULTIPLIER})",
    )
    parser.add_argument(
        "--citation-initial-multiplier",
        type=float,
        default=DEFAULT_CITATION_INITIAL_MULTIPLIER,
        help=(
            "Initial kN multiplier used inside the exact citation-ranking "
            f"algorithm (default: {DEFAULT_CITATION_INITIAL_MULTIPLIER})"
        ),
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=CITATION_WINDOW_MONTHS,
        help=f"Citation window in months (default: {CITATION_WINDOW_MONTHS})",
    )
    parser.add_argument(
        "--topic-id",
        help="OpenAlex topic ID (e.g. T10855) to narrow results",
    )
    parser.add_argument(
        "--topic-search",
        default="",
        help="Resolve topic ID from label when --topic-id is not set",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON and Markdown output",
    )
    parser.add_argument(
        "--include-reviews",
        action="store_true",
        help="Include reviews and other non-article types",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip citation-window count cache",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model (default: OPENROUTER_PRI_TOPIC_CHECK_MODEL from .env)",
    )
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Max concurrent LLM calls (default: {DEFAULT_MAX_CONCURRENT})",
    )
    args = parser.parse_args()

    topic_id: str | None = None
    topic_display_name: str | None = None
    query = args.query

    if args.topic:
        raw_topic_id, expected_name = parse_topic_spec(args.topic)
        topic_id = validate_topic(raw_topic_id, expected_name)
        topic_display_name = expected_name
        query = ""
        print(f"Validated topic {args.topic!r} -> {topic_id}")
    elif args.topic_id:
        topic_id = args.topic_id
    elif args.topic_search:
        topic_id = resolve_topic_id(args.topic_search)
        if topic_id:
            print(f"Resolved topic {args.topic_search!r} -> {topic_id}")

    topic_label = (
        topic_display_name
        or args.topic_search
        or args.topic
        or topic_id
        or query
        or "the research topic"
    )
    output_label = args.topic or query or topic_id or "all"
    output_dir = args.output_dir or default_output_dir(args.from_date, output_label)

    def llm_progress(completed: int, total: int) -> None:
        print(f"  LLM topic check: {completed}/{total}")

    print(
        f"Fetching and LLM-filtering: topic={topic_label!r}, N={args.limit}, "
        f"LLM pool multipliers {args.initial_multiplier}–{args.max_multiplier}, "
        f"citation kN multiplier {args.citation_initial_multiplier}..."
    )
    top_works, evaluation_pool, metadata = fetch_and_filter_top_n(
        query=query,
        n=args.limit,
        topic_label=topic_label,
        from_publication_date=args.from_date,
        to_publication_date=args.to_date,
        topic_id=topic_id,
        articles_only=not args.include_reviews,
        months=args.window_months,
        use_cache=not args.no_cache,
        initial_multiplier=args.initial_multiplier,
        max_multiplier=args.max_multiplier,
        citation_initial_multiplier=args.citation_initial_multiplier,
        model=args.model,
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        progress_callback=llm_progress,
    )

    if args.topic:
        metadata["topic_spec"] = args.topic
        metadata["topic_display_name"] = topic_display_name
    elif topic_id:
        metadata["topic_search"] = args.topic_search

    llm_filter = metadata.get("llm_filter") or {}
    if not llm_filter.get("sufficient"):
        print(
            f"WARNING: Only {llm_filter.get('passed_count', 0)} on-topic papers found "
            f"(target N={args.limit}). Returning all that passed."
        )

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "top_18m_llm_filtered.json")
    md_path = os.path.join(output_dir, "top_18m_llm_filtered.md")

    payload = {
        "metadata": metadata,
        "works": top_works,
        "evaluation_pool": evaluation_pool,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    _write_markdown_filtered(md_path, top_works, metadata)

    print(
        f"Evaluated {llm_filter.get('evaluated_total', 0)} papers; "
        f"{llm_filter.get('passed_count', 0)} on-topic; "
        f"LLM pool multiplier used={llm_filter.get('llm_pool_multiplier_used')}; "
        f"citation kN multiplier={llm_filter.get('citation_initial_multiplier')}"
    )
    print(f"Saved top {len(top_works)} LLM-filtered papers.")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print()
    for idx, work in enumerate(top_works, start=1):
        print(
            f"{idx}. [{work['citations_within_window']:4d} in {args.window_months}m] "
            f"{work['publication_date']} | {work['title'][:80]}"
        )


if __name__ == "__main__":
    main()
