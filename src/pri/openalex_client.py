"""OpenAlex API client for citation-ranked literature search."""

from __future__ import annotations

import calendar
import fcntl
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

load_dotenv()

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_PER_PAGE = 25
MAX_PER_PAGE = 200
CITATION_WINDOW_MONTHS = 18
CITATION_CACHE_TTL_DAYS = 7
GET_MAX_ATTEMPTS = 5
GET_BACKOFF_SECONDS = 2.0
GET_TIMEOUT_SECONDS = 60

_citation_cache: dict[str, Any] | None = None
_citation_cache_stats = {"hits": 0, "misses": 0}


def reset_citation_cache_stats() -> None:
    global _citation_cache_stats
    _citation_cache_stats = {"hits": 0, "misses": 0}


def get_citation_cache_stats() -> dict[str, int]:
    return dict(_citation_cache_stats)


def citation_cache_path() -> str:
    """Path to the on-disk citation-window count cache file."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, ".cache", "openalex_citation_window.json")


def _citation_cache_key(work_id: str, publication_date: str, months: int) -> str:
    work_id = work_id.removeprefix("https://openalex.org/")
    return f"{work_id}|{publication_date}|{months}"


def _citation_cache_lock_path() -> str:
    return f"{citation_cache_path()}.lock"


@contextmanager
def _citation_cache_file_lock() -> Iterator[None]:
    """Exclusive lock so parallel topic runs can safely update the shared cache."""
    lock_path = _citation_cache_lock_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _load_citation_cache_from_disk() -> dict[str, Any]:
    path = citation_cache_path()
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _ensure_citation_cache_loaded() -> dict[str, Any]:
    global _citation_cache
    if _citation_cache is not None:
        return _citation_cache
    _citation_cache = _load_citation_cache_from_disk()
    return _citation_cache


def _save_citation_cache(cache: dict[str, Any]) -> None:
    path = citation_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _get_cached_citation_count(key: str) -> tuple[int, str] | None:
    cache = _ensure_citation_cache_loaded()
    entry = cache.get(key)
    if not entry or not isinstance(entry, dict):
        return None
    cached_at_str = entry.get("cached_at")
    if cached_at_str:
        try:
            cached_at = datetime.fromisoformat(cached_at_str.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - cached_at > timedelta(days=CITATION_CACHE_TTL_DAYS):
                return None
        except ValueError:
            return None
    count = entry.get("count")
    window_end = entry.get("window_end")
    if count is None or not window_end:
        return None
    global _citation_cache_stats
    _citation_cache_stats["hits"] += 1
    return int(count), str(window_end)


def _set_cached_citation_count(key: str, count: int, window_end: str) -> None:
    global _citation_cache
    entry = {
        "count": count,
        "window_end": window_end,
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with _citation_cache_file_lock():
        cache = _load_citation_cache_from_disk()
        cache[key] = entry
        _save_citation_cache(cache)
        _citation_cache = cache


def _headers() -> dict[str, str]:
    api_key = (os.getenv("OPENALEX_API_KEY") or "").strip()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{OPENALEX_BASE}/{path.lstrip('/')}"
    last_error: BaseException | None = None
    for attempt in range(GET_MAX_ATTEMPTS):
        while True:
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=_headers(),
                    timeout=GET_TIMEOUT_SECONDS,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                break

            if response.status_code == 429:
                last_error = requests.HTTPError(
                    f"429 Client Error: Too Many Requests for url: {response.url}",
                    response=response,
                )
                if attempt >= GET_MAX_ATTEMPTS - 1:
                    raise last_error
                break

            try:
                if response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status is None or status < 500 or attempt >= GET_MAX_ATTEMPTS - 1:
                    raise
                break

            return response.json()

        if attempt >= GET_MAX_ATTEMPTS - 1:
            if last_error:
                raise last_error
            raise RuntimeError("OpenAlex _get exhausted retries without raising")

        delay = GET_BACKOFF_SECONDS * (2**attempt)
        print(
            f"  WARNING: OpenAlex API error ({last_error}); "
            f"retrying in {delay:.0f}s "
            f"(attempt {attempt + 1}/{GET_MAX_ATTEMPTS})",
            file=sys.stderr,
        )
        time.sleep(delay)

    raise RuntimeError("OpenAlex _get exhausted retries without raising") from last_error


def search_topics(query: str, per_page: int = 5) -> list[dict[str, Any]]:
    data = _get("topics", params={"search": query, "per_page": per_page})
    return data.get("results", [])


def resolve_topic_id(topic_search: str) -> str | None:
    """Return OpenAlex topic short ID (e.g. T10855) for the best topic match."""
    results = search_topics(topic_search, per_page=1)
    if not results:
        return None
    topic_url = results[0].get("id") or ""
    return topic_url.rsplit("/", 1)[-1]


def resolve_topic_id_by_display_name(display_name: str) -> str | None:
    """Return OpenAlex topic short ID for an exact display_name match."""
    if "," not in display_name:
        data = _get(
            "topics",
            params={"filter": f"display_name:{display_name}", "per_page": 5},
        )
        for topic in data.get("results", []):
            if topic.get("display_name") == display_name:
                topic_url = topic.get("id") or ""
                return topic_url.rsplit("/", 1)[-1]
    for topic in search_topics(display_name, per_page=25):
        if topic.get("display_name") == display_name:
            topic_url = topic.get("id") or ""
            return topic_url.rsplit("/", 1)[-1]
    return None


def parse_topic_spec(spec: str) -> tuple[str, str | None]:
    """Split ``T{id},{display_name}`` into topic ID and optional display name."""
    spec = spec.strip()
    if "," in spec:
        topic_id, display_name = spec.split(",", 1)
        return topic_id.strip(), display_name.strip() or None
    return spec, None


def validate_topic(topic_id: str, expected_name: str | None = None) -> str:
    """Validate topic ID via OpenAlex; optionally require exact display_name match."""
    topic_id = topic_id.removeprefix("https://openalex.org/").strip()
    if not topic_id:
        raise ValueError("Topic ID is required")

    data = _get(f"topics/{topic_id}")
    actual_name = (data.get("display_name") or "").strip()
    if not actual_name:
        raise ValueError(f"OpenAlex topic not found: {topic_id}")

    if expected_name is not None and actual_name != expected_name:
        raise ValueError(
            f"Topic display name mismatch for {topic_id}: "
            f"expected {expected_name!r}, got {actual_name!r}"
        )
    return topic_id


def _works_params_with_optional_search(query: str, **extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = dict(extra)
    if query.strip():
        params["search"] = query
    return params


def fetch_all_topics() -> dict[str, str]:
    """Return display_name -> short topic ID for all OpenAlex topics."""
    topics: dict[str, str] = {}
    cursor = "*"
    while cursor:
        data = _get(
            "topics",
            params={"per_page": MAX_PER_PAGE, "cursor": cursor},
        )
        for topic in data.get("results", []):
            name = (topic.get("display_name") or "").strip()
            topic_url = topic.get("id") or ""
            topic_id = topic_url.rsplit("/", 1)[-1]
            if name and topic_id:
                topics[name] = topic_id
        cursor = (data.get("meta") or {}).get("next_cursor") or ""
    return topics


def reconstruct_abstract(abstract_inverted_index: dict[str, list[int]] | None) -> str:
    if not abstract_inverted_index:
        return ""
    max_index = max(idx for indices in abstract_inverted_index.values() for idx in indices)
    words: list[str | None] = [None] * (max_index + 1)
    for word, indices in abstract_inverted_index.items():
        for idx in indices:
            words[idx] = word
    return " ".join(w for w in words if w)


def _parse_authorships(authorships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for authorship in authorships:
        author = authorship.get("author") or {}
        name = (author.get("display_name") or "").strip()
        if not name:
            continue

        raw_affiliations = [
            aff.strip()
            for aff in (authorship.get("raw_affiliation_strings") or [])
            if aff and aff.strip()
        ]
        institutions: list[dict[str, str]] = []
        for inst in authorship.get("institutions") or []:
            inst_name = (inst.get("display_name") or "").strip()
            if inst_name:
                institutions.append(
                    {
                        "name": inst_name,
                        "country_code": (inst.get("country_code") or "").strip(),
                        "openalex_id": (inst.get("id") or "").rsplit("/", 1)[-1],
                    }
                )

        if raw_affiliations:
            affiliations = raw_affiliations
        else:
            affiliations = [inst["name"] for inst in institutions]

        parsed.append(
            {
                "name": name,
                "affiliations": affiliations,
                "institutions": institutions,
                "is_corresponding": bool(authorship.get("is_corresponding")),
                "author_position": (authorship.get("author_position") or "").strip(),
            }
        )
    return parsed


def _format_authors(authorships: list[dict[str, Any]], max_authors: int = 5) -> str:
    names = [a["name"] for a in _parse_authorships(authorships) if a.get("name")]
    if len(names) > max_authors:
        return ", ".join(names[:max_authors]) + ", et al."
    return ", ".join(names)


def _format_journal(work: dict[str, Any]) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return (source.get("display_name") or "").strip()


def normalize_work(work: dict[str, Any]) -> dict[str, Any]:
    primary_topic = work.get("primary_topic") or {}
    raw_authorships = work.get("authorships") or []
    authorships = _parse_authorships(raw_authorships)
    return {
        "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1],
        "title": work.get("title") or "",
        "authors": _format_authors(raw_authorships),
        "authorships": authorships,
        "cited_by_count": work.get("cited_by_count") or 0,
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date") or "",
        "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
        "journal": _format_journal(work),
        "type": work.get("type") or "",
        "is_oa": ((work.get("open_access") or {}).get("is_oa")),
        "primary_topic": primary_topic.get("display_name") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "openalex_url": work.get("id") or "",
    }


def add_months_to_iso_date(iso_date: str, months: int) -> str:
    """Return ISO date string after adding calendar months."""
    d = date.fromisoformat(iso_date)
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day)).isoformat()


def citation_window_end(publication_date: str, months: int = CITATION_WINDOW_MONTHS) -> str:
    """Last day of the post-publication citation window (inclusive)."""
    return add_months_to_iso_date(publication_date, months)


def is_eligible_for_citation_window(
    publication_date: str,
    *,
    months: int = CITATION_WINDOW_MONTHS,
    as_of: date | None = None,
) -> bool:
    """True when the full citation window has elapsed as of ``as_of`` (default: today)."""
    if not publication_date:
        return False
    as_of = as_of or date.today()
    return date.fromisoformat(citation_window_end(publication_date, months)) <= as_of


def count_citations_in_publication_window(
    openalex_id: str,
    publication_date: str,
    *,
    months: int = CITATION_WINDOW_MONTHS,
    use_cache: bool = True,
) -> tuple[int, str]:
    """
    Count citations received within ``months`` after ``publication_date``.

    Uses OpenAlex ``cites`` filter with citing-work publication dates in
    ``[publication_date, publication_date + months]`` (inclusive).

    Returns ``(count, window_end_date)``.
    """
    work_id = openalex_id.removeprefix("https://openalex.org/")
    window_end = citation_window_end(publication_date, months)
    if use_cache:
        cache_key = _citation_cache_key(work_id, publication_date, months)
        cached = _get_cached_citation_count(cache_key)
        if cached is not None:
            return cached
    filter_str = (
        f"cites:{work_id},"
        f"from_publication_date:{publication_date},"
        f"to_publication_date:{window_end}"
    )
    data = _get(
        "works",
        params={
            "filter": filter_str,
            "per_page": 1,
            "select": "id",
        },
    )
    count = int((data.get("meta") or {}).get("count") or 0)
    if use_cache:
        global _citation_cache_stats
        _citation_cache_stats["misses"] += 1
        _set_cached_citation_count(
            _citation_cache_key(work_id, publication_date, months),
            count,
            window_end,
        )
    return count, window_end


def enrich_works_with_citation_window_counts(
    works: list[dict[str, Any]],
    *,
    months: int = CITATION_WINDOW_MONTHS,
    as_of: date | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Add ``citations_within_window`` and ``citation_window_end`` to each work."""
    enriched: list[dict[str, Any]] = []
    for work in works:
        pub_date = work.get("publication_date") or ""
        if not pub_date or not is_eligible_for_citation_window(
            pub_date, months=months, as_of=as_of
        ):
            continue
        count, window_end = count_citations_in_publication_window(
            work["openalex_id"],
            pub_date,
            months=months,
            use_cache=use_cache,
        )
        updated = dict(work)
        updated["citations_within_window"] = count
        updated["citation_window_months"] = months
        updated["citation_window_end"] = window_end
        enriched.append(updated)
        time.sleep(0.1)
    return enriched


def fetch_top_cited_works(
    query: str,
    year: int,
    *,
    topic_id: str | None = None,
    articles_only: bool = True,
    limit: int = 25,
    per_page: int = DEFAULT_PER_PAGE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Fetch works matching query and year, sorted by citation count descending.

    When articles_only is True, excludes reviews and keeps OpenAlex type ``article``.

    Returns normalized works and query metadata.
    """
    per_page = min(max(per_page, 1), MAX_PER_PAGE)
    filters = [f"publication_year:{year}"]
    if articles_only:
        filters.append("type:article")
    if topic_id:
        topic_id = topic_id.removeprefix("https://openalex.org/")
        filters.append(f"topics.id:{topic_id}")

    filter_str = ",".join(filters)
    collected: list[dict[str, Any]] = []
    cursor: str | None = "*"
    total_count = 0
    page = 0

    while cursor and len(collected) < limit:
        page += 1
        params = _works_params_with_optional_search(
            query,
            filter=filter_str,
            sort="cited_by_count:desc",
            per_page=min(per_page, limit - len(collected)),
            cursor=cursor,
        )
        data = _get("works", params=params)
        if page == 1:
            total_count = int((data.get("meta") or {}).get("count") or 0)

        results = data.get("results") or []
        if not results:
            break

        for work in results:
            collected.append(normalize_work(work))
            if len(collected) >= limit:
                break

        cursor = (data.get("meta") or {}).get("next_cursor")
        if cursor:
            time.sleep(0.1)

    metadata = {
        "query": query,
        "year": year,
        "topic_id": topic_id,
        "articles_only": articles_only,
        "filter": filter_str,
        "total_matching": total_count,
        "returned": len(collected),
        "sort": "cited_by_count:desc",
        "ranking_metric": "cited_by_count",
        "source": "OpenAlex",
        "api_base": OPENALEX_BASE,
    }
    return collected, metadata


def fetch_top_cited_works_by_date_range(
    query: str,
    from_publication_date: str,
    to_publication_date: str,
    *,
    topic_id: str | None = None,
    articles_only: bool = True,
    limit: int = 25,
    per_page: int = DEFAULT_PER_PAGE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Fetch works matching query and publication date range, sorted by citation count.

    Dates must be ISO 8601 (YYYY-MM-DD). When articles_only is True, keeps type ``article``.

    Returns normalized works and query metadata.
    """
    per_page = min(max(per_page, 1), MAX_PER_PAGE)
    filters = [
        f"from_publication_date:{from_publication_date}",
        f"to_publication_date:{to_publication_date}",
    ]
    if articles_only:
        filters.append("type:article")
    if topic_id:
        topic_id = topic_id.removeprefix("https://openalex.org/")
        filters.append(f"topics.id:{topic_id}")

    filter_str = ",".join(filters)
    collected: list[dict[str, Any]] = []
    cursor: str | None = "*"
    total_count = 0
    page = 0

    while cursor and len(collected) < limit:
        page += 1
        params = _works_params_with_optional_search(
            query,
            filter=filter_str,
            sort="cited_by_count:desc",
            per_page=min(per_page, limit - len(collected)),
            cursor=cursor,
        )
        data = _get("works", params=params)
        if page == 1:
            total_count = int((data.get("meta") or {}).get("count") or 0)

        results = data.get("results") or []
        if not results:
            break

        for work in results:
            collected.append(normalize_work(work))
            if len(collected) >= limit:
                break

        cursor = (data.get("meta") or {}).get("next_cursor")
        if cursor:
            time.sleep(0.1)

    metadata = {
        "query": query,
        "from_publication_date": from_publication_date,
        "to_publication_date": to_publication_date,
        "topic_id": topic_id,
        "articles_only": articles_only,
        "filter": filter_str,
        "total_matching": total_count,
        "returned": len(collected),
        "sort": "cited_by_count:desc",
        "ranking_metric": "cited_by_count",
        "source": "OpenAlex",
        "api_base": OPENALEX_BASE,
    }
    return collected, metadata


def _build_date_range_filter_str(
    from_publication_date: str,
    to_publication_date: str,
    *,
    topic_id: str | None = None,
    articles_only: bool = True,
    min_citations: int | None = None,
) -> str:
    filters = [
        f"from_publication_date:{from_publication_date}",
        f"to_publication_date:{to_publication_date}",
    ]
    if articles_only:
        filters.append("type:article")
    if topic_id:
        topic_id = topic_id.removeprefix("https://openalex.org/")
        filters.append(f"topics.id:{topic_id}")
    if min_citations is not None and min_citations > 0:
        # OpenAlex supports > but not >=; cited_by_count:>N-1 matches >= N.
        filters.append(f"cited_by_count:>{min_citations - 1}")
    return ",".join(filters)


def fetch_works_by_min_total_citations(
    query: str,
    from_publication_date: str,
    to_publication_date: str,
    min_citations: int,
    *,
    topic_id: str | None = None,
    articles_only: bool = True,
    per_page: int = DEFAULT_PER_PAGE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Fetch all works matching query, date range, and ``cited_by_count >= min_citations``.

    When ``min_citations`` is 0, no minimum-citation filter is applied.
    """
    per_page = min(max(per_page, 1), MAX_PER_PAGE)
    filter_str = _build_date_range_filter_str(
        from_publication_date,
        to_publication_date,
        topic_id=topic_id,
        articles_only=articles_only,
        min_citations=min_citations if min_citations > 0 else None,
    )
    collected: list[dict[str, Any]] = []
    cursor: str | None = "*"
    total_count = 0
    page = 0

    while cursor:
        page += 1
        params = _works_params_with_optional_search(
            query,
            filter=filter_str,
            sort="cited_by_count:desc",
            per_page=per_page,
            cursor=cursor,
        )
        data = _get("works", params=params)
        if page == 1:
            total_count = int((data.get("meta") or {}).get("count") or 0)

        results = data.get("results") or []
        if not results:
            break

        for work in results:
            collected.append(normalize_work(work))

        cursor = (data.get("meta") or {}).get("next_cursor")
        if cursor:
            time.sleep(0.1)

    metadata = {
        "query": query,
        "from_publication_date": from_publication_date,
        "to_publication_date": to_publication_date,
        "topic_id": topic_id,
        "articles_only": articles_only,
        "min_citations": min_citations,
        "filter": filter_str,
        "total_matching": total_count,
        "returned": len(collected),
        "sort": "cited_by_count:desc",
        "ranking_metric": "cited_by_count",
        "source": "OpenAlex",
        "api_base": OPENALEX_BASE,
    }
    return collected, metadata


def _score_work_for_citation_window(
    work: dict[str, Any],
    *,
    months: int = CITATION_WINDOW_MONTHS,
    as_of: date | None = None,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    """Return work enriched with window citation count, or None if ineligible."""
    pub_date = work.get("publication_date") or ""
    if not pub_date or not is_eligible_for_citation_window(
        pub_date, months=months, as_of=as_of
    ):
        return None
    count, window_end = count_citations_in_publication_window(
        work["openalex_id"],
        pub_date,
        months=months,
        use_cache=use_cache,
    )
    updated = dict(work)
    updated["citations_within_window"] = count
    updated["citation_window_months"] = months
    updated["citation_window_end"] = window_end
    return updated


def find_top_n_by_citation_window(
    query: str,
    n: int,
    from_publication_date: str,
    to_publication_date: str,
    *,
    initial_multiplier: float = 2.0,
    topic_id: str | None = None,
    articles_only: bool = True,
    months: int = CITATION_WINDOW_MONTHS,
    as_of: date | None = None,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Exact top-N retrieval by citations within ``months`` after publication.

    Implements the threshold-expansion algorithm from ``docs/pri/the_algorithm.md``.
    """
    reset_citation_cache_stats()
    as_of = as_of or date.today()
    initial_pool_size = max(n, math.ceil(initial_multiplier * n))

    # Step 1: initial candidate set S (top kN by total citations)
    initial_set, initial_meta = fetch_top_cited_works_by_date_range(
        query,
        from_publication_date,
        to_publication_date,
        topic_id=topic_id,
        articles_only=articles_only,
        limit=initial_pool_size,
    )

    # Step 2: compute window citations for eligible papers in S
    scored: dict[str, dict[str, Any]] = {}
    scored_in_step2 = 0
    for work in initial_set:
        enriched = _score_work_for_citation_window(
            work, months=months, as_of=as_of, use_cache=use_cache
        )
        if enriched is None:
            continue
        scored[work["openalex_id"]] = enriched
        scored_in_step2 += 1
        time.sleep(0.1)

    # Step 3: threshold m from N-th highest window count in S
    sorted_from_s = sorted(
        scored.values(),
        key=lambda w: (w["citations_within_window"], w["cited_by_count"]),
        reverse=True,
    )
    if len(sorted_from_s) >= n:
        threshold_m = sorted_from_s[n - 1]["citations_within_window"]
    else:
        threshold_m = 0

    # Step 4: expand candidate pool T (total citations >= m)
    expanded_set, expanded_meta = fetch_works_by_min_total_citations(
        query,
        from_publication_date,
        to_publication_date,
        threshold_m,
        topic_id=topic_id,
        articles_only=articles_only,
    )

    # Step 5: compute window citations for T \\ S
    scored_in_step5 = 0
    for work in expanded_set:
        if work["openalex_id"] in scored:
            continue
        enriched = _score_work_for_citation_window(
            work, months=months, as_of=as_of, use_cache=use_cache
        )
        if enriched is None:
            continue
        scored[work["openalex_id"]] = enriched
        scored_in_step5 += 1
        time.sleep(0.1)

    # Step 6: final leaderboard from all scored papers in T
    expanded_ids = {work["openalex_id"] for work in expanded_set}
    final_candidates = [
        scored[work_id]
        for work_id in expanded_ids
        if work_id in scored
    ]
    final_candidates.sort(
        key=lambda w: (w["citations_within_window"], w["cited_by_count"]),
        reverse=True,
    )
    top_works = final_candidates[:n]

    metadata: dict[str, Any] = {
        "query": query,
        "from_publication_date": from_publication_date,
        "to_publication_date": to_publication_date,
        "topic_id": topic_id,
        "articles_only": articles_only,
        "n": n,
        "initial_multiplier": initial_multiplier,
        "initial_pool_size": initial_pool_size,
        "threshold_m": threshold_m,
        "initial_set_size": len(initial_set),
        "expanded_set_size": len(expanded_set),
        "scored_in_step2": scored_in_step2,
        "scored_in_step5": scored_in_step5,
        "scored_total": len(final_candidates),
        "total_matching": initial_meta.get("total_matching", 0),
        "expanded_total_matching": expanded_meta.get("total_matching", 0),
        "initial_filter": initial_meta.get("filter"),
        "expanded_filter": expanded_meta.get("filter"),
        "citation_window_months": months,
        "ranking_metric": f"citations_within_{months}_months",
        "algorithm": "threshold_expansion",
        "source": "OpenAlex",
        "api_base": OPENALEX_BASE,
        "returned": len(top_works),
        "citation_cache": get_citation_cache_stats(),
        "citation_cache_path": citation_cache_path() if use_cache else None,
    }
    return top_works, metadata
