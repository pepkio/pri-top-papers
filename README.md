# Top Papers by 18-Month Citations

Find the most influential recent papers in an OpenAlex research topic, ranked by **exact** citations within a fixed post-publication window, with LLM topic-relevance filtering.

Traditional citation leaderboards favor older papers because total citations accumulate over many years. This tool compares papers at the same stage of their lifecycle by counting citations from **citing papers published within 18 months** after each target paper's publication date (the default window). Rankings are scoped to publication cohorts—for example, papers published in 2024 form the **Class of 2026** cohort once every paper in that year has completed its full 18-month window.

## Transparent, reproducible rankings

Every published leaderboard can be audited from the tool's output:

- **Exact citation ranking** — The threshold-expansion algorithm in [`openalex_client.py`](src/pri/openalex_client.py) is mathematically exact: it never misses a true top-N paper by 18-month citations. See [`docs/the_algorithm.md`](docs/the_algorithm.md) for the full write-up and correctness proof.
- **Fixed comparison window** — All papers in a cohort are evaluated with the same post-publication window (default: 18 months). Only papers that have completed the full window are eligible.
- **Full audit trail** — Output JSON includes:
  - `works` — the final ranked list returned to the user
  - `evaluation_pool` — every candidate paper sent to the LLM classifier, with `is_about_the_topic` and `is_research_article` for each
  - `metadata` — OpenAlex query parameters, multiplier ladder used, pool sizes, LLM model name, and pass/reject counts
- **Open classification rules** — The topic-relevance prompt (`TOPIC_RELEVANCE_PROMPT_TEMPLATE` in source) defines exactly how papers are judged on-topic or off-topic, and whether each paper counts as a research article.

You can verify every ranking decision: which papers were considered, how they were classified, and which citation counts determined their rank.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
cp .env.example .env   # fill in API keys (see Configuration)
uv run python src/pri/find_top_by_18m_citations_llm_filtered.py \
  --topic "T11287,Cancer Genomics and Diagnostics" \
  --from-date 2024-01-01 --to-date 2024-12-31 -n 50
```

**Output:** `results/<year>/<topic_slug>/top_18m_llm_filtered.json` and `top_18m_llm_filtered.md`

Example with a free-text query instead of an OpenAlex topic:

```bash
uv run python src/pri/find_top_by_18m_citations_llm_filtered.py \
  --query "CRISPR lung cancer" \
  --from-date 2024-01-01 --to-date 2024-12-31 -n 20
```

## How it works

The tool runs two steps:

### 1. Citation ranking (OpenAlex)

[`openalex_client.py`](src/pri/openalex_client.py) retrieves the exact top papers by 18-month citation count using a threshold-expansion algorithm. Instead of computing 18-month citations for every paper in a field, it:

1. Starts with a small set of highly cited candidates
2. Computes exact 18-month citation counts for that set
3. Uses the resulting threshold to eliminate papers that cannot possibly rank in the top N
4. Expands only as needed until the exact top N are proven

The `--citation-initial-multiplier` flag controls the initial candidate set size (the *kN* multiplier in [`docs/the_algorithm.md`](docs/the_algorithm.md)). Default: `1.5`.

### 2. Topic relevance filter (LLM)

An LLM classifies each candidate paper as on-topic or off-topic for the target research area, and whether it is a research article (vs. review, editorial, etc.). The tool returns the top *N* **on-topic research articles**, still ranked by 18-month citations.

Two separate multipliers control pool sizing:

- `--initial-multiplier` / `--max-multiplier` — control the LLM evaluation pool. For `-n 50` and the default initial multiplier `1.8`, the tool fetches the exact top 90 papers by 18-month citations, then LLM-filters them for topic relevance. If fewer than 50 pass, it escalates by adding 25 papers per step up to the max multiplier (250 for n=50).
- `--citation-initial-multiplier` — controls the internal threshold-expansion algorithm when computing exact 18-month citation ranks for whatever LLM pool size was requested.

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Required | Purpose |
|----------|----------|---------|
| `PRI_TOPIC_CHECK_CONCEPT_EXTRACT_MODEL_PROVIDER` | No | LLM provider: `openrouter` (default), `ofox`, or `n1n` |
| `OPENROUTER_PRI_TOPIC_CHECK_MODEL` | Yes* | LLM model for topic relevance checks (OpenRouter slug, e.g. `openai/gpt-4o-mini`) |
| `OPENROUTER_API_KEY` | Yes* | LLM API access via OpenRouter |
| `OPENALEX_API_KEY` | No | Improves OpenAlex rate limits |

\* When using OpenRouter (default). For Ofox or N1N, set the corresponding `OFOX_*` or `N1N_*` model and API key variables instead.

Optional:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_BASE_URL` | Override LLM API base URL (default: `https://openrouter.ai/api/v1`) |
| `OPENROUTER_PROVIDER_ORDER` | Comma-separated OpenRouter upstream providers to prefer |
| `OPENROUTER_HTTP_REFERER` | Attribution header for OpenRouter |
| `OPENROUTER_APP_TITLE` | Attribution header for OpenRouter |
| `OFOX_BASE_URL`, `OFOX_PROVIDER_ROUTING` | Ofox provider settings |
| `N1N_BASE_URL` | N1N provider base URL |
| `LLM_CACHE_DISABLED` | Set to `1` to disable LLM response caching |
| `LLM_CACHE_TTL_SECONDS` | LLM cache TTL in seconds (default: 604800 = 7 days) |

LLM responses are cached under `.cache/llm/` (7-day TTL by default) to avoid redundant API calls on re-runs. OpenAlex citation-window counts are cached in `.cache/openalex_citation_window.json` (7-day TTL). Use `--no-cache` to skip the citation cache, or `--no-llm-cache` to skip the LLM cache.

## CLI reference

| Flag | Description |
|------|-------------|
| `--topic` | OpenAlex topic spec as `"T{id},{display_name}"` (exact match; no text search) |
| `--query`, `-q` | Free-text search query (ignored when `--topic` is set) |
| `--from-date` | Start of publication date range (`YYYY-MM-DD`) **(required)** |
| `--to-date` | End of publication date range (`YYYY-MM-DD`) **(required)** |
| `--limit`, `-n` | Number of top on-topic papers to return (default: 20) |
| `--initial-multiplier` | First LLM candidate pool size multiplier (default: 1.8) |
| `--max-multiplier` | Maximum LLM pool multiplier before giving up (default: 5.0) |
| `--citation-initial-multiplier` | Initial *kN* multiplier for the exact citation-ranking algorithm (default: 1.5) |
| `--window-months` | Citation window in months (default: 18) |
| `--topic-id` | OpenAlex topic ID (e.g. `T10855`) to narrow results |
| `--topic-search` | Resolve topic ID from label when `--topic-id` is not set |
| `--output-dir` | Directory for JSON and Markdown output (default: `results/<year>/<topic_slug>/`) |
| `--include-reviews` | Include reviews and other non-article types |
| `--no-cache` | Skip citation-window count cache |
| `--no-llm-cache` | Disable local LLM response cache |
| `--strict` | Fail immediately if topic classification cannot be completed for a paper |
| `--model` | Override LLM model (default: from provider env vars) |
| `--base-url` | Override LLM API base URL |
| `--max-concurrent` | Max concurrent LLM calls (default: 5) |

## Output schema

### `top_18m_llm_filtered.json`

```json
{
  "metadata": { "...": "OpenAlex query info, algorithm params, llm_filter stats" },
  "works": [ "... ranked on-topic papers ..." ],
  "evaluation_pool": [ "... all candidates with is_about_the_topic ..." ]
}
```

Each work entry includes fields such as `title`, `abstract`, `authors`, `journal`, `publication_date`, `doi`, `openalex_id`, `openalex_url`, `citations_within_window`, `citation_window_end`, and (in `evaluation_pool`) `is_about_the_topic` and `is_research_article`.

The `metadata.llm_filter` block records the multiplier ladder, pool size fetched, evaluated/passed/rejected counts, LLM model used, and whether the target N was reached.

### `top_18m_llm_filtered.md`

Human-readable summary table with rank, 18-month citation count, on-topic flag, title, journal, publication date, and DOI, plus per-paper detail sections.

This methodology and the accompanying dataset were developed by [PRI](https://pri.pepkio.com), an open-science initiative supported by [Pepkio](https://pepkio.com), to provide transparent, reproducible, and accessible bibliometric datasets.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for non-commercial use (personal, educational, research, and other non-commercial purposes). Commercial use requires a separate license from the copyright holder.
