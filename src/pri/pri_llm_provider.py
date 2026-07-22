"""Resolve LLM provider config for PRI topic relevance checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from utils.llm_processor import LLMProcessor, OPENROUTER_DEFAULT_BASE_URL

PriLlmTask = Literal["topic_check"]

_TASK_PROVIDER_SELECTOR = {
    "topic_check": "PRI_TOPIC_CHECK_CONCEPT_EXTRACT_MODEL_PROVIDER",
}

OPENROUTER_BASE_URL = (
    (os.getenv("OPENROUTER_BASE_URL") or "").strip() or OPENROUTER_DEFAULT_BASE_URL
)
OFOX_DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
N1N_DEFAULT_BASE_URL = "https://api.n1n.ai/v1"

_PROVIDER_ALIASES = {
    "openrouter": "openrouter",
    "ofox": "ofox",
    "ofoxai": "ofox",
    "n1n": "n1n",
}


@dataclass(frozen=True)
class PriLlmConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    use_openrouter_routing: bool
    provider_routing: str | None


def _resolve_ofox_provider_routing() -> str | None:
    """Ofox gateway routing strategy; default ``cost`` (cheapest upstream)."""
    raw = (os.getenv("OFOX_PROVIDER_ROUTING") or "cost").strip()
    return raw or None


def _strip_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        return cleaned[1:-1].strip()
    return cleaned


def _resolve_provider_name(selector_var: str) -> str:
    raw = (os.getenv(selector_var) or "openrouter").strip()
    provider = _PROVIDER_ALIASES.get(raw.lower())
    if not provider:
        allowed = ", ".join(sorted(set(_PROVIDER_ALIASES.values())))
        raise ValueError(
            f"Unknown {selector_var}={raw!r}. Expected one of: {allowed}."
        )
    return provider


def _model_env_var(provider: str) -> str:
    return {
        "openrouter": "OPENROUTER_PRI_TOPIC_CHECK_MODEL",
        "ofox": "OFOX_PRI_TOPIC_CHECK_MODEL",
        "n1n": "N1N_PRI_TOPIC_CHECK_MODEL",
    }[provider]


def _provider_env(provider: str) -> tuple[str, str, str]:
    model_var = _model_env_var(provider)
    if provider == "openrouter":
        return model_var, "OPENROUTER_API_KEY", OPENROUTER_BASE_URL
    if provider == "ofox":
        base_url = (os.getenv("OFOX_BASE_URL") or "").strip() or OFOX_DEFAULT_BASE_URL
        return model_var, "OFOX_API_KEY", base_url
    if provider == "n1n":
        base_url = (os.getenv("N1N_BASE_URL") or "").strip() or N1N_DEFAULT_BASE_URL
        return model_var, "N1N_API_KEY", base_url
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_pri_llm_config(
    task: PriLlmTask,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> PriLlmConfig:
    """Resolve model, API key, and base URL for a PRI LLM task."""
    selector_var = _TASK_PROVIDER_SELECTOR[task]
    provider = _resolve_provider_name(selector_var)
    model_var, api_key_var, default_base_url = _provider_env(provider)

    resolved_model = (model or "").strip() or (os.getenv(model_var) or "").strip()
    if not resolved_model:
        raise ValueError(
            f"{model_var} is required for {task.replace('_', ' ')} "
            f"with provider {provider!r}. Set it in .env."
        )

    resolved_api_key = _strip_quotes((os.getenv(api_key_var) or "").strip())
    if not resolved_api_key:
        raise ValueError(
            f"{api_key_var} is required for provider {provider!r}. Set it in .env."
        )

    resolved_base_url = (base_url or "").strip() or default_base_url
    provider_routing = _resolve_ofox_provider_routing() if provider == "ofox" else None
    return PriLlmConfig(
        provider=provider,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        use_openrouter_routing=provider == "openrouter",
        provider_routing=provider_routing,
    )


def build_llm_processor(config: PriLlmConfig, **kwargs: Any) -> LLMProcessor:
    """Build an LLMProcessor from a resolved PRI provider config."""
    provider_order = None if config.use_openrouter_routing else ""
    return LLMProcessor(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        provider_order=provider_order,
        provider_routing=config.provider_routing,
        **kwargs,
    )
