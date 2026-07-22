"""
LLM-based record processing utility.

This module provides functions to process lists of records (dictionaries) using LLMs
to extract, parse, or enrich information. Each record is processed individually,
and the LLM response (parsed as JSON) is merged into the original record.
"""

import json
import os
import asyncio
import random
from typing import List, Dict, Any, Optional, Callable, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import cache utilities
from .cache_utils import async_cache, llm_async_cache, llm_cache_ttl_seconds

try:
    import openai
    from openai import AsyncOpenAI
    from openai._exceptions import (
        APIConnectionError,
        APIResponseValidationError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
    )
    import httpx
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    httpx = None
    # Empty tuples make ``isinstance(exc, NAME)`` return False when the openai
    # package is unavailable, so ``_is_retryable_error`` stays safe.
    APIConnectionError = APIResponseValidationError = APIStatusError = ()  # type: ignore[assignment]
    APITimeoutError = AuthenticationError = BadRequestError = ()  # type: ignore[assignment]
    InternalServerError = NotFoundError = PermissionDeniedError = ()  # type: ignore[assignment]
    RateLimitError = ()  # type: ignore[assignment]

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"


class LLMRetryExhausted(RuntimeError):
    """Raised when ``_call_llm`` exhausts retries on transient errors.

    Wraps the most recent transient failure (``last_error``) so callers can
    decide whether to degrade gracefully or surface it. Non-retryable errors
    (auth, bad request, ...) propagate immediately and are NOT wrapped here.
    """

    def __init__(self, last_error: BaseException, attempts: int):
        super().__init__(
            f"LLM call failed after {attempts} attempt(s); last error: {last_error!r}"
        )
        self.last_error = last_error
        self.attempts = attempts


def default_openrouter_model() -> str:
    """Model id when OPENROUTER_MODEL is unset or empty (OpenRouter slug, e.g. openai/gpt-4o-mini)."""
    return (os.getenv("OPENROUTER_MODEL") or "").strip() or OPENROUTER_DEFAULT_MODEL


def llm_call_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Stable cache key for ``_call_llm`` based on model settings and prompt text."""
    import hashlib

    processor = args[0]
    prompt = args[1] if len(args) > 1 else ""
    system_prompt = args[2] if len(args) > 2 else kwargs.get("system_prompt")
    use_json_format = kwargs.get("use_json_format")
    if use_json_format is None:
        use_json_format = not getattr(processor, "model", "").startswith("nvidia/")

    payload = {
        "func": func.__name__,
        "model": getattr(processor, "model", ""),
        "base_url": str(getattr(getattr(processor, "client", None), "base_url", "")),
        "temperature": getattr(processor, "temperature", None),
        "max_tokens": getattr(processor, "max_tokens", None),
        "reasoning_effort": getattr(processor, "reasoning_effort", None),
        "provider_order": getattr(processor, "provider_order", None),
        "provider_routing": getattr(processor, "provider_routing", None),
        "prompt": prompt,
        "system_prompt": system_prompt,
        "use_json_format": use_json_format,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return digest


class LLMProcessor:
    """
    A processor that uses LLMs to enrich records with extracted information.
    
    Processes records one by one, sends them to an LLM, parses the JSON response,
    and merges the extracted fields back into each record.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: int = 60,
        reasoning_effort: Optional[str] = None,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        provider_order: Optional[str] = None,
        provider_routing: Optional[str] = None,
    ):
        """
        Initialize the LLM processor.

        Args:
            model: OpenRouter model slug (e.g. openai/gpt-4o-mini). If None/empty, uses
                OPENROUTER_MODEL or OPENROUTER_DEFAULT_MODEL.
            api_key: If None, uses OPENROUTER_API_KEY, then OPENAI_API_KEY.
            base_url: OpenAI-compatible API base. If None/empty, uses OPENROUTER_BASE_URL
                or OpenRouter (https://openrouter.ai/api/v1).
            temperature: Sampling temperature (default: 0.0 for deterministic output)
            max_tokens: Maximum tokens in response (default: 2000)
            timeout: Request timeout in seconds (default: 60)
            reasoning_effort: Optional reasoning effort for reasoning models.
            max_retries: Max attempts (incl. first) for transient HTTP/JSON errors
                such as a truncated response body. Default 4.
            retry_backoff: Base seconds for exponential backoff between retries
                (``retry_backoff * 2**attempt`` plus up to 1s jitter). 0 disables
                backoff/jitter (useful for tests). Default 2.0.
            provider_order: Optional comma-separated list of OpenRouter upstream
                providers to prefer (e.g. "Novita"). If None/empty, reads
                OPENROUTER_PROVIDER_ORDER; if that is also unset, no provider
                routing is applied. When set, sends ``provider.order`` with
                ``allow_fallbacks=true`` so a known-good provider is tried first
                while still falling back (the existing retry catches any fallback
                that misbehaves, e.g. truncates a large response at ~8KB).
            provider_routing: Optional OfoxAI gateway routing strategy (e.g.
                ``"cost"`` for lowest-cost upstream). When set, sends
                ``provider.routing`` in ``extra_body``.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        resolved_model = (model or "").strip() or default_openrouter_model()
        resolved_base = (base_url or "").strip() or (
            (os.getenv("OPENROUTER_BASE_URL") or "").strip() or OPENROUTER_DEFAULT_BASE_URL
        )
        api_key = (api_key or "").strip() or (
            (os.getenv("OPENROUTER_API_KEY") or "").strip()
            or (os.getenv("OPENAI_API_KEY") or "").strip()
        )
        if not api_key:
            raise ValueError(
                "API key is required. Set OPENROUTER_API_KEY (default), OPENAI_API_KEY, "
                "or pass api_key=..."
            )

        self.model = resolved_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

        # Preferred upstream providers for OpenRouter routing. Empty list => no
        # ``provider`` field is sent (OpenRouter picks freely). An explicit
        # provider_order (even "") takes precedence over the env var so callers
        # can opt out of routing regardless of the environment.
        if provider_order is None:
            order_str = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
        else:
            order_str = provider_order.strip()
        self.provider_order = [p.strip() for p in order_str.split(",") if p.strip()]
        self.provider_routing = (provider_routing or "").strip() or None

        # Initialize client (OpenRouter: optional attribution headers)
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "base_url": resolved_base,
        }
        referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
        app_title = (os.getenv("OPENROUTER_APP_TITLE") or "").strip()
        if referer or app_title:
            headers: Dict[str, str] = {}
            if referer:
                headers["HTTP-Referer"] = referer
            if app_title:
                headers["X-OpenRouter-Title"] = app_title
            client_kwargs["default_headers"] = headers
        
        self.client = AsyncOpenAI(**client_kwargs)
    
    @llm_async_cache(
        prefix="llm_call",
        ttl=llm_cache_ttl_seconds(),
        key_func=llm_call_cache_key,
    )
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        use_json_format: Optional[bool] = None,
    ) -> str:
        """
        Call the LLM with a prompt and return the response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response text
        """
        request_params = self._build_request_params(
            prompt, system_prompt, use_json_format=use_json_format
        )

        # Retry transient failures (truncated/malformed HTTP body, timeouts,
        # connection blips, provider 5xx/429). Non-retryable client errors
        # (auth, bad request, ...) propagate immediately. After exhausting
        # retries on a transient error, raise LLMRetryExhausted so callers can
        # degrade gracefully instead of crashing.
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(**request_params)
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = getattr(choice, "finish_reason", None)
                if content is None or not str(content).strip():
                    last_error = ValueError("LLM returned empty content")
                    if attempt == self.max_retries - 1:
                        break
                    delay = self.retry_backoff * (2 ** attempt)
                    if delay > 0:
                        delay += random.uniform(0.0, 1.0)
                    print(
                        f"  [llm] empty response on attempt "
                        f"{attempt + 1}/{self.max_retries}; retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                # Reasoning models can burn max_tokens on reasoning_tokens and cut
                # the visible JSON mid-string (finish_reason=length). Retry rather
                # than returning truncated content that fails JSON parse.
                if finish_reason == "length":
                    last_error = ValueError(
                        f"LLM response truncated (finish_reason=length, "
                        f"max_tokens={self.max_tokens}): {(content or '')[:120]!r}"
                    )
                    if attempt == self.max_retries - 1:
                        break
                    delay = self.retry_backoff * (2 ** attempt)
                    if delay > 0:
                        delay += random.uniform(0.0, 1.0)
                    print(
                        f"  [llm] truncated response on attempt "
                        f"{attempt + 1}/{self.max_retries}; retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                return content
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_error(exc):
                    raise
                if attempt == self.max_retries - 1:
                    break
                delay = self.retry_backoff * (2 ** attempt)
                if delay > 0:
                    delay += random.uniform(0.0, 1.0)
                print(
                    f"  [llm] transient {type(exc).__name__} on attempt "
                    f"{attempt + 1}/{self.max_retries}; retrying in {delay:.1f}s: {exc}"
                )
                await asyncio.sleep(delay)
        raise LLMRetryExhausted(last_error, self.max_retries)  # type: ignore[arg-type]

    def _build_request_params(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        use_json_format: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Build the request kwargs sent to chat.completions.create.

        Exposed so diagnostic scripts can replay an identical request outside the
        SDK (e.g. via httpx) to inspect the raw HTTP response body.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Some models (e.g. DeepInfra's Nemotron) don't support response_format.
        if use_json_format is None:
            use_json_format = not self.model.startswith("nvidia/")

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if use_json_format:
            request_params["response_format"] = {"type": "json_object"}

        # extra_body is merged into the top-level request JSON by the SDK.
        # Reasoning effort + provider routing (OpenRouter order / Ofox cost) live here.
        extra_body: Dict[str, Any] = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        provider_extra: Dict[str, Any] = {}
        if self.provider_order:
            # Prefer the listed providers but keep allow_fallbacks=true so the
            # request still succeeds if the preferred provider is unavailable; the
            # retry-with-backoff above catches any fallback that returns a truncated
            # body. See docs/pri for the ~8KB truncation investigation.
            provider_extra["order"] = list(self.provider_order)
            provider_extra["allow_fallbacks"] = True
        if self.provider_routing:
            # OfoxAI extension: route to the lowest-cost upstream node.
            provider_extra["routing"] = self.provider_routing
        if provider_extra:
            extra_body["provider"] = provider_extra
        if extra_body:
            request_params["extra_body"] = extra_body
        return request_params

    def _is_retryable_error(self, exc: BaseException) -> bool:
        """Return True for transient errors worth retrying.

        Covers the truncated-response-body failure we hit in practice
        (``json.JSONDecodeError`` raised while the SDK parses the HTTP body),
        plus network/timeout and provider-side 5xx / 429 status errors.
        """
        # Non-retryable 4xx client errors: retrying won't change the outcome.
        if isinstance(
            exc,
            (BadRequestError, AuthenticationError, NotFoundError, PermissionDeniedError),
        ):
            return False
        # Malformed/truncated HTTP response body — the failure in the bug report.
        if isinstance(exc, json.JSONDecodeError):
            return True
        # Network / transport layer.
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return True
        if httpx is not None and isinstance(exc, httpx.TransportError):
            return True
        # Provider-side transient status codes.
        if isinstance(exc, (InternalServerError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if status is None:
                return False
            try:
                status_int = int(status)
            except (TypeError, ValueError):
                return False
            return status_int == 429 or 500 <= status_int < 600
        # SDK could not coerce the response into the expected shape (e.g. a
        # body that parsed as JSON but was structurally incomplete).
        if isinstance(exc, APIResponseValidationError):
            return True
        return False
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks if present.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary
        """
        if response_text is None or not str(response_text).strip():
            raise ValueError("Cannot parse empty LLM response")

        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```"):
            # Extract content between code blocks
            lines = text.split("\n")
            if lines[0].startswith("```"):
                # Remove first and last lines if they are code block markers
                if lines[-1].strip() == "```" or lines[-1].strip().startswith("```"):
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON. Response: {response_text[:500]}"
            ) from e
    
    async def process_record(
        self,
        record: Dict[str, Any],
        prompt_template: str,
        system_prompt: Optional[str] = None,
        output_field_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single record with the LLM.
        
        Args:
            record: Dictionary representing a single record
            prompt_template: Prompt template string. Use {record} or {record_json} to insert the record.
                           Example: "Extract key information from this record: {record_json}"
            system_prompt: Optional system prompt. If None, uses a default JSON-focused prompt.
            output_field_names: Optional list of expected field names in the JSON output.
                               Used to guide the LLM if provided.
        
        Returns:
            Original record dictionary with added fields from LLM response
        """
        # Format the prompt with the record
        record_json = json.dumps(record, indent=2, ensure_ascii=False)
        prompt = prompt_template.format(record=record, record_json=record_json)
        
        # Default system prompt if not provided
        if system_prompt is None:
            # Check if model supports JSON response format
            use_json_format = not self.model.startswith("nvidia/")

            if use_json_format:
                system_prompt = (
                    "You are a helpful assistant that extracts and processes information. "
                    "You MUST respond with a valid JSON object only. "
                    "Do not include any introductory text, explanations, or markdown formatting. "
                    "Just return the JSON object itself."
                )
            else:
                system_prompt = (
                    "You are a helpful assistant that extracts and processes information. "
                    "You MUST respond with a valid JSON object only. "
                    "Format your response as a valid JSON object. "
                    "Do not include any introductory text, explanations, or markdown formatting. "
                    "Just return the JSON object itself."
                )
            if output_field_names:
                system_prompt += (
                    f" The JSON object should include these fields: "
                    f"{', '.join(output_field_names)}."
                )
        
        # Call LLM
        response_text = await self._call_llm(prompt, system_prompt)
        
        # Parse JSON response
        extracted_fields = self._parse_json_response(response_text)
        
        # Merge extracted fields into the original record
        enriched_record = record.copy()
        enriched_record.update(extracted_fields)
        
        return enriched_record

    def _record_label(self, record: Dict[str, Any]) -> str:
        """Short identifier for log messages when a record fails."""
        for key in ("pmid", "openalex_id", "id", "title"):
            value = record.get(key)
            if value:
                text = str(value).strip()
                if text:
                    return text[:80]
        return "unknown record"

    async def _process_record_with_retry(
        self,
        record: Dict[str, Any],
        prompt_template: str,
        system_prompt: Optional[str] = None,
        output_field_names: Optional[List[str]] = None,
        record_retries: int = 2,
        skip_failed_records: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Process one record, retrying transient/parse failures before skip or raise."""
        label = self._record_label(record)
        last_error: Optional[BaseException] = None
        attempts = max(1, int(record_retries))

        for attempt in range(attempts):
            try:
                return await self.process_record(
                    record,
                    prompt_template,
                    system_prompt,
                    output_field_names,
                )
            except (ValueError, LLMRetryExhausted, TypeError, AttributeError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    print(
                        f"  WARNING: LLM failed for {label!r} "
                        f"(attempt {attempt + 1}/{attempts}); retrying: {exc}"
                    )
                    continue
                if skip_failed_records:
                    print(
                        f"  WARNING: Skipping {label!r} after {attempts} "
                        f"failed attempt(s): {last_error}"
                    )
                    return None
                raise RuntimeError(
                    f"LLM processing failed for {label!r} after {attempts} attempt(s): "
                    f"{last_error}"
                ) from last_error

        return None
    
    async def process_records(
        self,
        records: List[Dict[str, Any]],
        prompt_template: str,
        system_prompt: Optional[str] = None,
        output_field_names: Optional[List[str]] = None,
        batch_size: int = 1,
        max_concurrent: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        record_retries: int = 2,
        skip_failed_records: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process a list of records with the LLM.

        Processes records one by one (or in small batches) and adds extracted fields
        to each record.

        Args:
            records: List of dictionaries to process
            prompt_template: Prompt template string. Use {record} or {record_json} to insert the record.
            system_prompt: Optional system prompt
            output_field_names: Optional list of expected field names in the JSON output
            batch_size: Number of records to include in each LLM call (default: 1, process individually)
            max_concurrent: Maximum number of concurrent LLM calls (default: 5)
            progress_callback: Optional callback function called with (completed, total) after each record
            record_retries: Per-record attempts (incl. first) before skip or raise (default: 2)
            skip_failed_records: If True, omit records that still fail after record_retries

        Returns:
            List of enriched records. When skip_failed_records is False, same length as input.
            When True, failed records are omitted (list may be shorter than input).
        """
        if batch_size == 1:
            # Process records individually with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            completed = 0

            async def process_with_semaphore(
                record: Dict[str, Any],
            ) -> Optional[Dict[str, Any]]:
                nonlocal completed
                async with semaphore:
                    result = await self._process_record_with_retry(
                        record,
                        prompt_template,
                        system_prompt,
                        output_field_names,
                        record_retries=record_retries,
                        skip_failed_records=skip_failed_records,
                    )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(records))
                    return result

            tasks = [process_with_semaphore(record) for record in records]
            results = await asyncio.gather(*tasks)
            if skip_failed_records:
                return [result for result in results if result is not None]
            return results  # type: ignore[return-value]
        else:
            # Process in batches
            enriched_records = []
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                # For batch processing, combine records in the prompt
                batch_json = json.dumps(batch, indent=2, ensure_ascii=False)
                prompt = prompt_template.format(record=batch, record_json=batch_json)
                
                if system_prompt is None:
                    system_prompt = (
                        "You are a helpful assistant that extracts and processes information. "
                        "You MUST respond with a valid JSON array of objects. "
                        "Each object in the array corresponds to one record in the input. "
                        "Do not include any introductory text, explanations, or markdown formatting. "
                        "Just return the JSON array itself."
                    )
                
                response_text = await self._call_llm(prompt, system_prompt)
                extracted_batch = self._parse_json_response(response_text)
                
                # If response is a list, merge each item with corresponding record
                if isinstance(extracted_batch, list):
                    for record, extracted in zip(batch, extracted_batch):
                        enriched_record = record.copy()
                        enriched_record.update(extracted)
                        enriched_records.append(enriched_record)
                else:
                    # If response is a single dict, apply to all records in batch
                    for record in batch:
                        enriched_record = record.copy()
                        enriched_record.update(extracted_batch)
                        enriched_records.append(enriched_record)
            
            return enriched_records


@async_cache(
    prefix="llm_process_records",
    ttl=3600,  # Cache for 1 hour
)
async def process_records_with_llm(
    records: List[Dict[str, Any]],
    prompt_template: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    output_field_names: Optional[List[str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    batch_size: int = 1,
    max_concurrent: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    record_retries: int = 2,
    skip_failed_records: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function to process records with LLM.
    
    This is a standalone function that creates an LLMProcessor and processes records.
    Use this for simple use cases.
    
    Args:
        records: List of dictionaries to process
        prompt_template: Prompt template string. Use {record} or {record_json} to insert the record.
        model: OpenRouter model slug; if None, uses OPENROUTER_MODEL or openai/gpt-4o-mini.
        api_key: If None, uses OPENROUTER_API_KEY then OPENAI_API_KEY.
        base_url: If None, uses OPENROUTER_BASE_URL or https://openrouter.ai/api/v1.
        system_prompt: Optional system prompt
        output_field_names: Optional list of expected field names in the JSON output
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens in response (default: 2000)
        batch_size: Number of records per LLM call (default: 1)
        max_concurrent: Maximum concurrent LLM calls (default: 5)
        progress_callback: Optional callback function called with (completed, total) after each record
        record_retries: Per-record attempts before skip or raise (default: 2)
        skip_failed_records: If True, omit records that still fail after record_retries
    
    Returns:
        List of enriched records (shorter than input when skip_failed_records is True)
    
    Example:
        >>> records = [
        ...     {"title": "AI in Medicine", "author": "John Doe"},
        ...     {"title": "Machine Learning", "author": "Jane Smith"}
        ... ]
        >>> enriched = await process_records_with_llm(
        ...     records,
        ...     prompt_template="Extract key topics from: {record_json}",
        ...     output_field_names=["topics", "summary"]
        ... )
    """
    processor = LLMProcessor(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return await processor.process_records(
        records=records,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        output_field_names=output_field_names,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        progress_callback=progress_callback,
        record_retries=record_retries,
        skip_failed_records=skip_failed_records,
    )

