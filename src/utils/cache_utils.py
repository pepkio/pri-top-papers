"""
Generic async caching utilities.

This module provides decorators for caching async function results.
"""

import asyncio
import hashlib
import json
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime, timedelta


class AsyncCache:
    """
    Simple file-based async cache with TTL support.
    """

    def __init__(self, cache_dir: str = ".cache", default_ttl: int = 3600):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl

    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate a cache key from function name and arguments.

        Args:
            func_name: Function name
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Cache key string
        """
        # Create a hash of the arguments
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        content = f"{func_name}:{args_str}:{kwargs_str}"

        # Use SHA256 hash for the key
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache if it exists and hasn't expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # Check TTL
            if datetime.now() > data['expires_at']:
                # Expired, remove file
                cache_path.unlink(missing_ok=True)
                return None

            return data['value']

        except (pickle.UnpicklingError, KeyError, EOFError):
            # Corrupted cache file, remove it
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl

        expires_at = datetime.now() + timedelta(seconds=ttl)

        data = {
            'value': value,
            'expires_at': expires_at,
        }

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            # If caching fails, just continue (fail silently)
            pass

    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)


# Global cache instance
_cache = AsyncCache()


def async_cache(
    prefix: str = "",
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
) -> Callable:
    """
    Decorator for caching async function results.

    Args:
        prefix: Prefix for cache keys to avoid conflicts
        ttl: Time to live in seconds (uses cache default if None)
        key_func: Custom function to generate cache key from (func, args, kwargs)

    Returns:
        Decorated function

    Example:
        @async_cache(prefix="llm_process", ttl=3600)
        async def process_records(records, model="gpt-4"):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(func, args, kwargs)
            else:
                func_name = f"{prefix}:{func.__name__}" if prefix else func.__name__
                cache_key = _cache._get_cache_key(func_name, args, kwargs)

            # Try to get from cache first
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Not in cache, call the function
            result = await func(*args, **kwargs)

            # Cache the result
            _cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


def clear_cache() -> None:
    """Clear all cached results."""
    _cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache_files = list(_cache.cache_dir.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files if f.exists())

    return {
        "cache_dir": str(_cache.cache_dir),
        "total_files": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
    }


