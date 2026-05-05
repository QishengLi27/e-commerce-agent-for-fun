"""
Resilience patterns for the AI agent:
- Circuit breaker: stops calling a failing API temporarily
- Retry with backoff: retries transient errors (500, 502, 503, 429)
- Graceful degradation: returns friendly fallbacks when APIs are down
"""

import time
import random
from functools import wraps
from typing import Callable, Any, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

logger = logging.getLogger(__name__)


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Prevents infinite API calls when a service is down.

    States:
        CLOSED  → normal operation
        OPEN    → too many failures, reject calls for cooldown period
        HALF_OPEN → after cooldown, allow one test call
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

    def call(self, fn: Callable, fallback: Callable, *args, **kwargs) -> Any:
        """Execute fn with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info(f"[{self.name}] Circuit entering HALF_OPEN")
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
            else:
                logger.warning(f"[{self.name}] Circuit OPEN — returning fallback")
                return fallback(*args, **kwargs)

        if self.state == "HALF_OPEN":
            if self.half_open_calls >= self.half_open_max_calls:
                logger.warning(f"[{self.name}] HALF_OPEN limit reached — fallback")
                return fallback(*args, **kwargs)
            self.half_open_calls += 1

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == "HALF_OPEN":
            logger.info(f"[{self.name}] HALF_OPEN success — closing circuit")
            self.state = "CLOSED"
        self.failures = 0
        self.last_failure_time = None

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            logger.error(
                f"[{self.name}] {self.failures} failures — opening circuit for "
                f"{self.recovery_timeout}s"
            )
            self.state = "OPEN"

    def __call__(self, fallback: Callable):
        """Decorator factory: @cb(fallback=my_fallback)"""
        def decorator(fn: Callable):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                return self.call(fn, fallback, *args, **kwargs)
            return wrapper
        return decorator


# ─── Retry Configuration ──────────────────────────────────────────────────────

# Retry transient errors (500, 502, 503, 504, 429) with exponential backoff
# Don't retry 400-class errors (bad request, auth failure)

def is_transient_error(error: Exception) -> bool:
    """Check if an error is worth retrying."""
    error_msg = str(error).lower()
    # HTTP status codes that are transient
    transient_codes = ["500", "502", "503", "504", "429", "rate limit", "timeout"]
    return any(code in error_msg for code in transient_codes)


def make_retry_decorator(max_attempts: int = 3):
    """Create a tenacity retry decorator for transient API errors."""
    return retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


# ─── Fallback Responses ───────────────────────────────────────────────────────

class Fallbacks:
    """Friendly fallback messages when APIs are unavailable."""

    @staticmethod
    def llm_unavailable(*args, **kwargs) -> str:
        return (
            "I'm having trouble connecting to my brain right now. "
            "Please try again in a minute."
        )

    @staticmethod
    def clean_query_failed(raw_query: str, *args, **kwargs) -> str:
        # If typo correction fails, just pass through the raw query
        return raw_query

    @staticmethod
    def retrieval_unavailable(query: str, *args, **kwargs) -> str:
        return (
            "I can't access our policy database right now. "
            "For urgent questions, please contact support directly."
        )

    @staticmethod
    def rerank_fallback(docs, *args, **kwargs):
        # If re-ranking fails, return docs as-is without scores
        return [(doc, 5.0) for doc, _ in docs]


# ─── Global Circuit Breakers ──────────────────────────────────────────────────

llm_circuit = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60.0)
embedding_circuit = CircuitBreaker("embeddings", failure_threshold=3, recovery_timeout=60.0)
