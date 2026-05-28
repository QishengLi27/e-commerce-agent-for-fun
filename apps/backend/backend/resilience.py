"""
Resilience patterns for the AI agent:
- Circuit breaker: stops calling a failing API temporarily
- Retry with backoff: retries transient errors (500, 502, 503, 429)
- Graceful degradation: returns friendly fallbacks when APIs are down
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# -- Circuit Breaker ----------------------------------------------------------

class CircuitBreaker:
    """
    Prevents infinite API calls when a service is down.

    States:
        CLOSED  -> normal operation
        OPEN    -> too many failures, reject calls for cooldown period
        HALF_OPEN -> after cooldown, allow one test call
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, fallback: Callable, *args, **kwargs) -> Any:
        if self.state == "OPEN":
            assert self.last_failure_time is not None
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info(f"[{self.name}] Circuit HALF_OPEN - testing...")
            else:
                logger.warning(f"[{self.name}] Circuit OPEN - using fallback")
                return fallback(*args, **kwargs)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            if self.state == "OPEN":
                logger.warning(f"[{self.name}] Circuit tripped - using fallback")
                return fallback(*args, **kwargs)
            raise

    def _on_success(self):
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info(f"[{self.name}] Circuit CLOSED")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"[{self.name}] Circuit OPENED after {self.failure_count} failures")


# -- Retry Decorator ----------------------------------------------------------

def make_retry_decorator(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    """Create a tenacity retry decorator for transient API errors."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# -- Fallbacks ----------------------------------------------------------------

class Fallbacks:
    """Static fallback responses when services are unavailable."""

    @staticmethod
    def llm_unavailable(*args, **kwargs) -> str:
        return (
            "I'm currently experiencing high demand and can't process your request. "
            "Please try again in a few moments."
        )

    @staticmethod
    def clean_query_failed(*args, **kwargs) -> str:
        return ""

    @staticmethod
    def retrieval_failed(*args, **kwargs) -> str:
        return "I'm unable to search our policies right now. Please contact support directly."


# -- Transient Error Detection ------------------------------------------------

def is_transient_error(e: Exception) -> bool:
    """Check if an exception looks like a transient network/API issue."""
    transient_codes = (429, 500, 502, 503, 504)
    error_str = str(e).lower()
    if "rate limit" in error_str or "timeout" in error_str:
        return True
    return any(str(code) in error_str for code in transient_codes)
