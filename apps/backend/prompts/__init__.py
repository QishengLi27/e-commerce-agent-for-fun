"""prompts/ — Centralized prompt management with atomic hot reload.

Usage:
    from backend.prompts import get_prompt, reload_prompts, init_prompts

    # Render a prompt (returns PromptOutput with version metadata)
    prompt = get_prompt("reply")
    output = prompt.render(history=..., question=..., result=...)
    # output.text  — the rendered string
    # output.prompt_name  — "reply"
    # output.prompt_version — 3

    # Hot reload (atomic — all-or-nothing)
    reload_prompts()

    # Rollback a single prompt to a git ref
    rollback_prompt("reply", "HEAD~1")
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from backend.prompts.loader import PromptLoader
from backend.prompts.models import PromptOutput, PromptTemplate

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).parent / "templates"

_registry: dict[str, PromptTemplate] = {}
_loader = PromptLoader(str(_DEFAULT_DIR))
_lock = threading.Lock()


def init_prompts(templates_dir: str | None = None) -> None:
    """Load all prompts at startup. Call once during app initialization."""
    global _registry, _loader
    directory = templates_dir or str(_DEFAULT_DIR)
    _loader = PromptLoader(directory)
    with _lock:
        _registry = _loader.load_all()
    logger.info("Prompt registry initialized with %d prompts", len(_registry))


def reload_prompts(
    golden_tests: dict[str, tuple[dict, list[str]]] | None = None,
) -> tuple[int, list[str]]:
    """Atomic hot reload: parse all → validate → replace.

    Strategy: build a new registry first. If any prompt fails to load
    or fails its golden test, the old registry stays untouched. This
    guarantees all-or-nothing — no half-updated state.

    Args:
        golden_tests: Optional dict of {prompt_name: (test_vars, expected_patterns)}.
            If provided, each prompt must pass its golden test before reload.

    Returns:
        (count, errors) — count of loaded prompts, list of error messages.
    """
    global _registry

    # Phase 1: Parse all prompts into a temporary dictionary
    new_registry: dict[str, PromptTemplate] = {}
    errors: list[str] = []
    for filepath in sorted(Path(_loader._dir).glob("*.j2")):
        prompt = _loader._load_one(filepath)
        if not prompt:
            errors.append(f"Failed to parse: {filepath.name}")
            continue

        # Validate declared variables exist in template
        var_errors = prompt.validate()
        if var_errors:
            errors.append(f"{prompt.name}: {'; '.join(var_errors)}")

        new_registry[prompt.name] = prompt

    if errors:
        logger.warning("Reload aborted — %d parse/validation errors. Registry unchanged.", len(errors))
        return len(_registry), errors

    # Phase 2: Run golden tests on the new prompts (if provided)
    if golden_tests:
        for name, (test_vars, expected_patterns) in golden_tests.items():
            if name not in new_registry:
                errors.append(f"Golden test target '{name}' not found")
                continue
            prompt = new_registry[name]
            test_errors = prompt.validate_with_golden(test_vars, expected_patterns)
            if test_errors:
                errors.append(f"{name} golden test failed: {'; '.join(test_errors)}")

    if errors:
        logger.warning("Reload aborted — %d golden test failures. Registry unchanged.", len(errors))
        return len(_registry), errors

    # Phase 3: Atomic replace (Python dict assignment is atomic, lock for multi-thread)
    with _lock:
        _registry = new_registry

    logger.info("Atomic reload complete: %d prompts loaded, 0 errors", len(_registry))
    return len(_registry), []


def get_prompt(name: str) -> PromptTemplate:
    """Thread-safe prompt lookup by name. Raises KeyError if not found.

    Usage:
        prompt = get_prompt("reply")
        output = prompt.render(history=h, question=q, result=r)
    """
    with _lock:
        if name not in _registry:
            raise KeyError(
                f"Prompt '{name}' not found. Available: {list(_registry.keys())}"
            )
        return _registry[name]


def list_prompts() -> list[PromptTemplate]:
    """Return all registered prompts, sorted by name."""
    with _lock:
        return sorted(_registry.values(), key=lambda p: p.name)


def get_prompt_names() -> list[str]:
    """Return names of all registered prompts."""
    with _lock:
        return sorted(_registry.keys())


def reload_one(name: str) -> Optional[PromptTemplate]:
    """Hot reload a single prompt. Returns the updated prompt or None."""
    prompt = _loader.load_one(name)
    if prompt:
        with _lock:
            _registry[name] = prompt
        logger.info("Prompt '%s' reloaded (v%d)", name, prompt.version)
    return prompt


def rollback_prompt(name: str, git_ref: str) -> Optional[PromptTemplate]:
    """Rollback a prompt to a historical version from git.

    Example: rollback_prompt("reply", "HEAD~1") loads the previous commit's version.

    This is a fast, one-line rollback — no git revert + commit + push cycle.
    For permanent rollback, follow up with a git commit updating the file.
    """
    prompt = _loader.load_version_at(name, git_ref)
    if prompt:
        with _lock:
            _registry[name] = prompt
        logger.info("Prompt '%s' rolled back to %s (v%d)", name, git_ref, prompt.version)
    else:
        logger.warning("Rollback failed: prompt '%s' not found at %s", name, git_ref)
    return prompt
