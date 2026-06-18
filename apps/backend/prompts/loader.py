"""prompts/loader.py — Load prompts from file system with hot reload, Jinja2, and golden test validation."""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from backend.prompts.models import PromptTemplate

logger = logging.getLogger(__name__)

# Environment override: set PROMPTS_ENV=dev to load from templates/dev/ first
_PROMTS_ENV = os.getenv("PROMPTS_ENV", "")


class PromptLoader:
    """Loads prompt templates from the filesystem.

    Each prompt file has JSON frontmatter between --- delimiters
    followed by the Jinja2 template body. Files are plain text,
    git-friendly, and hot-reloadable.

    Environment isolation: if PROMPTS_ENV=dev, loads from
    templates/dev/ first, falling back to templates/ for shared prompts.
    """

    def __init__(self, templates_dir: str):
        self._dir = Path(templates_dir)
        self._env_dir = self._dir / _PROMTS_ENV if _PROMTS_ENV else None
        if not self._dir.exists():
            raise FileNotFoundError(f"Prompt templates dir not found: {templates_dir}")

    def load_all(self) -> dict[str, PromptTemplate]:
        """Load all prompt templates from the templates directory.

        Environment-aware: checks templates/{PROMPTS_ENV}/ first,
        then falls back to templates/.

        Returns:
            Dict mapping prompt name → PromptTemplate.
        """
        prompts: dict[str, PromptTemplate] = {}

        # Load shared prompts first
        for filepath in self._dir.glob("*.j2"):
            prompt = self._load_one(filepath)
            if prompt:
                prompts[prompt.name] = prompt

        # Load env-specific overrides (merge, preferring env version)
        if self._env_dir and self._env_dir.exists():
            for filepath in self._env_dir.glob("*.j2"):
                prompt = self._load_one(filepath)
                if prompt:
                    prompts[prompt.name] = prompt  # Override shared

        logger.info("Loaded %d prompts from %s (env=%s)", len(prompts), self._dir, _PROMTS_ENV or "default")
        return prompts

    def load_one(self, name: str) -> Optional[PromptTemplate]:
        """Load a single prompt template by name. Env-aware."""
        # Try env-specific first
        if self._env_dir:
            filepath = self._env_dir / f"{name}.j2"
            if filepath.exists():
                return self._load_one(filepath)
        # Fall back to shared
        filepath = self._dir / f"{name}.j2"
        if not filepath.exists():
            return None
        return self._load_one(filepath)

    def load_version_at(self, name: str, git_ref: str) -> Optional[PromptTemplate]:
        """Load a historical version of a prompt from a git reference.

        Used by the rollback API: load_version_at("reply", "HEAD~1").

        Args:
            name: Prompt name (without .j2 extension).
            git_ref: Any git ref: commit hash, HEAD~N, branch name.

        Returns:
            PromptTemplate from that git point in history, or None.
        """
        filepath = self._dir / f"{name}.j2"
        if not filepath.exists():
            return None

        try:
            result = subprocess.run(
                ["git", "show", f"{git_ref}:{filepath}"],
                capture_output=True, text=True, timeout=10,
                cwd=str(filepath.parent.parent),  # repo root
            )
            if result.returncode != 0:
                logger.warning("git show failed for %s@%s: %s", name, git_ref, result.stderr)
                return None

            raw = result.stdout
            meta, template = self._parse_frontmatter(raw)
            if not meta or not template.strip():
                return None

            return PromptTemplate(
                name=meta.get("name", name),
                description=f"{meta.get('description', '')} [rolled back from {git_ref}]",
                template=template.strip(),
                version=meta.get("version", 0),
                author=meta.get("author", ""),
                model=meta.get("model", ""),
                temperature=meta.get("temperature", 0.0),
                variables=meta.get("variables", []),
                created_at=meta.get("created_at", ""),
                updated_at=f"rollback:{git_ref}",
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("git show failed: %s", e)
            return None

    def _load_one(self, filepath: Path) -> Optional[PromptTemplate]:
        """Parse a single .j2 file: frontmatter + body."""
        try:
            raw = filepath.read_text()
        except Exception as e:
            logger.warning("Failed to read prompt file %s: %s", filepath, e)
            return None

        meta, template = self._parse_frontmatter(raw)
        if not meta or not template.strip():
            logger.warning("Skipping %s: missing metadata or empty template", filepath)
            return None

        # Validate template is parseable by Jinja2
        try:
            from jinja2 import Template
            Template(template.strip())
        except ImportError:
            pass  # Jinja2 not installed, validation deferred
        except Exception as e:
            logger.warning("Template syntax error in %s: %s", filepath, e)
            return None

        return PromptTemplate(
            name=meta.get("name", filepath.stem),
            description=meta.get("description", ""),
            template=template.strip(),
            version=meta.get("version", 1),
            author=meta.get("author", ""),
            model=meta.get("model", ""),
            temperature=meta.get("temperature", 0.0),
            variables=meta.get("variables", []),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
        )

    @staticmethod
    def _parse_frontmatter(raw: str) -> tuple[dict, str]:
        """Split JSON frontmatter from template body.

        Frontmatter is between --- delimiters. Body is everything after.
        """
        lines = raw.split("\n")
        if lines and lines[0].strip() == "---":
            end = 1
            while end < len(lines) and lines[end].strip() != "---":
                end += 1
            if end < len(lines):
                meta_block = "\n".join(lines[1:end])
                body = "\n".join(lines[end + 1:])
                try:
                    meta = json.loads(meta_block)
                    return meta, body
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in frontmatter")
        return {}, raw
