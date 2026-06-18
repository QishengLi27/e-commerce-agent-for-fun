"""prompts/models.py — Prompt template dataclass with metadata."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned, metadata-rich prompt template.

    Immutable by design — "modify" creates a new version.
    """

    name: str
    """Unique identifier: 'reply', 'reply_strict', 'validation', etc."""

    description: str
    """What this prompt does and when it's used."""

    template: str
    """The prompt text with {placeholder} variables for .format()."""

    version: int = 1
    """Monotonic version number. Incremented on every change."""

    author: str = ""
    """Who last modified this prompt."""

    model: str = ""
    """Which LLM model this prompt is optimized for."""

    temperature: float = 0.0
    """Recommended temperature for this prompt."""

    variables: list[str] = field(default_factory=list)
    """Expected template variables: ['history', 'question', 'result']."""

    created_at: str = ""
    """ISO timestamp of creation."""

    updated_at: str = ""
    """ISO timestamp of last modification."""

    def render(self, **kwargs) -> "PromptOutput":
        """Render the template with Jinja2.

        Template syntax: {{ variable }} for substitution,
        {% raw %}...{% endraw %} for literal curly braces (JSON examples).

        Returns a PromptOutput carrying the rendered text AND version metadata.
        """
        from jinja2 import Template
        text = Template(self.template).render(**kwargs)

        return PromptOutput(
            text=text,
            prompt_name=self.name,
            prompt_version=self.version,
        )

    def validate(self) -> list[str]:
        """Check that all declared variables exist in the template.

        Returns list of error messages. Empty list = valid.
        """
        errors = []
        for var in self.variables:
            if f"{{{var}}}" not in self.template:
                errors.append(f"Missing {{{var}}} in template body")
        return errors

    def validate_with_golden(
        self, test_variables: dict, expected_patterns: list[str]
    ) -> list[str]:
        """Run a golden test: render with test input, check output.

        Args:
            test_variables: Dict of variable values to test-render with.
            expected_patterns: List of strings that MUST appear in output.

        Returns:
            List of errors. Empty = golden test passed.
        """
        errors = []
        try:
            output = self.render(**test_variables)
        except KeyError as e:
            return [f"Render failed: missing variable {e}"]
        except Exception as e:
            return [f"Render crashed: {e}"]

        for i, pattern in enumerate(expected_patterns):
            if pattern not in output.text:
                errors.append(
                    f"Golden test pattern #{i} not found: '{pattern[:60]}...'"
                )
        return errors


@dataclass(frozen=True)
class PromptOutput:
    """The rendered prompt text with version traceability built in.

    Every LLM call should log prompt_name + prompt_version alongside
    the response. This enables: "why did the agent say X at 14:32?"
    → "reply.j2 v3, rendered with history=[...], question='...'"
    """

    text: str
    """The rendered prompt string ready for the LLM."""

    prompt_name: str
    """Which prompt produced this output: 'reply', 'validation', etc."""

    prompt_version: int
    """Which version of the prompt was used. Traceable to git history."""


@dataclass
class PromptVersion:
    """Lightweight version info for listing/rollback."""

    name: str
    version: int
    updated_at: str
    author: str
    description: str
    change_note: str = ""
