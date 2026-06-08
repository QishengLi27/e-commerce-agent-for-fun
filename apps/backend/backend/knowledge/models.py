"""Shared dataclass models for knowledge graph entities.

These replace raw dict returns from KnowledgeStore and Neo4jStore,
providing type-safe, self-documenting return values.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProductRef:
    """Lightweight product reference (list/search results)."""

    name: str
    price: float | None
    sku: str | None = None
    category_name: str | None = None


@dataclass(frozen=True)
class CategoryRef:
    """Category node reference with hierarchy info."""

    name: str
    level: int
    description: str | None = None
    children: list["CategoryRef"] = field(default_factory=list)


@dataclass(frozen=True)
class ProductAttribute:
    """Typed attribute value for a product."""

    name: str
    display_name: str
    value: str | int | float | bool
    data_type: str
    unit: str | None = None


@dataclass(frozen=True)
class PolicySummary:
    """Policy reference returned from graph traversal."""

    name: str
    policy_type: str
    summary: str
    details: str


@dataclass(frozen=True)
class ProductInfo:
    """Full product detail including attributes, policies, and relations."""

    name: str
    price: float | None
    sku: str | None
    category_name: str | None
    category_path: list[str] = field(default_factory=list)
    attributes: list[ProductAttribute] = field(default_factory=list)
    policies: list[PolicySummary] = field(default_factory=list)
    accessories: list[ProductRef] = field(default_factory=list)
    alternatives: list[ProductRef] = field(default_factory=list)


@dataclass(frozen=True)
class QAResult:
    """Result from product Q&A: retrieved context + synthesized answer."""

    answer: str
    matched_products: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
