# Taxonomy Application Layer Design

**Status:** Draft  
**Date:** 2026-05-31  
**Scope:** Wire the 5 new taxonomy tables (entity_synonyms, attribute_definitions, product_attributes, product_relations, hierarchical categories) into the application layer — KnowledgeStore, retrievers, tools, and intent classifier.

## Background

`backend/knowledge/schema.py` has been updated with 5 new taxonomy tables and rich seed data covering 28 hierarchical categories, 29 products, 9 attribute definitions, 50+ bilingual synonyms, 6 entity disambiguation tags, and 20+ product relations. The schema is complete but the application layer (`graph_store.py`, `retrievers.py`, `tools/knowledge.py`, `intent/semantic.py`) still uses the old flat model.

## Goals

1. **Better product discovery** — synonym-aware search, attribute filtering, hierarchical category navigation
2. **Smarter policy matching** — taxonomy-aware policy lookup with category-tree inheritance
3. **Richer product recommendations** — accessory/alternative/compatible suggestions via product_relations

## Approach: 4 Vertical Slices

Each slice delivers working, testable value end-to-end before moving to the next. Layering within each slice is always: KnowledgeStore → retrievers → tools → intent (if applicable).

---

## Slice 1: Entity Synonyms

### User Impact
"苹果手机" resolves to iPhone 15 Pro. Brand/category/attribute-value synonyms in Chinese all work. Queries with bilingual terms match correctly.

### KnowledgeStore changes (`graph_store.py`)

Three new methods:

```
resolve_synonym(term: str, entity_type: str | None = None) -> str | None
```
- Looks up a single term in `entity_synonyms` table
- Returns canonical_name if found, None otherwise
- When entity_type is specified, restricts match to that type (brand/product/category/attribute_value)
- When multiple matches exist, returns the one with highest confidence
- SQL: `SELECT canonical_name, confidence FROM entity_synonyms WHERE synonym ILIKE %s [AND entity_type = %s] ORDER BY confidence DESC LIMIT 1`

```
expand_query(query: str) -> list[str]
```
- Tokenizes the query into n-grams (single words, adjacent pairs, triples)
- For each n-gram, checks if it's a synonym
- Returns the original query + all expanded variants with canonical names substituted
- Uses longest-match-first: if "苹果手机" matches as a 3-gram, don't also match "手机" separately
- Example: "return 苹果手机" → ["return 苹果手机", "return iPhone 15 Pro"]

```
get_product_info_with_synonyms(query: str) -> dict | None
```
- Convenience wrapper: tries `get_product_info(query)` first, then resolves synonyms and retries
- Returns product info dict or None

Also update existing methods:
- `_extract_product_name(query)` — after loading `_product_names`, also check entity_synonyms for each word in the query, add canonical names to the search set
- `search_products(keyword)` — try keyword first, then try synonym-expanded terms as fallback

### Retrievers changes (`retrievers.py`)

- `GraphPolicyRetriever.retrieve()` — before calling `store.query_product_policies(query)`, call `store.expand_query(query)` and try the expanded terms. Return results from the first expansion that yields policies.
- `VectorPolicyRetriever` — no changes (embedding-based, synonyms are a graph concern)
- `HybridRetriever` — inherits changes via GraphPolicyRetriever

### Tools changes (`tools/knowledge.py`)

- `product_info_tool(query)` — wraps existing logic with `get_product_info_with_synonyms` instead of raw `get_product_info`
- `category_info_tool(query)` — resolves category synonyms before lookup

### Intent changes (`intent/semantic.py`)

- `_search_products_semantic(query)` already calls `kg.search_products(keyword)`. The KnowledgeStore update to `search_products` handles synonym expansion internally, so no direct changes needed here.

### What stays unchanged
- `entity_tags` (disambiguation) — saved for a future slice
- No new endpoints or API changes
- No embedding changes

### Tests
- Unit: `resolve_synonym` — exact match, no match, ambiguous (returns highest confidence)
- Unit: `expand_query` — single synonym, multiple synonyms, no synonyms, longest-match priority
- Integration: `product_info_tool("苹果手机")` returns iPhone 15 Pro info
- Integration: `product_info_tool("索尼耳机")` returns Sony WH-1000XM5 info

---

## Slice 2: Hierarchical Categories

### User Impact
Querying "Electronics" returns results from Smartphones, Laptops, Headphones, and all sub-categories. Policy lookups walk up the category tree to find all applicable policies. `get_all_categories` returns a tree instead of a flat list.

### KnowledgeStore changes

New methods:

```
get_category_tree(parent_id: int | None = None) -> list[dict]
```
- Returns the full category hierarchy as nested dicts
- Uses recursive CTE on `parent_id` and `path` columns
- Each node: `{name, description, level, path, children: [...]}`
- When parent_id is None, returns all root categories with their subtrees

```
get_subcategory_ids(category_name: str) -> list[int]
```
- Given a category name, returns the category's own ID + all descendant IDs
- Uses the materialized `path` column for efficient lookup: `WHERE path LIKE (SELECT path FROM categories WHERE name = %s) || '.%'`
- Used by policy lookups to include sub-categories

```
get_category_policies_inherited(category_name: str) -> list[dict]
```
- Traverses UP the tree: finds policies for the category, then its parent, then grandparent, etc.
- Merges results, deduplicating by policy name
- Uses `parent_id` chain: `WITH RECURSIVE ancestors AS (...)`

Update existing methods:
- `query_product_policies(query)` — after finding the product's category, use `get_subcategory_ids` to also match policies mapped to parent categories. Currently policies are mapped to leaf categories only; with hierarchy, a product in "Flagship Phones" inherits policies from "Smartphones", "Mobile Devices", and "Electronics".
- `query_category_policies(query)` — same treatment: include inherited policies from ancestor categories
- `get_all_categories()` — return tree structure instead of flat list
- `search_products(keyword)` — when searching by category, include sub-category products

### Retrievers changes

- `GraphPolicyRetriever` — already calls `store.query_product_policies()` and `store.query_category_policies()`, which now return inherited policies automatically
- `GraphPolicyRetriever._format_policies()` — add a line showing the inheritance chain, e.g., "Applies to: Flagship Phones → Smartphones → Mobile Devices → Electronics"

### Tools changes

- `category_info_tool(query)` — when listing categories, return the tree view; when searching for products in a category, include sub-category results
- `product_info_tool(query)` — policies list now includes inherited policies that were already resolved by KnowledgeStore

### Intent changes

- No direct changes needed — existing keyword extraction already works with category names

### Tests
- Unit: `get_subcategory_ids("Electronics")` returns IDs for all sub-categories
- Unit: `get_category_policies_inherited("Flagship Phones")` returns electronics_return
- Unit: `get_category_tree()` returns proper nesting (max depth 4)
- Integration: product in "Flagship Phones" gets electronics_return policy inherited from "Electronics"
- Integration: `search_products("Electronics")` returns products from all sub-categories

---

## Slice 3: Product Attributes

### User Impact
Queries like "red headphones under $200" or "Apple laptops with 512GB storage" return filtered results. The agent can answer attribute-specific questions: "What color is the iPhone 15 Pro?" / "Which phones have 256GB storage?"

### KnowledgeStore changes

New methods:

```
get_product_attributes(product_name: str) -> list[dict]
```
- Returns all attributes for a product as a list of `{name, display_name, value, data_type, unit}`
- JOINs product_attributes → attribute_definitions
- Picks the correct value column based on data_type (value_text / value_number / value_boolean)

```
filter_products_by_attributes(filters: list[dict]) -> list[dict]
```
- Each filter dict: `{attribute_name: str, operator: str, value: any}`
- Operators: `eq`, `gt`, `gte`, `lt`, `lte`, `contains` (for text), `between`
- SQL: dynamic WHERE clauses joining product_attributes + attribute_definitions
- Returns matching products with their attribute values
- Example: `[{attribute_name: "brand", operator: "eq", value: "Apple"}, {attribute_name: "storage", operator: "eq", value: "256GB"}]`

```
get_facet_counts(attribute_name: str, category_name: str | None = None) -> dict
```
- Returns distinct values + counts for a given attribute, optionally scoped to a category
- Used for faceted navigation ("filter by brand" shows Apple (5), Samsung (3), etc.)

Update existing methods:
- `get_product_info(query)` — also return product attributes in the result dict
- `search_products(keyword)` — no changes needed (attributes are additive)

### Retrievers changes

- No direct changes — retrievers handle policy queries, not product filtering
- `GraphPolicyRetriever` uses `get_product_info` internally, which now includes attributes automatically

### Tools changes

- `product_info_tool(query)` — already uses `get_product_info`; update the output formatter to display attributes
- New optional tool (decide during implementation): `filter_products_tool` exposing `filter_products_by_attributes` — only if there's a clear user-facing query pattern that needs it (YAGNI check at implementation time)

### Intent changes

- `_search_products_semantic(query)` — if the query contains attribute-like patterns ("red", "256GB", "Apple"), extract them and pass to `filter_products_by_attributes` before falling back to keyword search

### Tests
- Unit: `get_product_attributes("iPhone 15 Pro")` returns all 8 attributes
- Unit: `filter_products_by_attributes([{attribute: "brand", op: "eq", value: "Apple"}])` returns all Apple products
- Unit: `filter_products_by_attributes([{attribute: "color", op: "eq", value: "Black"}])` returns black products only
- Integration: `product_info_tool("iPhone 15 Pro")` output includes attributes section

---

## Slice 4: Product Relations

### User Impact
"What accessories do you have for iPhone 15 Pro?" returns cases, chargers, screen protectors. "What's an alternative to MacBook Pro 16?" returns Dell XPS 15, ASUS ROG Strix G16. "Is this cable compatible with my Samsung?" answers yes/no.

### KnowledgeStore changes

New methods:

```
get_related_products(product_name: str, relation_type: str | None = None) -> list[dict]
```
- Returns products related to the given product
- JOINs product_relations → products on both source and target
- When relation_type specified, filters to that type only (accessory/alternative/compatible/bundle/upgrade)
- Results include: `{product_name, relation_type, strength, direction ("source"|"target"), price, category}`
- Checks both directions (product as source AND as target)

```
get_accessories(product_name: str) -> list[dict]
```
- Convenience method: `get_related_products(product_name, relation_type="accessory")`

```
get_alternatives(product_name: str) -> list[dict]
```
- Convenience method: `get_related_products(product_name, relation_type="alternative")`

Update existing methods:
- `get_product_info(query)` — include `accessories` and `alternatives` lists in the return dict

### Retrievers changes

- No direct changes — retrievers handle policy queries, not product recommendations

### Tools changes

- `product_info_tool(query)` — output includes "Accessories:" and "Alternatives:" sections if the product has relations
- No new standalone tools needed — the relation data is surfaced inline with product info

### Intent changes

- No changes needed — the agent discovers relations through `product_info_tool` naturally

### Tests
- Unit: `get_accessories("iPhone 15 Pro")` returns case, AirPods, screen protector, charger
- Unit: `get_alternatives("MacBook Pro 16")` returns Dell XPS 15, ASUS ROG Strix G16
- Integration: `product_info_tool("iPhone 15 Pro")` output includes accessories and alternatives sections

---

## Cross-Cutting Concerns

### Error handling
- All new KnowledgeStore methods follow the existing pattern: return empty lists/None on no match, let exceptions propagate to the circuit breaker
- No try/except in query methods — the resilience layer (`backend/resilience.py`) handles retries

### Performance
- Synonym lookup is a single indexed query (UNIQUE constraint on `synonym, entity_type, language`)
- Hierarchy queries use materialized `path` column + recursive CTEs — fine for 28 categories
- Attribute filtering uses indexed JOINs on (product_id, attribute_id) — fine for hundreds of products
- Product relations are simple JOINs with UNIQUE constraint — negligible for <1000 relations

### Backward compatibility
- All existing public method signatures remain unchanged
- New methods are additive only
- Existing callers (agent.py, graph/nodes.py, api/routes.py) continue to work without modification

### What this design does NOT cover
- Entity disambiguation tags (`entity_tags`) — requires LLM integration to resolve "Apple" (brand vs fruit), deferred to a future pass
- Product embeddings for vector search (mentioned in `intent/semantic.py` production note) — out of scope
- Faceted search API — the `facet_counts` method is built for internal use; a public faceted search endpoint is a separate feature
- Admin UI for managing taxonomy data — schema.py seeding is sufficient for now
