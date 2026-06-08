"""Unit tests for product description parsing and ingestion."""

import tempfile
from pathlib import Path

from backend.rag.ingestion import parse_product_descriptions

SAMPLE_CONTENT = """[product: Test Phone]
category: Smartphones
brand: TestCorp
price: 599.00
color: Red
storage: 128GB

Test Phone features a vibrant display and long battery life.
The camera system captures stunning photos in any light.

Wireless charging is supported with Qi-compatible accessories.
Battery lasts up to 20 hours of typical usage.
"""


class TestParseProductDescriptions:
    def test_parses_single_product(self):
        """Single product block produces one Document with metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(SAMPLE_CONTENT)
            tmp_path = f.name

        try:
            docs = parse_product_descriptions(tmp_path)
            assert len(docs) == 1
            doc = docs[0]

            assert doc.metadata["product_name"] == "Test Phone"
            assert doc.metadata["brand"] == "TestCorp"
            assert doc.metadata["category"] == "Smartphones"
            assert doc.metadata["price"] == 599.00
            assert doc.metadata["color"] == "Red"

            assert "vibrant display" in doc.text
            assert "Wireless charging" in doc.text
        finally:
            Path(tmp_path).unlink()

    def test_parses_multiple_products(self):
        """Multiple product blocks produce multiple Documents."""
        content = (
            SAMPLE_CONTENT + "\n[product: Test Tablet]\ncategory: Tablets\nbrand: TestCorp\n"
            "price: 399.00\n\nA lightweight tablet for everyday use.\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            docs = parse_product_descriptions(tmp_path)
            assert len(docs) == 2
            assert docs[0].metadata["product_name"] == "Test Phone"
            assert docs[1].metadata["product_name"] == "Test Tablet"
        finally:
            Path(tmp_path).unlink()
