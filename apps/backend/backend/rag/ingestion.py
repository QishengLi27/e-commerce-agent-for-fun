"""
LlamaIndex ingestion pipeline for product descriptions.

Parses product_descriptions.txt, chunks with SentenceWindowNodeParser,
embeds with Zhipu embedding-2, and stores in pgvector product_chunks collection.

Usage:
    python -m backend.rag.ingestion
"""

import contextlib
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from langchain_openai import OpenAIEmbeddings
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.postgres import PGVectorStore

from backend.config import settings
from backend.rag.embedding_adapter import LangChainEmbeddingAdapter

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PRODUCT_DESCRIPTIONS_PATH = (
    Path(__file__).parent.parent.parent / "data" / "product_descriptions.txt"
)
COLLECTION_NAME = "product_chunks"
SENTENCE_WINDOW_SIZE = 5
CHUNK_SIZE = 64  # batch size for embedding API


def _parse_db_url() -> dict:
    """Parse SQLAlchemy database_url into PGVectorStore parameters."""
    parsed = urlparse(settings.database_url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "postgres",
        "password": parsed.password or "postgres",
        "database": parsed.path.lstrip("/") if parsed.path else "ecommerce",
    }


# ── Parser ────────────────────────────────────────────────────────────────────

_BLOCK_HEADER_RE = re.compile(r"^\[product:\s*(.+?)\]$")
_METADATA_RE = re.compile(r"^(\w[\w\s]*?):\s*(.+)$")


def parse_product_descriptions(filepath: str | Path) -> list[Document]:
    """Parse [product: Name] blocks with metadata headers and body text.

    Format:
        [product: iPhone 15 Pro]
        category: Smartphones
        brand: Apple
        price: 999.00

        The product description body text follows the metadata lines...

    Args:
        filepath: Path to product_descriptions.txt.

    Returns:
        List of LlamaIndex Document objects with metadata attached.
    """
    documents: list[Document] = []
    current_meta: dict[str, str] = {}
    current_name: str | None = None
    current_body: list[str] = []

    def _flush():
        """Save accumulated product block as a Document."""
        nonlocal current_meta, current_name, current_body
        if current_name and current_body:
            text = "\n".join(current_body).strip()
            if text:
                num_meta: dict[str, str | int | float | bool] = {
                    "product_name": current_name,
                    "brand": current_meta.get("brand", ""),
                    "category": current_meta.get("category", ""),
                    "color": current_meta.get("color", ""),
                    "storage": current_meta.get("storage", ""),
                }
                # Convert numeric metadata
                if "price" in current_meta:
                    with contextlib.suppress(ValueError):
                        num_meta["price"] = float(current_meta["price"])
                if "screen_size" in current_meta:
                    with contextlib.suppress(ValueError):
                        num_meta["screen_size"] = float(current_meta["screen_size"])
                if "release_year" in current_meta:
                    with contextlib.suppress(ValueError):
                        num_meta["release_year"] = int(current_meta["release_year"])
                if "wireless" in current_meta:
                    num_meta["wireless"] = current_meta["wireless"].lower() == "true"

                documents.append(Document(text=text, metadata=num_meta))
                logger.debug(
                    "Parsed product: %s (%d chars, %d metadata fields)",
                    current_name,
                    len(text),
                    len(num_meta),
                )

        current_meta = {}
        current_name = None
        current_body = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # Empty line — could be separator between metadata and body
            if not line:
                if current_name and current_meta and not current_body:
                    continue  # skip blank lines between metadata and body
                if current_body:
                    current_body.append("")  # preserve paragraph breaks
                continue

            # New product block header
            header_match = _BLOCK_HEADER_RE.match(line)
            if header_match:
                _flush()
                current_name = header_match.group(1)
                continue

            # Metadata line (key: value) — only before body starts
            meta_match = _METADATA_RE.match(line)
            if meta_match and not current_body:
                key = meta_match.group(1).strip().lower().replace(" ", "_")
                value = meta_match.group(2).strip()
                current_meta[key] = value
                continue

            # Body text
            if current_name:
                current_body.append(line)

    _flush()  # Don't forget the last block
    logger.info("Parsed %d product documents from %s", len(documents), filepath)
    return documents


# ── Ingestion Pipeline ─────────────────────────────────────────────────────────


def build_product_index():
    """Run the full ingestion pipeline: parse → chunk → embed → store.

    Creates or replaces the 'product_chunks' collection in pgvector.
    """
    logger.info("Starting product description ingestion...")

    # Parse
    documents = parse_product_descriptions(PRODUCT_DESCRIPTIONS_PATH)
    if not documents:
        logger.warning("No product documents found in %s", PRODUCT_DESCRIPTIONS_PATH)
        return

    # Create embedding model (Zhipu via OpenAI-compatible API)
    lc_embed = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        chunk_size=CHUNK_SIZE,
    )
    embed_model = LangChainEmbeddingAdapter(lc_embed)

    # Create vector store (pgvector)
    db_params = _parse_db_url()
    vector_store = PGVectorStore.from_params(
        database=db_params["database"],
        host=db_params["host"],
        port=db_params["port"],
        user=db_params["user"],
        password=db_params["password"],
        table_name=COLLECTION_NAME,
        embed_dim=settings.embedding_dim,
        perform_setup=True,  # ensure table + index exist
    )

    # Build pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceWindowNodeParser(
                window_size=SENTENCE_WINDOW_SIZE,
            ),
            embed_model,
        ],
        vector_store=vector_store,
    )

    # Run
    nodes = pipeline.run(documents=documents)
    logger.info(
        "Ingestion complete: %d documents → %d nodes stored in %s",
        len(documents),
        len(nodes),
        COLLECTION_NAME,
    )

    return nodes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_product_index()
