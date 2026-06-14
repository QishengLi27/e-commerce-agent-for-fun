"""
LlamaIndex query engine for product description RAG.

Provides metadata-filtered vector search over product_chunks
with sentence-window context expansion.
"""

from urllib.parse import urlparse

from langchain_openai import OpenAIEmbeddings
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.postgres import PGVectorStore

from backend.config import settings
from backend.rag.embedding_adapter import LangChainEmbeddingAdapter

COLLECTION_NAME = "product_chunks"
DEFAULT_TOP_K = 5


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


def _build_vector_store() -> PGVectorStore:
    """Create a PGVectorStore connected to the product_chunks collection."""
    db_params = _parse_db_url()
    return PGVectorStore.from_params(
        database=db_params["database"],
        host=db_params["host"],
        port=db_params["port"],
        user=db_params["user"],
        password=db_params["password"],
        table_name=COLLECTION_NAME,
        embed_dim=settings.embedding_dim,
        perform_setup=True,
    )


def _build_embed_model() -> LangChainEmbeddingAdapter:
    """Create an embedding model configured for Zhipu API."""
    lc_embed = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        chunk_size=64,
    )
    return LangChainEmbeddingAdapter(lc_embed)


def create_product_query_engine() -> RetrieverQueryEngine:
    """Factory: build a metadata-filtered query engine for product Q&A.

    Returns a RetrieverQueryEngine configured with:
      - Vector search over product_chunks in pgvector
      - Sentence-window context expansion (MetadataReplacementPostProcessor)
      - Metadata filtering support (filter by product_name, brand, category)
    """
    embed_model = _build_embed_model()
    vector_store = _build_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=DEFAULT_TOP_K,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ],
    )


def create_filtered_query_engine(
    product_names: list[str] | None = None,
    brand: str | None = None,
    category: str | None = None,
) -> RetrieverQueryEngine:
    """Factory: build a query engine with metadata filters applied.

    Args:
        product_names: If provided, limit search to these products' chunks.
        brand: If provided, filter by brand.
        category: If provided, filter by category.

    Returns:
        RetrieverQueryEngine with metadata filters pre-applied.
    """
    embed_model = _build_embed_model()
    vector_store = _build_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # Build metadata filters
    filters_list: list[MetadataFilter | MetadataFilters] = []
    if product_names:
        filters_list.append(
            MetadataFilter(
                key="product_name",
                value=product_names,
                operator=FilterOperator.IN,
            )
        )
    if brand:
        filters_list.append(
            MetadataFilter(
                key="brand",
                value=brand,
                operator=FilterOperator.EQ,
            )
        )
    if category:
        filters_list.append(
            MetadataFilter(
                key="category",
                value=category,
                operator=FilterOperator.EQ,
            )
        )

    filters = MetadataFilters(filters=filters_list) if filters_list else None

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=DEFAULT_TOP_K,
        filters=filters,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ],
    )
