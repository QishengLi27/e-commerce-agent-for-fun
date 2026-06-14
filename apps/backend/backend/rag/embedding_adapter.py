"""
LlamaIndex-compatible embedding adapter using LangChain's OpenAIEmbeddings.

Bypasses llama-index-embeddings-openai model name validation, which rejects
non-OpenAI model names like Zhipu's 'embedding-2'.
"""

from typing import Any

from langchain_openai import OpenAIEmbeddings
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr


class LangChainEmbeddingAdapter(BaseEmbedding):
    """Wraps a LangChain embeddings model for use with LlamaIndex pipelines.

    Usage:
        from backend.rag.embedding_adapter import LangChainEmbeddingAdapter
        from langchain_openai import OpenAIEmbeddings

        lc_embed = OpenAIEmbeddings(model="embedding-2", ...)
        embed_model = LangChainEmbeddingAdapter(lc_embed)
    """

    _lc: OpenAIEmbeddings = PrivateAttr()

    def __init__(self, lc_embeddings: OpenAIEmbeddings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lc = lc_embeddings

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._lc.embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._lc.embed_query(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._lc.embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._lc.embed_query(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._lc.embed_query(text)
