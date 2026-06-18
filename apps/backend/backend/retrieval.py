"""
Hybrid Retrieval Module for RAG (pgvector edition)

Combines:
- Dense retrieval: pgvector (PostgreSQL) vector search
- Sparse retrieval: BM25 keyword search
- Fusion: Reciprocal Rank Fusion (RRF)
- Re-ranking: LLM-based relevance scoring
"""

import os
import pickle
import re

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from backend.config import settings
from backend.resilience import (
    CircuitBreaker,
    make_retry_decorator,
)
from backend.resilience import (
    logger as resilience_logger,
)

PG_CONNECTION = settings.database_url


class HybridPolicyRetriever:
    """
    Hybrid retriever that fuses pgvector (dense) and BM25 (sparse) results,
    with optional LLM-based re-ranking.
    """

    def __init__(
        self,
        text_file: str = "data/store_policies.txt",
        bm25_cache: str = "data/bm25_index.pkl",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.text_file = text_file
        self.bm25_cache = bm25_cache
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )

        # LLM for re-ranking
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            temperature=0.0,
            max_retries=2,
            timeout=15,
        )

        # Load pgvector dense store
        self.vectorstore = PGVector(
            connection_string=PG_CONNECTION,
            embedding_function=self.embeddings,
            collection_name="store_policies",
            distance_strategy="cosine",
        )

        # Build or load BM25 index
        self.chunks: list[Document] = []
        self.bm25 = None
        self._load_or_build_bm25()

        self.llm_circuit = CircuitBreaker("rerank-llm", failure_threshold=3, recovery_timeout=60.0)
        self.embedding_circuit = CircuitBreaker(
            "embeddings", failure_threshold=3, recovery_timeout=60.0
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for BM25."""
        return re.findall(r"\b\w+\b", text.lower())

    def _load_or_build_bm25(self):
        """Load BM25 from cache or build from source text."""
        if os.path.exists(self.bm25_cache):
            try:
                with open(self.bm25_cache, "rb") as f:
                    cache = pickle.load(f)
                self.chunks = cache["chunks"]
                self.bm25 = cache["bm25"]
                print(f"[retrieval] Loaded BM25 index from cache ({len(self.chunks)} chunks)")
                return
            except Exception as e:
                print(f"[retrieval] BM25 cache load failed: {e}. Rebuilding...")

        # Build from scratch
        loader = TextLoader(self.text_file)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.chunks = splitter.split_documents(docs)

        tokenized = [self._tokenize(c.page_content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        os.makedirs(os.path.dirname(self.bm25_cache) or ".", exist_ok=True)
        with open(self.bm25_cache, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
        print(f"[retrieval] Built and cached BM25 index ({len(self.chunks)} chunks)")

    def _dense_retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """pgvector search. Returns (doc, score) where higher is better."""

        def _search():
            return self.vectorstore.similarity_search_with_score(query, k=k)

        try:
            docs_and_scores = self.embedding_circuit.call(
                _search,
                lambda *a, **kw: [],
            )
            # pgvector COSINE distance: lower = better, same as Chroma
            # Invert so higher = better for RRF ranking
            return [(doc, -score) for doc, score in docs_and_scores]
        except Exception as e:
            resilience_logger.warning(f"[dense_retrieve] Failed: {e}")
            return []

    def _sparse_retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """BM25 keyword search. Returns (doc, score) where higher is better."""
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices]

    @staticmethod
    def _rrf_fuse(
        list_a: list[tuple[Document, float]],
        list_b: list[tuple[Document, float]],
        k: int = 60,
    ) -> list[tuple[Document, float]]:
        """
        Reciprocal Rank Fusion of two ranked lists.
        score(doc) = sum(1 / (k + rank)) for each list
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (doc, _) in enumerate(list_a, start=1):
            key = doc.page_content
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)

        for rank, (doc, _) in enumerate(list_b, start=1):
            key = doc.page_content
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[key], score) for key, score in sorted_docs]

    @make_retry_decorator(max_attempts=2)
    def _batch_rerank_call(self, prompt: str) -> str:
        """Single LLM call to score all documents at once."""
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _llm_rerank(
        self,
        query: str,
        docs: list[tuple[Document, float]],
        top_n: int = 3,
    ) -> list[tuple[Document, float]]:
        """
        Batch LLM re-rank: scores all documents in a single LLM call.
        Returns re-ranked (doc, score) list.
        """
        if not docs:
            return []

        doc_blocks = []
        for i, (doc, _) in enumerate(docs, start=1):
            doc_blocks.append(f"[{i}] {doc.page_content[:400]}")

        from backend.prompts import get_prompt as _get_prompt

        docs_text = "\n\n".join(doc_blocks)
        output = _get_prompt("rerank").render(query=query, documents=docs_text)
        prompt = output.text

        try:
            raw = self.llm_circuit.call(
                self._batch_rerank_call,
                lambda *a, **kw: "",
                prompt,
            )
            scores = self._parse_batch_scores(raw, len(docs))
        except Exception as e:
            resilience_logger.warning(f"[llm_rerank] Batch rerank failed: {e}")
            scores = {}

        scored = []
        for i, (doc, _) in enumerate(docs, start=1):
            score = float(scores.get(i, 5))
            score = max(1.0, min(10.0, score))
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @staticmethod
    def _parse_batch_scores(raw: str, num_docs: int) -> dict:
        """Parse scores from batch rerank response.

        Handles multiple formats the LLM might produce:
          - Strict:    [1] 8  [2] 5  [3] 9
          - With text: [1] Shipping and Delivery: 3  [2] Return Policy: 9
          - Multi-line with colons anywhere.
        """
        scores = {}
        # Match [N] followed by any text, then a colon + score: "[1] Shipping: 3"
        for match in re.finditer(r"\[(\d+)\].*?:\s*(\d+)", raw):
            idx = int(match.group(1))
            score = int(match.group(2))
            if 1 <= idx <= num_docs:
                scores[idx] = score

        # Fallback: match [N] immediately followed by score: "[1] 8"
        if not scores:
            for match in re.finditer(r"\[(\d+)\]\s*(\d+)", raw):
                idx = int(match.group(1))
                score = int(match.group(2))
                if 1 <= idx <= num_docs:
                    scores[idx] = score

        return scores

    def retrieve(
        self,
        query: str,
        k: int = 5,
        rerank: bool = True,
    ) -> list[tuple[Document, float]]:
        """
        Full pipeline: dense + sparse -> RRF fusion -> optional LLM re-rank.
        """
        dense = self._dense_retrieve(query, k=k)
        sparse = self._sparse_retrieve(query, k=k)
        fused = self._rrf_fuse(dense, sparse)

        if rerank:
            fused = self._llm_rerank(query, fused, top_n=k)

        return fused


# Singleton instance for easy importing
_retriever_instance = None


def get_policy_retriever() -> HybridPolicyRetriever:
    """Get or create the shared retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridPolicyRetriever()
    return _retriever_instance
