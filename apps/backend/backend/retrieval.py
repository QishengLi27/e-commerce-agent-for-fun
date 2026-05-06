"""
Hybrid Retrieval Module for RAG (pgvector edition)

Combines:
- Dense retrieval: pgvector (PostgreSQL) vector search
- Sparse retrieval: BM25 keyword search
- Fusion: Reciprocal Rank Fusion (RRF)
- Re-ranking: LLM-based relevance scoring
"""

import os
import re
import pickle
from typing import List, Tuple

from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from backend.resilience import (
    CircuitBreaker,
    Fallbacks,
    make_retry_decorator,
    logger as resilience_logger,
)

PG_CONNECTION = "postgresql+psycopg2://postgres:postgres@localhost:5432/ecommerce"


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
            model="embedding-2",
            openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        )

        # LLM for re-ranking
        self.llm = ChatOpenAI(
            model="glm-4-flash",
            openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
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
        self.chunks: List[Document] = []
        self.bm25 = None
        self._load_or_build_bm25()

        self.llm_circuit = CircuitBreaker("rerank-llm", failure_threshold=3, recovery_timeout=60.0)
        self.embedding_circuit = CircuitBreaker("embeddings", failure_threshold=3, recovery_timeout=60.0)

    def _tokenize(self, text: str) -> List[str]:
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

    def _dense_retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
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

    def _sparse_retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """BM25 keyword search. Returns (doc, score) where higher is better."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices]

    @staticmethod
    def _rrf_fuse(
        list_a: List[Tuple[Document, float]],
        list_b: List[Tuple[Document, float]],
        k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion of two ranked lists.
        score(doc) = sum(1 / (k + rank)) for each list
        """
        doc_scores = {}
        doc_map = {}

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
    def _score_single_doc(self, prompt: str) -> int:
        """Score one document with retry logic."""
        response = self.llm.invoke(prompt)
        content = response.content.strip()
        match = re.search(r"\b(\d+)\b", content)
        return int(match.group(1)) if match else 5

    def _llm_rerank(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
        top_n: int = 3,
    ) -> List[Tuple[Document, float]]:
        """
        Use the LLM to score each document's relevance to the query.
        Returns re-ranked (doc, score) list.
        """
        scored = []
        for doc, _ in docs:
            prompt = (
                "Rate how relevant the following document is to answering the user's question. "
                "Respond with only a number from 1 to 10, where 10 means perfectly relevant.\n\n"
                f"User question: {query}\n\n"
                f"Document: {doc.page_content[:400]}\n\n"
                "Relevance score (1-10):"
            )
            try:
                score = self.llm_circuit.call(
                    self._score_single_doc,
                    lambda *a, **kw: 5,
                    prompt,
                )
                score = max(1, min(10, score))
            except Exception as e:
                resilience_logger.warning(f"[llm_rerank] Failed for doc: {e}")
                score = 5
            scored.append((doc, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        rerank: bool = True,
    ) -> List[Tuple[Document, float]]:
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
