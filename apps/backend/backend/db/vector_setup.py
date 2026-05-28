"""
Set up the vector database using PostgreSQL + pgvector.

Run this to (re)build the policy embeddings:
    python -m backend.db.vector_setup

Requires:
    - Docker container running: pgvector/pgvector
    - psycopg2-binary installed
"""

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings

CONNECTION_STRING = settings.database_url


def setup_vector_db():
    # Load the text file
    loader = TextLoader("data/store_policies.txt")
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    # chunk_size=64: Zhipu embedding-2 API limits batch size to 64
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        chunk_size=64,
    )

    # Create pgvector collection (replaces Chroma)
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="store_policies",
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
        distance_strategy="cosine",  # same as Chroma
    )

    print(f"Vector database setup complete. Stored {len(chunks)} chunks in pgvector.")
    return vectorstore


if __name__ == "__main__":
    setup_vector_db()
