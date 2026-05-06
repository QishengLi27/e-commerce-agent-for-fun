"""
Set up the vector database using PostgreSQL + pgvector.

Run this to (re)build the policy embeddings:
    python -m backend.db.vector_setup

Requires:
    - Docker container running: pgvector/pgvector
    - psycopg2-binary installed
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/ecommerce"


def setup_vector_db():
    # Load the text file
    loader = TextLoader("data/store_policies.txt")
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
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
