from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def setup_vector_db():
    # Load the text file
    loader = TextLoader('store_policies.txt')
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    # embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(
	    model="embedding-2",
	    openai_api_key = "51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
	    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/"
	)

    # Create Chroma vector store
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

    print("Vector database setup complete.")
    return vectorstore

if __name__ == "__main__":
    setup_vector_db()