"""
retriever.py

Loads the Chroma vector database and creates
a retriever for semantic search.
"""
import os
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.abspath("./hf_cache")
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "vectorstore/chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedding_model():
    """
    Load the embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vectorstore():
    """
    Load the existing Chroma vector database.
    """

    embeddings = get_embedding_model()

    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vector_db


def get_retriever(
    vector_db=None,
    k: int = 5
):
    """
    Create a retriever from a Chroma database.

    Args:
        vector_db: Existing Chroma instance.
        k: Number of chunks to retrieve.

    Returns:
        LangChain Retriever
    """

    if vector_db is None:
        vector_db = load_vectorstore()

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    return retriever


def search_documents(
    query: str,
    k: int = 5
):
    """
    Search the vector database directly.

    Returns:
        List of matching Documents
    """

    vector_db = load_vectorstore()

    results = vector_db.similarity_search(
        query,
        k=k
    )

    return results


def print_search_results(results):
    """
    Print retrieved chunks for debugging.
    """

    print("=" * 60)
    print("Retrieved Documents")
    print("=" * 60)

    for i, doc in enumerate(results):

        print(f"\nResult {i+1}")

        print("\nMetadata:")
        print(doc.metadata)

        print("\nContent Preview:")

        print(doc.page_content[:500])

        print("-" * 60)


if __name__ == "__main__":

    query = "What are the major financial risks?"

    docs = search_documents(query)

    print_search_results(docs)