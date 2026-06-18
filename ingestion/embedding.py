"""
embedding.py

Creates and loads a Chroma vector database
using HuggingFace embeddings.
"""

import os
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.abspath("./hf_cache")
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Directory where ChromaDB will be stored
VECTOR_DB_PATH = "vectorstore/chroma_db"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedding_model():
    """
    Load the HuggingFace embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


def create_vectorstore(chunks: List[Document]):
    """
    Create a new Chroma vector database.

    Args:
        chunks: List of chunked LangChain Documents

    Returns:
        Chroma vector database
    """

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    embeddings = get_embedding_model()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    vector_db.persist()

    return vector_db


def load_vectorstore():
    """
    Load an existing Chroma vector database.
    """

    embeddings = get_embedding_model()

    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vector_db


def delete_vectorstore():
    """
    Delete the existing vector database.
    Useful when rebuilding from scratch.
    """

    import shutil

    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("Vector database deleted.")

    else:
        print("No vector database found.")


if __name__ == "__main__":

    from pdf_loader import load_pdf
    from chunking import split_docs

    docs = load_pdf("uploads/sample.pdf")

    chunks = split_docs(docs)

    db = create_vectorstore(chunks)

    print("Vector database created successfully!")

    print(f"Total Chunks Stored: {db._collection.count()}")