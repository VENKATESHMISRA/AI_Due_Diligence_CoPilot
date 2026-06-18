"""
pdf_loader.py

Loads PDF documents using LangChain's PyPDFLoader.
Returns a list of LangChain Document objects.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file.

    Args:
        file_path (str): Path to the PDF.

    Returns:
        List[Document]: One Document object per page.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(
            "Only PDF files are supported."
        )

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Load multiple PDF files.

    Args:
        file_paths (List[str])

    Returns:
        List[Document]
    """

    all_documents = []

    for pdf in file_paths:

        docs = load_pdf(pdf)

        all_documents.extend(docs)

    return all_documents


def print_document_summary(documents: List[Document]):
    """
    Print basic information about loaded documents.
    Useful for debugging.
    """

    print("=" * 50)
    print("Document Summary")
    print("=" * 50)

    print(f"Total Pages Loaded : {len(documents)}")

    for i, doc in enumerate(documents[:5]):

        print(f"\nPage {i + 1}")

        print("Metadata:")
        print(doc.metadata)

        print("\nPreview:")

        print(doc.page_content[:300])

        print("-" * 50)


if __name__ == "__main__":

    sample_pdf = "uploads/sample.pdf"

    docs = load_pdf(sample_pdf)

    print_document_summary(docs)