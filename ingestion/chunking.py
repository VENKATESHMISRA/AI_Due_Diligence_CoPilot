"""
chunking.py

Splits LangChain Document objects into smaller chunks
for embedding and retrieval.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_docs(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    Args:
        documents: List of LangChain Documents
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked Documents
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks


def print_chunk_summary(chunks: List[Document]):
    """
    Print chunk statistics.
    Useful for debugging.
    """

    print("=" * 60)
    print("Chunk Summary")
    print("=" * 60)

    print(f"Total Chunks: {len(chunks)}")

    if len(chunks) == 0:
        return

    print("\nFirst Chunk Metadata:")
    print(chunks[0].metadata)

    print("\nFirst Chunk Preview:\n")
    print(chunks[0].page_content[:500])


if __name__ == "__main__":

    from pdf_loader import load_pdf

    docs = load_pdf("uploads/sample.pdf")

    chunks = split_docs(docs)

    print_chunk_summary(chunks)