"""
chain.py

Modern LCEL RAG Chain using:
- Chroma Retriever
- PromptTemplate
- Ollama
- StrOutputParser
"""

import os

# Hugging Face cache
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.abspath("./hf_cache")

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama

# Works with both:
# python -m rag.chain
# streamlit run app.py
try:
    from rag.prompts import DUE_DILIGENCE_PROMPT
    from rag.retriever import get_retriever
except ImportError:
    from prompts import DUE_DILIGENCE_PROMPT
    from retriever import get_retriever


# ------------------------------------
# LLM
# ------------------------------------
def get_llm():
    """
    Returns the Ollama LLM.
    """

    return ChatOllama(
        model="qwen2:7b",
        temperature=0,
        num_ctx=8192,
        num_predict=1024
    )


# ------------------------------------
# Helper
# ------------------------------------
def format_docs(docs):
    """
    Combine retrieved documents into one string.
    """

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# ------------------------------------
# Build Chain
# ------------------------------------
def get_qa_chain():

    retriever = get_retriever()

    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | DUE_DILIGENCE_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# ------------------------------------
# Ask Question
# ------------------------------------
def ask_question(chain, question):

    # Get retriever separately
    retriever = get_retriever()

    docs = retriever.invoke(question)

    print("\n================ RETRIEVED DOCUMENTS ================\n")

    for i, doc in enumerate(docs):
        print(f"Document {i+1}")
        print(doc.metadata)
        print(doc.page_content[:600])
        print("\n" + "-"*70 + "\n")

    answer = chain.invoke(question)

    return answer


# ------------------------------------
# Testing
# ------------------------------------
if __name__ == "__main__":

    chain = get_qa_chain()

    question = "What are the company's major financial risks?"

    print("=" * 70)

    print("QUESTION\n")

    print(question)

    print("\n")

    answer = ask_question(
        chain,
        question
    )

    print("=" * 70)

    print("ANSWER\n")

    print(answer)