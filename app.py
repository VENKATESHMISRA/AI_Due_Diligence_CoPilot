"""
app.py

AI Due Diligence Copilot
Streamlit Frontend
"""

import os
import shutil

import streamlit as st

from ingestion.pdf_loader import load_pdf
from ingestion.chunking import split_docs
from ingestion.embedding import create_vectorstore

from rag.chain import get_qa_chain, ask_question

from reports.generator import generate_report

# ----------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="AI Due Diligence Copilot",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------
# Create Required Folders
# ----------------------------------------------------

os.makedirs("uploads", exist_ok=True)
os.makedirs("reports/generated_reports", exist_ok=True)
os.makedirs("vectorstore/chroma_db", exist_ok=True)

# ----------------------------------------------------
# Session State
# ----------------------------------------------------

if "chain" not in st.session_state:
    st.session_state.chain = None

if "history" not in st.session_state:
    st.session_state.history = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.title("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload Company PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "🚀 Process Documents",
        use_container_width=True
    )

    st.markdown("---")

    if st.session_state.documents_loaded:

        st.success("Documents Ready")

    else:

        st.warning("No Documents Loaded")

# ----------------------------------------------------
# Main Title
# ----------------------------------------------------

st.title("📊 AI Due Diligence Copilot")

st.markdown(
"""
Ask questions about:

- Financial Reports
- Annual Reports
- Pitch Decks
- Legal Documents
- Company Profiles
"""
)

# ----------------------------------------------------
# Process PDFs
# ----------------------------------------------------

if process_button:

    if not uploaded_files:

        st.error("Please upload at least one PDF.")

    else:

        with st.spinner("Saving uploaded files..."):

            all_docs = []

            for file in uploaded_files:

                save_path = os.path.join(
                    "uploads",
                    file.name
                )

                with open(save_path, "wb") as f:

                    f.write(file.getbuffer())

                docs = load_pdf(save_path)

                all_docs.extend(docs)

        st.success(
            f"Loaded {len(all_docs)} pages."
        )

        with st.spinner("Creating chunks..."):

            chunks = split_docs(all_docs)

        st.success(
            f"{len(chunks)} chunks created."
        )

        with st.spinner("Creating embeddings..."):

            create_vectorstore(chunks)

        st.success("Vector Database Created")

        with st.spinner("Loading AI..."):

            st.session_state.chain = get_qa_chain()

        st.session_state.documents_loaded = True

        st.success("AI Copilot Ready!")

# ----------------------------------------------------
# Ask Questions
# ----------------------------------------------------

st.markdown("---")

st.header("💬 Ask Questions")

question = st.text_input(
    "Enter your question"
)

ask = st.button(
    "Ask AI",
    use_container_width=True
)
# ----------------------------------------------------
# Generate Answer
# ----------------------------------------------------

if ask:

    if not st.session_state.documents_loaded:

        st.error(
            "Please upload and process documents first."
        )

    elif question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner("Analyzing documents..."):

            answer = ask_question(
                st.session_state.chain,
                question
            )

        st.session_state.history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        st.success("Analysis Complete!")

        st.markdown("## AI Response")

        st.write(answer)

# ----------------------------------------------------
# Chat History
# ----------------------------------------------------

if st.session_state.history:

    st.markdown("---")

    st.header("Conversation History")

    for chat in reversed(st.session_state.history):

        with st.expander(chat["question"]):

            st.write(chat["answer"])

# ----------------------------------------------------
# Report Generator
# ----------------------------------------------------

st.markdown("---")

st.header("Generate Due Diligence Report")

if st.button(
    "Generate Report",
    use_container_width=True
):

    if len(st.session_state.history) == 0:

        st.warning(
            "No analyses available."
        )

    else:

        report = ""

        for chat in st.session_state.history:

            report += (
                f"Question:\n"
                f"{chat['question']}\n\n"
            )

            report += (
                f"Answer:\n"
                f"{chat['answer']}\n\n"
            )

            report += (
                "=" * 80
            )

            report += "\n\n"

        report_path = generate_report(report)

        st.success("Report Generated Successfully!")

        with open(report_path, "rb") as file:

            st.download_button(
                "Download Report",
                data=file,
                file_name="Due_Diligence_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# ----------------------------------------------------
# Clear Chat
# ----------------------------------------------------

st.markdown("---")

if st.button(
    "Clear Conversation",
    use_container_width=True
):

    st.session_state.history = []

    st.success("Conversation Cleared!")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.markdown("---")

st.caption(
    "AI Due Diligence Copilot | "
    "Powered by LangChain • Ollama • ChromaDB • Streamlit"
)