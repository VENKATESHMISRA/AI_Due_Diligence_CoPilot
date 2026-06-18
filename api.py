"""
api.py

FastAPI Backend
AI Due Diligence Copilot
"""

import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingestion.pdf_loader import load_pdf
from ingestion.chunking import split_docs
from ingestion.embedding import create_vectorstore

from rag.chain import (
    get_qa_chain,
    ask_question
)

from reports.generator import generate_report

# --------------------------------------------------
# Create folders
# --------------------------------------------------

os.makedirs("uploads", exist_ok=True)
os.makedirs("reports/generated_reports", exist_ok=True)
os.makedirs("vectorstore/chroma_db", exist_ok=True)

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="AI Due Diligence Copilot API",
    version="2.0.0",
    description="Modern RAG API using LangChain + Ollama"
)

# --------------------------------------------------
# Global Variables
# --------------------------------------------------

chain = None
history = []

# --------------------------------------------------
# Request Models
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str

# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "AI Due Diligence Copilot API Running"
    }

# --------------------------------------------------
# Upload PDFs
# --------------------------------------------------

@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...)
):

    global chain

    if len(files) == 0:

        raise HTTPException(
            status_code=400,
            detail="No files uploaded."
        )

    all_docs = []

    for file in files:

        if not file.filename.endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF."
            )

        save_path = os.path.join(
            "uploads",
            file.filename
        )

        with open(save_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        docs = load_pdf(save_path)

        all_docs.extend(docs)

    chunks = split_docs(all_docs)

    create_vectorstore(chunks)

    chain = get_qa_chain()

    return {

        "status": "success",

        "pages_loaded": len(all_docs),

        "chunks_created": len(chunks),

        "message": "Documents processed successfully."
    }
# --------------------------------------------------
# Ask Question
# --------------------------------------------------

@app.post("/ask")
def ask(request: QuestionRequest):

    global chain
    global history

    if chain is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload and process documents first."
        )

    answer = ask_question(
        chain,
        request.question
    )

    history.append(
        {
            "question": request.question,
            "answer": answer
        }
    )

    return {

        "status": "success",

        "question": request.question,

        "answer": answer
    }


# --------------------------------------------------
# Chat History
# --------------------------------------------------

@app.get("/history")
def get_history():

    return {

        "total_questions": len(history),

        "history": history
    }


# --------------------------------------------------
# Generate Report
# --------------------------------------------------

@app.get("/report")
def download_report():

    if len(history) == 0:

        raise HTTPException(
            status_code=400,
            detail="No conversation available."
        )

    report = ""

    for item in history:

        report += (
            f"Question:\n"
            f"{item['question']}\n\n"
        )

        report += (
            f"Answer:\n"
            f"{item['answer']}\n\n"
        )

        report += "=" * 80

        report += "\n\n"

    report_path = generate_report(report)

    return FileResponse(

        path=report_path,

        filename="Due_Diligence_Report.docx",

        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


# --------------------------------------------------
# Clear Session
# --------------------------------------------------

@app.delete("/clear")
def clear():

    global chain
    global history

    chain = None
    history = []

    return {

        "status": "success",

        "message": "Conversation cleared."
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "ollama": "connected",

        "vector_database": "ready" if chain else "not loaded"
    }


# --------------------------------------------------
# API Information
# --------------------------------------------------

@app.get("/info")
def info():

    return {

        "application": "AI Due Diligence Copilot",

        "version": "2.0",

        "llm": "Ollama (Qwen2:7B)",

        "vector_database": "ChromaDB",

        "framework": "LangChain LCEL",

        "frontend": "Streamlit",

        "backend": "FastAPI"
    }