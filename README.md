# 📊 AI Due Diligence Copilot

An Enterprise AI-powered Due Diligence Assistant that analyzes company documents using Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and Vector Databases.

Built using **LangChain LCEL**, **Ollama**, **ChromaDB**, **HuggingFace Embeddings**, **FastAPI**, and **Streamlit**.

---

## 🚀 Features

- 📄 Upload one or multiple PDF documents
- 🧠 AI-powered document understanding
- 🔍 Semantic search using ChromaDB
- 🤖 Local LLM inference with Ollama (Qwen2)
- 📚 Retrieval-Augmented Generation (RAG)
- 💬 Ask questions about uploaded documents
- 📊 Financial & business risk analysis
- 📑 Generate professional Due Diligence reports
- ⚡ FastAPI backend
- 🎨 Interactive Streamlit frontend

---

# 🏗️ Project Architecture

```
                PDF Documents
                      │
                      ▼
              PDF Loader (PyPDF)
                      │
                      ▼
              Document Chunking
                      │
                      ▼
        HuggingFace Embeddings
                      │
                      ▼
                Chroma Vector DB
                      │
                      ▼
                 Semantic Retriever
                      │
                      ▼
            LangChain LCEL Pipeline
                      │
                      ▼
             Ollama (Qwen2:7B LLM)
                      │
                      ▼
            AI Due Diligence Response
                      │
                      ▼
           Professional Report Generator
```

---

# 📂 Project Structure

```
AI-Due-Diligence-Copilot/

│
├── app.py
├── api.py
├── requirements.txt
├── README.md
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── chunking.py
│   └── embedding.py
│
├── rag/
│   ├── retriever.py
│   ├── chain.py
│   └── prompts.py
│
├── reports/
│   ├── generator.py
│   └── generated_reports/
│
├── vectorstore/
│   └── chroma_db/
│
├── uploads/
│
└── .env
```

---

# ⚙️ Technologies Used

- Python
- LangChain LCEL
- Ollama
- Qwen2:7B
- ChromaDB
- HuggingFace Embeddings
- Streamlit
- FastAPI
- PyPDF
- Sentence Transformers

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/VENKATESHMISRA/AI_Due_Diligence_CoPilot

cd AI-Due-Diligence-Copilot
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Install Ollama

Download Ollama

https://ollama.com/

Pull the model

```bash
ollama pull qwen2:7b
```

Verify

```bash
ollama list
```

---

# ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---

# ▶️ Run the API

```bash
uvicorn api:app --reload
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# 📄 How It Works

### Step 1

Upload company PDF documents.

↓

### Step 2

Documents are split into semantic chunks.

↓

### Step 3

Embeddings are generated using HuggingFace.

↓

### Step 4

Chunks are stored inside ChromaDB.

↓

### Step 5

User asks a question.

↓

### Step 6

Retriever fetches the most relevant chunks.

↓

### Step 7

Ollama (Qwen2) generates an answer using the retrieved context.

↓

### Step 8

Professional Due Diligence report is generated.

---

# 💬 Example Questions

- What are the company's major financial risks?
- Summarize the business model.
- What legal risks does the company face?
- What are the major revenue sources?
- Does the company mention acquisitions?
- What operational risks exist?
- What opportunities are identified?
- What is the overall risk level?

---

# 📊 Example Output

```
Question

What are the company's major financial risks?

Answer

• Advertising revenue dependence

• Strategic acquisition risks

• Financing and debt obligations

• Investment value fluctuations

• Operational disruptions

• International business risks
```

---

# 🛠️ API Endpoints

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | / | Home |
| POST | /upload | Upload PDFs |
| POST | /ask | Ask Questions |
| GET | /history | Chat History |
| GET | /report | Download Report |
| DELETE | /clear | Clear Session |
| GET | /health | API Health |

---

# 🎯 Future Improvements

- Multi-document comparison
- Financial KPI extraction
- Automatic risk scoring
- Executive summary generation
- Interactive dashboards
- PDF report generation
- Citation support
- Company comparison
- Multi-user authentication
- Cloud deployment

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create your feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 👨‍💻 Author

**Venkatesh Misra**

AI • Machine Learning • RAG • LLMs • LangChain • FastAPI • Streamlit

GitHub: https://github.com/VENKATESHMISRA

LinkedIn: https://www.linkedin.com/in/venkatesh-misra-233611378/

---

⭐ If you found this project useful, consider giving it a star!
