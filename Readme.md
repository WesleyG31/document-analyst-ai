# AI Document Analyst — Streamlit + RAG App + LangChain + GPT functions + HuggingFace Spaces for PDF Question Answering

---

## Overview

**AI Document Analyst** is an interactive web application that uses a **Retrieval-Augmented Generation (RAG)** pipeline to analyze PDF documents and answer user questions using large language models (LLMs).

This project is fully containerized with Docker and deployable on **Hugging Face Spaces**. It supports uploading new PDFs, persistent vector database creation with **ChromaDB**, and seamless LLM integration through **OpenRouter**.

---

## Find more information in design doc

[Design Doc](https://github.com/WesleyG31/document-analyst-ai/blob/main/design_doc.md)

- Assumptions and scope of work.
- Step-by-step implementation details.
- Discussion on failure cases and iterative improvements.
- Experiments, decision-making process, and learning points.

```
├── desing_doc.md                   # Provide a detailed design doc 
```

---

# 📽️ Demo

📽️ [Demo - YouTube](https://youtu.be/BQe3YDy7w4Q)

🧪 Try it live on Hugging Face:  

**[👉 AI Document Analyst (Streamlit App)](https://huggingface.co/spaces/WesleyGonzales/document-analyst-ai)**

---

## 🚀 Features

- ✅ Upload and preview PDF documents with real-time page rendering
- ✅ Convert PDF content to Markdown using `docling` and `pdf2image`
- ✅ Intelligent text chunking using `LangChain`'s `MarkdownHeaderTextSplitter`
- ✅ Vector storage and retrieval using `ChromaDB`
- ✅ Embeddings powered by HuggingFace model: `BAAI/bge-small-en`
- ✅ LLM response generation using OpenRouter API
- ✅ Persistent storage: previously uploaded documents are remembered
- ✅ Interactive UI built with Streamlit

---

## 🧠 Technologies Used

| Component                    | Tech Stack                                   |
|------------------------------|----------------------------------------------|
| Frontend UI                  | Streamlit                                    |
| Backend LLM                  | OpenRouter                                   |
| Embeddings                   | `BAAI/bge-small-en` via HuggingFace          |
| Vector Store                 | `ChromaDB` with MMR search                   |
| PDF Conversion               | `pdf2image` + `docling`                      |
| Deployment                   | Hugging Face Spaces + Docker                 |


---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit app
├── rag/                    # RAG pipeline logic (RAG_MODEL)
├── logs/                   # Logs - tries
├── src/                    # Logger + custom exceptions
├── tmp/vector_store/       # Persisted Chroma vector DBs 
├── Dockerfile              # Docker setup for HF Spaces
├── requirements.txt        # Python dependencies
├── README.md               # You're here!
```

---

## 🚀 How to Run (Local PC / Streamlit Local / Streamlit Cloud)

### Local PC

1. Clone the repo
```bash
git clone https://github.com/WesleyG31/document-analyst-ai
cd document-analyst-ai
```

2. (Optional) Create a virtual environment with Anaconda
```bash
conda create -n document-analyst python=3.10
conda activate document-analyst
```

3. Install dependencies
```bash
pip install -e .
```

4. (Optional) Install Cuda
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

5. Make sure you have Poppler installed
```bash
https://github.com/oschwartz10612/poppler-windows/releases
```

6. Add your OpenRouter API key
```bash
Create an file called .env inside "rag/"
OPENROUTER_API_KEY=your-key-here
```

7. Run the python file locally
```bash
streamlit run application.py
```

###  HuggingFace Spaces

1. No installation required. Just open the link and upload a PDF.
```bash
https://huggingface.co/spaces/WesleyGonzales/document-analyst-ai
PD: It's the same if you run it locally
```

---

## 🔒 API & Credentials
This app uses OpenRouter to access LLMs. Create a free account and get your API key.

In production, store the API key securely via Hugging Face Secrets or .env.

---

## 📄 Sample Outputs

- ✅ Answers about your PDF document.

---

## How It Works

1. User uploads a PDF

2. App:

    Converts PDF → Markdown

    Splits text into hierarchical chunks

    Embeds chunks and stores in ChromaDB

3. User asks a question

4. App retrieves relevant chunks using MMR search

5. Prompts an LLM with the retrieved context and user's question

6. Returns and streams the answer in the UI

---

## 💼 Why This Project Matters

This project demonstrates:

- Solid architecture with LangChain, vector stores, and custom RAG

- Practical LLM integration with real-world file formats (PDFs)

- Full deployment with Docker & Hugging Face Spaces

- Thoughtful error handling, logging, and modular code structure


---

## 👨‍💻 Author

**Wesley Gonzales**  
AI & ML Engineer  
📫 wes.gb31@gmail.com  
🔗 [https://www.linkedin.com/in/wesleygb/](https://www.linkedin.com/in/wesleygb/)  
🤖 [My Github](https://github.com/WesleyG31)
---

## 🪪 License

This project is licensed under the MIT License.
