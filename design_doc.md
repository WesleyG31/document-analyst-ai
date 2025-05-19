#  AI agent - LangChain + GPT functions + HuggingFace Spaces + Streamlit

## Overview

AI Document Analyst is a Streamlit-based web application that enables users to upload PDF documents and ask questions about their content using a Retrieval-Augmented Generation (RAG) system. The pipeline integrates:

- LangChain for orchestration

- HuggingFace embeddings

- ChromaDB for vector search

- LLMs served via OpenRouter

- Streamlit for UI

- Docker for reproducibility and deployment

The system is designed to support real-time document QA over user-provided PDFs, with persistent vector storage and modular architecture.

--- 

## Assumptions & Scope

### What I included:

- PDF to Markdown conversion via docling

- Visual preview of PDF pages (via pdf2image and PIL)

- Hierarchical chunking using MarkdownHeaderTextSplitter

- Embedding with BAAI/bge-small-en

- ChromaDB vector storage (MMR retrieval)

- OpenRouter-powered LLM querying

- Full Dockerized deployment to Hugging Face Spaces

### What I excluded (due to time constraints):

- Multi-PDF cross-document search

- Citation display with source page numbers

### Assumptions:

- Expected usage is single-user, single-PDF per session

- Embedding model remains constant per deployment

- Hugging Face Spaces will provide CPU-only execution

---

## Architecture & Pipeline

### System Architecture
```plaintext
PDF Upload
   ↓
PDF Preview Generator (pdf2image)
   ↓
DocumentConverter (docling)
   ↓
LangChain Markdown Splitter
   ↓
Embeddings via HF (BAAI/bge-small-en)
   ↓
ChromaDB (persisted)
   ↓
MMR Retriever
   ↓
Prompt Template + OpenRouter LLM
   ↓
Streamed Answer in Streamlit UI
```

---

### Pipeline

1. PDF Upload & Preview

    PDF is uploaded via Streamlit

    Preview images are generated using pdf2image

2. Text Conversion

    Document is converted to Markdown via docling

3. Chunking

    Markdown is split into nested chunks using LangChain's MarkdownHeaderTextSplitter

4. Embedding + Vector Storage

    Chunks are embedded using HuggingFaceEmbeddings

    Stored in Chroma with persist_directory

5. Question Answering (RAG)

    ChromaDB performs Max Marginal Relevance (MMR) retrieval

    LangChain formats a system prompt

    OpenRouter LLM generates the answer

    Output is streamed back to the UI

![workflow](https://github.com/user-attachments/assets/80a3a337-3f27-4373-b309-03f3404bf701)


### Prompt Design

```
You are an assistant for {assistant}. Use the retrieved context to answer questions.
If you don't know the answer, say so.
Always answer in a professional tone.

Question: {question}
Context: {context}
Answer:
```

- Designed for domain flexibility (e.g., "Finance Assistant", "HR Analyst")

- Ensures grounded, context-based generation

### 🧠 Prompt Generalization & Customization

In this implementation, I opted for a generic prompt intentionally, to ensure that the system remains adaptable across a wide variety of PDF document types — from technical manuals to academic papers or corporate reports.

However, this prompt can be easily customized through prompt engineering to optimize responses for specific domains (e.g., finance, healthcare, legal). By tailoring the prompt instructions or adding few-shot examples, we can significantly improve the relevance and tone of the generated answers for targeted use cases.

---

### Design Decisions & Alternatives

    | Choise                  | Why? / Why not?                                                            |
    |-------------------------|----------------------------------------------------------------------------|
    | docling for conversion  | Maintains markdown structure, better for header-based splitting            |
    | MarkdownHeaderSplitter  | Retains hierarchy (titles, subtitles) - improves context relevance         |
    | BAAI/bge-small-en       | Lightweight, performant embedding model with good semantic understanding   |
    | ChromaDB                | Persistent, in-memory, fast, and open-source.                              |
    | MMR Search              | Avoids redundancy in retrieved context                                     |
    | OpenRouter API          | Multi-model LLM access with cost control and OpenAI compatibility          |
    | Streamlit               | Rapid UI deployment for demo and iteration                                 |
    | Docker                  | Reproducible environment for Hugging Face Spaces                           |


---

## Iterations & Learnings

### LOG 

Throughout the project, I explored several approaches, learned from failures, and refined the system through multiple iterations. Here's a breakdown of my process:

Every step was saved in logs 

```
├── logs/                   # Logs - attemps
```

![logs](https://github.com/user-attachments/assets/c84fd70a-6609-41ab-a782-3352bc6965c7)


### 🔹 Initial Setup 

**Approach:**  

- Started with plain PDF-to-text 

**Why I tried this:**  

- (PyMuPDF) — lacked structure

**Result:**  
- Switched to docling to retain headers, which improved chunking

---

### 🔹 Embedding & Chroma

**Approach:**  
- Needed a fast  and robust embedding 

**Why I tried this:**  
- Tried sentence-transformers/all-MiniLM-L6-v2 — decent but noisy.

- Also "BAAI/bge-large-en" also could be used a little bit big.

- Thi intfloat/e5-mistral-7b-instruct really big.

**Result:**  
- Found BAAI/bge-small-en more robust for document QA

**Learning:**  
- Realized importance of consistent persist_directory


![first_try](https://github.com/user-attachments/assets/7395c575-0d40-46a8-827b-403820572391)

---

### 🔹 Streamlit Integration

**Approach:**  
- Added PDF preview

**Why I tried this:**  
- Added PDF preview with pdf2image for usability

**Result:**  
- Real-time streaming of LLM responses improved UX
- Easy testing

---

### 🔹 Docker Deployment

**Approach:**  
Docker container to upload in HuggingFace Spaces

**Result:**  
- Needed Poppler + Pillow system deps for pdf2image
- Shifted to python:3.10-slim + apt install for poppler-utils

---


### 🔹 Video of attempts

- Click on the videos to watch

[![attempts](https://img.youtube.com/vi/Oa1ajCfaZE0/hqdefault.jpg)](https://youtu.be/Oa1ajCfaZE0)


---

### 🎯 General Learnings

- **Iterative development matters.** Every step taught me something valuable.

---

## Future Improvements

✚ Multilingual PDFs with dynamic embedding model selection

✚ Per-user workspaces with login/session support

✚ Citation tracing (highlight chunks that contributed to the answer)

---

## Summary

I approached this challenge as a chance to show:

- That I can think clearly and structure a system

- That I can balance creativity with pragmatism

- That I enjoy building and iterating under constraints

- I’m proud of what I built — not because it’s perfect, but because it’s practical, modular, and real.

- Thanks for this opportunity. I hope to continue building with you.

This project demonstrates:

- Solid understanding of RAG architecture

- Modular thinking using LangChain components

- Real-world deployment via Docker & Hugging Face

- Clear UX priorities via Streamlit + PDF previews

It reflects my ability to:

- Take full ownership from data to deployment

- Balance simplicity with power (Chroma + LLMs)

- Communicate design through code, docs, and visuals


- Wesley




