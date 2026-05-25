# RAG System

A complete Retrieval-Augmented Generation (RAG) application built from scratch on a personal laptop, with the goal of understanding every line of code rather than copy-pasting from tutorials.

Upload a PDF, ask questions, get grounded answers with citations from the document.

## What This Project Demonstrates

- PDF text extraction and intelligent chunking with overlap
- Local semantic embeddings using `sentence-transformers` (no API calls needed)
- Persistent vector storage with ChromaDB
- Cosine similarity-based retrieval
- LLM generation with grounded, hallucination-resistant prompting via Groq

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (Llama-3.1-8b-instant) — free tier |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) — local, CPU-only, 384-dim |
| Vector DB | ChromaDB (local, persistent) |
| PDF parsing | `pypdf` |
| Secrets | `python-dotenv` |
| Python | 3.11 |

## Architecture

PDF → Chunk (1000 chars, 150 overlap) → Embed (384-dim) → ChromaDB
│
User question → Embed (same model) → Retrieve top-K 
│
System prompt + chunks + question → Groq LLM → Grounded answer

## Project Structure
rag-learning/
├── test_groq.py        # Verify Groq API connection
├── chunk_pdf.py        # PDF text extraction + chunking
├── embed_chunks.py     # Generate embeddings + sanity-check semantics
├── index_pdf.py        # Persist embeddings to ChromaDB (one-time)
├── query_db.py         # Retrieve top-K chunks for a question
├── inspect_db.py       # Inspect what's actually stored in ChromaDB
├── rag_app.py          # Full RAG CLI application
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

## Where This Could Go Next (Production-Grade)

- Evaluation framework with RAGAS (faithfulness, answer relevance, context precision/recall)
- Hybrid search: BM25 keyword + vector similarity
- Cross-encoder reranker on top-20 retrievals
- Source citations with page numbers
- Multi-document support with metadata filtering
- Web UI with Streamlit
- Conversation memory for follow-up questions

## License

MIT — feel free to learn from this, fork it, break it.
