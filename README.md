Project: AI PDF RAG Assistant

Problem: Users need to quickly extract information from lengthy documents.

Solution: A RAG application that retrieves relevant document content before generating an answer.

Architecture:

PDF
 ↓
PyPDF
 ↓
Embeddings
 ↓
FAISS
 ↓
Similarity Search
 ↓
Gemini
 ↓
Answer

Tech Stack:

Python
LangChain
Gemini
FAISS
Streamlit

Key Features:

PDF question answering
Semantic retrieval
RAG
Source references
Hallucination control
Interactive UI
