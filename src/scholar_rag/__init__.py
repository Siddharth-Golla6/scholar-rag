"""ScholarRAG — an agentic Retrieval-Augmented Generation research assistant.

Pipeline: PDF/arXiv ingestion -> chunking -> local embeddings -> vector store
-> retrieval (+MMR) -> LLM answer with citations -> tool-using agent fallback
-> LLM-as-judge evaluation. Served via FastAPI + Streamlit.
"""

__version__ = "0.1.0"
