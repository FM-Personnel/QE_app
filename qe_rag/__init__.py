"""qe_rag — pipeline d'ingestion RAG autonome (extract -> clean -> chunk -> embed -> upsert).

Module partagé destiné à remplacer, à terme, le chemin d'upload de `app.py` et les
notebooks Colab d'ingestion. Backend d'embedding pluggable (local / endpoint / cache).

Rien ici n'importe `streamlit` ni `app.py`.
"""
from .report import IngestionReport, StepResult
from .pipeline import ingest_document

__all__ = ["ingest_document", "IngestionReport", "StepResult"]

__version__ = "0.1.0"
