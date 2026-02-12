from __future__ import annotations

from pathlib import Path
import streamlit as st

from rag.core.config import get_settings
from rag.embeddings.embedder import Embedder
from rag.rag_pipeline.generator import OllamaGenerator
from rag.rag_pipeline.indexer import Indexer
from rag.rag_pipeline.prompt import build_prompt
from rag.rag_pipeline.retriever import Retriever
from rag.store.metadata_store import MetadataStore
from rag.store.vector_store import VectorStore


st.set_page_config(page_title="Personal RAG OCR", layout="wide")
settings = get_settings()

st.title("Personal Document RAG (OCR-First)")
index_tab, chat_tab = st.tabs(["Index", "Chat"])

with index_tab:
    st.subheader("Build / refresh index")
    path_str = st.text_input("Document folder", str(settings.docs_dir))
    status = st.empty()
    progress = st.progress(0)
    if st.button("Index"):
        idx = Indexer(settings)

        def on_progress(current: int, total: int, msg: str) -> None:
            progress.progress(current / max(total, 1))
            status.info(msg)

        summary = idx.index_path(Path(path_str), progress_cb=on_progress)
        idx.close()
        st.success(f"Done. Total={summary['total']} Reindexed={summary['reindexed']} Skipped={summary['skipped']}")

with chat_tab:
    st.subheader("Ask questions")
    question = st.text_area("Question")
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            embedder = Embedder(settings.embedding_model)
            metadata_store = MetadataStore(settings.sqlite_path)
            vector_store = VectorStore(settings.index_dir, dim=384)
            retriever = Retriever(embedder, vector_store, metadata_store)
            chunks = retriever.retrieve(question, settings.top_k)
            if not chunks:
                answer = "I don’t have enough information in the indexed documents.\n\nSources\n- None"
            else:
                prompt = build_prompt(question, chunks)
                generator = OllamaGenerator(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds)
                try:
                    answer = generator.generate(prompt)
                except RuntimeError as exc:
                    answer = f"Error: {exc}"
            st.markdown(answer)
            st.markdown("### Sources")
            seen = set()
            for chunk in chunks:
                key = (chunk.file_name, chunk.locator)
                if key in seen:
                    continue
                seen.add(key)
                st.write(f"- {chunk.file_name} ({chunk.locator})")
            metadata_store.close()
