from __future__ import annotations

from rag.core.models import RetrievedChunk


def build_prompt(question: str, context_chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{c.file_name} {c.locator}]\n{c.text}" for c in context_chunks
    )
    return (
        "You are a grounded assistant. Answer only using provided context. "
        "If insufficient context, reply exactly: I don’t have enough information in the indexed documents. "
        "Include inline citations like [file p.2] and end with a Sources section listing each source once.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
    )
