"""
app/services/rag_service.py  —  Retrieval-Augmented Generation pipeline.

Flow
────
1. Embed the user's question (via EmbeddingService)
2. Load session chunks + embeddings from the session store
3. Build a transient FAISS flat-L2 index (never stored globally)
4. Return the top-k most relevant chunks for Gemini to use
"""
import logging
import re
from typing import List

import numpy as np

log = logging.getLogger("jurisai.rag")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# Chunking profiles keyed by detected document type
CHUNK_PROFILES = {
    "legal":     {"chunk_size": 200, "overlap": 40},
    "technical": {"chunk_size": 350, "overlap": 50},
    "narrative": {"chunk_size": 500, "overlap": 30},
    "default":   {"chunk_size": 300, "overlap": 40},
}

_LEGAL_PAT = re.compile(
    r"\b(whereas|hereinafter|indemnif|liabilit|termination|arbitration"
    r"|jurisdiction|governing law|force majeure|warranty|covenant"
    r"|parties?|agreement|contract|clause|obligation|penalty"
    r"|breach|confidential|intellectual property|non-disclosure)\b", re.I
)
_TECH_PAT = re.compile(
    r"\b(specification|protocol|algorithm|implementation|interface"
    r"|architecture|database|server|bandwidth|latency|API|SDK|deployment)\b", re.I
)
_NARR_PAT = re.compile(
    r"\b(chapter|story|narrator|character|plot|scene|dialogue|novel|fiction)\b", re.I
)
_HEADER_RE = re.compile(
    r"(?=^(?:[A-Z][A-Z\s\-]{3,}|(?:ARTICLE|SECTION|CLAUSE)\s+\w+|\d+[\.\d]*\s+[A-Z]))",
    re.MULTILINE,
)


class RAGService:
    """
    Manages the chunking → embedding → retrieval pipeline.
    Session isolation is enforced via the injected session_store.
    """

    def __init__(self, embedding_service, session_store):
        self._embedder = embedding_service
        self._store = session_store

    # ── Document type detection ───────────────────────────────────────────────

    @staticmethod
    def detect_doc_type(text: str) -> str:
        sample = text[:3000]
        scores = {
            "legal":     len(_LEGAL_PAT.findall(sample)),
            "technical": len(_TECH_PAT.findall(sample)),
            "narrative": len(_NARR_PAT.findall(sample)),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] >= 3 else "default"

    # ── Adaptive chunking ─────────────────────────────────────────────────────

    def chunk_text(self, text: str, filename: str = "") -> List[str]:
        """Split text into overlapping word-level chunks based on document type."""
        doc_type = self.detect_doc_type(text)
        profile = CHUNK_PROFILES[doc_type]
        chunk_size = profile["chunk_size"]
        overlap = profile["overlap"]
        log.info("Chunking '%s': type=%s chunk=%d overlap=%d",
                 filename, doc_type, chunk_size, overlap)

        sections = _HEADER_RE.split(text)
        sections = [s.strip() for s in sections if s.strip()] or [text]

        result = []
        for sec in sections:
            words = sec.split()
            if not words:
                continue
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk = " ".join(words[start:end]).strip()
                if chunk:
                    result.append(chunk)
                start += max(1, chunk_size - overlap)
        return result

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, sid: str, filename: str, text: str) -> dict:
        """
        Chunk → embed → store in the session store.
        Returns a stats dict: {chunks, dim}.
        """
        chunks = self.chunk_text(text, filename=filename)
        log.info("'%s': %d chunks", filename, len(chunks))

        all_embs = []
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            emb = self._embedder.embed(batch, batch_size=batch_size)
            all_embs.append(emb)

        embeddings = (
            np.vstack(all_embs).astype("float32")
            if all_embs
            else np.zeros((0, 1), dtype="float32")
        )
        self._store.append_document(sid, filename, chunks, embeddings)
        return {
            "chunks": len(chunks),
            "dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else 0,
        }

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, sid: str, question: str, top_k: int = 5) -> List[str]:
        """
        Embed *question*, search session embeddings, return top-k chunks.
        All data access is scoped to *sid* — no global index is used.
        """
        if not FAISS_AVAILABLE:
            log.warning("faiss not installed — returning first chunks as fallback.")
            chunks, _ = self._store.get_document(sid)
            return (chunks or [])[:top_k]

        chunks, embs = self._store.get_document(sid)
        if chunks is None or embs is None or len(chunks) == 0:
            return []

        q_emb = self._embedder.embed([question]).reshape(1, -1).astype("float32")
        dim = embs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embs)
        k = min(top_k, index.ntotal)
        _, I = index.search(q_emb, k)
        return [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
