"""
app/services/embedding_service.py  —  Unified text-embedding interface.

Priority order
──────────────
1. Local SentenceTransformer  (default, fast, free)
2. Gemini Embeddings           (if USE_GEMINI_EMBEDDINGS=true)
3. Fallback between the two on failure
"""
import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger("jurisai.embedding")

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBED_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    LOCAL_EMBED_AVAILABLE = False


class EmbeddingService:
    """
    Generates dense vector embeddings for text using a local model or Gemini.
    Inject a GeminiService instance to enable the Gemini fallback.
    """

    def __init__(self, local_model_name: str,
                 prefer_local: bool = True,
                 gemini_service=None):
        self._prefer_local = prefer_local
        self._gemini = gemini_service
        self._local: Optional["SentenceTransformer"] = None

        if LOCAL_EMBED_AVAILABLE:
            try:
                self._local = SentenceTransformer(local_model_name)
                log.info("Local embedding model loaded: %s", local_model_name)
            except Exception as exc:
                log.warning("Failed to load local embedder: %s", exc)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _local_embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self._local.encode(texts, convert_to_numpy=True, show_progress_bar=False),
            dtype="float32",
        )

    def _gemini_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            res = self._gemini.embed(batch, task_type="retrieval_document")
            emb_list = getattr(res, "embeddings", None)
            if emb_list:
                for e in emb_list:
                    vals = getattr(e, "values", None) or getattr(e, "embedding", None)
                    all_embs.append(np.array(vals, dtype="float32"))
            elif isinstance(res, dict) and "embeddings" in res:
                for e in res["embeddings"]:
                    v = e.get("values") or e.get("embedding") or e
                    all_embs.append(np.array(v, dtype="float32"))
        return np.vstack(all_embs) if all_embs else np.zeros((0, 0), dtype="float32")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return a (N, D) float32 ndarray of embeddings for *texts*."""
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        # Strategy 1: prefer local
        if self._prefer_local and self._local is not None:
            try:
                return self._local_embed(texts)
            except Exception as exc:
                log.warning("Local embed failed, falling back to Gemini: %s", exc)

        # Strategy 2: Gemini
        if self._gemini and self._gemini.available:
            try:
                return self._gemini_embed(texts, batch_size)
            except Exception as exc:
                log.warning("Gemini embed failed, falling back to local: %s", exc)

        # Strategy 3: local as last resort
        if self._local is not None:
            return self._local_embed(texts)

        raise RuntimeError("No embedding method available.")

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience: embed a single string, return (D,) vector."""
        return self.embed([text])[0]

    @property
    def local_model(self):
        """Expose the underlying SentenceTransformer for precomputation tasks."""
        return self._local
