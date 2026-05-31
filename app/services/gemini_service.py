"""
app/services/gemini_service.py  —  Gemini API integration with key rotation.
All Gemini calls go through this module; no direct genai imports elsewhere.
"""
import itertools
import logging
from typing import Optional

log = logging.getLogger("jurisai.gemini")

# ── Optional import ───────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False


class GeminiService:
    """
    Wraps google.generativeai with automatic key rotation across a pool of
    API keys and graceful degradation when Gemini is unavailable.
    """

    def __init__(self, primary_key: Optional[str], key_pool: list,
                 chat_model_name: str, embed_model: str):
        self.chat_model_name = chat_model_name
        self.embed_model = embed_model
        self.available = False

        if not GENAI_AVAILABLE:
            log.warning("google-generativeai not installed — Gemini unavailable.")
            return

        # Build key pool; primary key always first
        all_keys = []
        if primary_key:
            all_keys.append(primary_key)
        for k in key_pool:
            if k and k != primary_key:
                all_keys.append(k)

        if not all_keys:
            log.warning("No Gemini API key configured.")
            return

        self._key_cycle = itertools.cycle(all_keys)
        self._current_key = next(self._key_cycle)
        self._configure(self._current_key)
        self.available = True
        log.info("GeminiService ready (model=%s, keys=%d).", chat_model_name, len(all_keys))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _configure(self, key: str):
        try:
            genai.configure(api_key=key)
        except Exception as e:
            log.warning("genai.configure error: %s", e)

    def _rotate(self):
        if hasattr(self, "_key_cycle"):
            self._current_key = next(self._key_cycle)
            self._configure(self._current_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, inputs, tries: int = 3):
        """Call GenerativeModel.generate_content with automatic key rotation."""
        if not self.available:
            raise RuntimeError("Gemini unavailable — no API key or library missing.")
        last_exc = None
        for _ in range(max(tries, 1)):
            try:
                model = genai.GenerativeModel(self.chat_model_name)
                return model.generate_content(inputs)
            except Exception as exc:
                last_exc = exc
                log.warning("Gemini generate error: %s", exc)
                self._rotate()
        raise RuntimeError(f"Gemini generate failed after {tries} tries: {last_exc}")

    def embed(self, content, task_type: str = "retrieval_document", tries: int = 3):
        """Call genai.embed_content with automatic key rotation."""
        if not self.available:
            raise RuntimeError("Gemini unavailable — no API key or library missing.")
        last_exc = None
        for _ in range(max(tries, 1)):
            try:
                return genai.embed_content(
                    model=self.embed_model,
                    content=content,
                    task_type=task_type,
                )
            except Exception as exc:
                last_exc = exc
                log.warning("Gemini embed error: %s", exc)
                self._rotate()
        raise RuntimeError(f"Gemini embed failed after {tries} tries: {last_exc}")
