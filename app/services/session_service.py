"""
app/services/session_service.py  —  Privacy-first session store.

Two backends
────────────
RedisSessionStore      — production, survives restarts, scales across workers.
InProcessSessionStore  — local dev fallback when Redis is not available.

Key structure (Redis)
─────────────────────
  session:{sid}:meta        → JSON  {created_at, files:[...]}
  session:{sid}:chunks      → JSON  list[str]
  session:{sid}:embeddings  → raw bytes  (float32 ndarray, shape N×D)
  session:{sid}:dim         → str  embedding dimension

All keys share the same TTL (sliding expiry — refreshed on every access).
No raw document bytes are persisted; privacy is preserved by design.
"""
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("jurisai.session")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  Redis-backed store
# ══════════════════════════════════════════════════════════════════════════════

class RedisSessionStore:
    """Session store backed by Redis with sliding TTL expiry."""

    def __init__(self, url: str, ttl: int):
        self.ttl = ttl
        self._client = None
        self._url = url
        self._connect()

    def _connect(self):
        if not REDIS_AVAILABLE:
            log.warning("redis-py not installed — Redis store unavailable.")
            return
        try:
            self._client = redis.Redis.from_url(
                self._url,
                decode_responses=False,
                socket_connect_timeout=3,
                socket_timeout=5,
            )
            self._client.ping()
            log.info("Redis connected: %s", self._url)
        except Exception as exc:
            log.warning("Redis unavailable (%s).", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── Key helpers ───────────────────────────────────────────────────────────

    def _key(self, sid: str, suffix: str) -> str:
        return f"session:{sid}:{suffix}"

    def _touch(self, sid: str):
        """Refresh TTL on all session keys (sliding expiry)."""
        if not self._client:
            return
        for suffix in ("meta", "chunks", "embeddings", "dim"):
            try:
                self._client.expire(self._key(sid, suffix), self.ttl)
            except Exception:
                pass

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def create_session(self, sid: str):
        if not self._client:
            return
        meta = {"created_at": str(np.datetime64("now")), "files": []}
        try:
            self._client.setex(
                self._key(sid, "meta"), self.ttl, json.dumps(meta).encode()
            )
            log.info("Session created: %s (TTL=%ds)", sid, self.ttl)
        except Exception as exc:
            log.warning("create_session error: %s", exc)

    def session_exists(self, sid: str) -> bool:
        if not self._client:
            return False
        try:
            return bool(self._client.exists(self._key(sid, "meta")))
        except Exception:
            return False

    def delete_session(self, sid: str):
        if not self._client:
            return
        for suffix in ("meta", "chunks", "embeddings", "dim"):
            try:
                self._client.delete(self._key(sid, suffix))
            except Exception:
                pass
        log.info("Session deleted: %s", sid)

    # ── Document data ─────────────────────────────────────────────────────────

    def store_document(self, sid: str, filename: str,
                       chunks: List[str], embeddings: np.ndarray) -> bool:
        if not self._client:
            log.warning("Redis unavailable; document not stored for session %s.", sid)
            return False
        try:
            self._client.setex(
                self._key(sid, "chunks"), self.ttl,
                json.dumps(chunks).encode("utf-8"),
            )
            emb_bytes = embeddings.astype(np.float32).tobytes()
            self._client.setex(self._key(sid, "embeddings"), self.ttl, emb_bytes)
            self._client.setex(
                self._key(sid, "dim"), self.ttl,
                str(embeddings.shape[1]).encode(),
            )
            meta = self._get_meta(sid)
            if filename not in meta.get("files", []):
                meta.setdefault("files", []).append(filename)
            self._client.setex(
                self._key(sid, "meta"), self.ttl, json.dumps(meta).encode()
            )
            log.info("Stored %d chunks / %s for session %s.",
                     len(chunks), embeddings.shape, sid)
            return True
        except Exception as exc:
            log.exception("store_document error (session=%s): %s", sid, exc)
            return False

    def append_document(self, sid: str, filename: str,
                        new_chunks: List[str], new_embeddings: np.ndarray) -> bool:
        existing_chunks, existing_embs = self.get_document(sid)
        if existing_chunks is None:
            return self.store_document(sid, filename, new_chunks, new_embeddings)
        merged_chunks = existing_chunks + new_chunks
        merged_embs = np.vstack([existing_embs, new_embeddings.astype(np.float32)])
        return self.store_document(sid, filename, merged_chunks, merged_embs)

    def get_document(self, sid: str) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
        if not self._client:
            return None, None
        try:
            chunks_raw = self._client.get(self._key(sid, "chunks"))
            emb_raw = self._client.get(self._key(sid, "embeddings"))
            dim_raw = self._client.get(self._key(sid, "dim"))
            if not chunks_raw or not emb_raw or not dim_raw:
                return None, None
            chunks = json.loads(chunks_raw.decode("utf-8"))
            dim = int(dim_raw.decode())
            arr = np.frombuffer(emb_raw, dtype=np.float32).reshape(-1, dim).copy()
            self._touch(sid)
            return chunks, arr
        except Exception as exc:
            log.exception("get_document error (session=%s): %s", sid, exc)
            return None, None

    def get_files(self, sid: str) -> List[str]:
        return self._get_meta(sid).get("files", [])

    def remove_file(self, sid: str, filename: str,
                    all_chunks: List[str], all_embeddings: np.ndarray,
                    file_chunks: List[str]):
        file_set = set(file_chunks)
        kept_idx = [i for i, c in enumerate(all_chunks) if c not in file_set]
        if not kept_idx:
            self._client.delete(self._key(sid, "chunks"))
            self._client.delete(self._key(sid, "embeddings"))
            self._client.delete(self._key(sid, "dim"))
        else:
            new_chunks = [all_chunks[i] for i in kept_idx]
            new_embs = all_embeddings[kept_idx]
            self.store_document(sid, "__retained__", new_chunks, new_embs)
        meta = self._get_meta(sid)
        meta["files"] = [f for f in meta.get("files", []) if f != filename]
        self._client.setex(
            self._key(sid, "meta"), self.ttl, json.dumps(meta).encode()
        )

    def ttl_seconds(self, sid: str) -> int:
        if not self._client:
            return -1
        try:
            return int(self._client.ttl(self._key(sid, "meta")))
        except Exception:
            return -1

    def _get_meta(self, sid: str) -> dict:
        if not self._client:
            return {}
        try:
            raw = self._client.get(self._key(sid, "meta"))
            return json.loads(raw.decode()) if raw else {}
        except Exception:
            return {}


# ══════════════════════════════════════════════════════════════════════════════
#  In-process fallback
# ══════════════════════════════════════════════════════════════════════════════

class InProcessSessionStore:
    """
    Thread-safe in-memory fallback.
    Does NOT survive restarts and does NOT scale across workers.
    """

    def __init__(self, ttl: int):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._store: Dict[str, dict] = {}

    def _expired(self, entry: dict) -> bool:
        return (time.time() - entry.get("ts", 0)) > self.ttl

    def _evict(self):
        dead = [sid for sid, e in self._store.items() if self._expired(e)]
        for sid in dead:
            del self._store[sid]

    @property
    def available(self) -> bool:
        return True

    def create_session(self, sid: str):
        with self._lock:
            self._store[sid] = {
                "chunks": [], "embeddings": None, "files": [], "ts": time.time()
            }

    def session_exists(self, sid: str) -> bool:
        with self._lock:
            self._evict()
            return sid in self._store

    def delete_session(self, sid: str):
        with self._lock:
            self._store.pop(sid, None)

    def store_document(self, sid: str, filename: str,
                       chunks: List[str], embeddings: np.ndarray) -> bool:
        with self._lock:
            if sid not in self._store:
                self._store[sid] = {"files": [], "ts": time.time()}
            e = self._store[sid]
            e["chunks"] = chunks
            e["embeddings"] = embeddings.astype(np.float32)
            if filename not in e["files"] and filename != "__retained__":
                e["files"].append(filename)
            e["ts"] = time.time()
        return True

    def append_document(self, sid: str, filename: str,
                        new_chunks: List[str], new_embeddings: np.ndarray) -> bool:
        existing_chunks, existing_embs = self.get_document(sid)
        if existing_chunks is None:
            return self.store_document(sid, filename, new_chunks, new_embeddings)
        merged_chunks = existing_chunks + new_chunks
        merged_embs = np.vstack([existing_embs, new_embeddings.astype(np.float32)])
        return self.store_document(sid, filename, merged_chunks, merged_embs)

    def get_document(self, sid: str) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
        with self._lock:
            e = self._store.get(sid)
            if not e or self._expired(e):
                return None, None
            e["ts"] = time.time()
            return e.get("chunks"), e.get("embeddings")

    def get_files(self, sid: str) -> List[str]:
        with self._lock:
            return list(self._store.get(sid, {}).get("files", []))

    def remove_file(self, sid: str, filename: str,
                    all_chunks: List[str], all_embeddings: np.ndarray,
                    file_chunks: List[str]):
        file_set = set(file_chunks)
        kept_idx = [i for i, c in enumerate(all_chunks) if c not in file_set]
        new_chunks = [all_chunks[i] for i in kept_idx]
        new_embs = (
            all_embeddings[kept_idx]
            if kept_idx
            else np.zeros((0, all_embeddings.shape[1]), dtype=np.float32)
        )
        with self._lock:
            e = self._store.get(sid, {})
            e["chunks"] = new_chunks
            e["embeddings"] = new_embs
            e["files"] = [f for f in e.get("files", []) if f != filename]
            e["ts"] = time.time()

    def ttl_seconds(self, sid: str) -> int:
        with self._lock:
            e = self._store.get(sid)
            if not e:
                return -1
            remaining = self.ttl - (time.time() - e.get("ts", 0))
            return max(-1, int(remaining))

    def _get_meta(self, sid: str) -> dict:
        return {}


# ── Factory ───────────────────────────────────────────────────────────────────

def create_session_store(redis_url: str, ttl: int):
    """Return the best available session store."""
    if REDIS_AVAILABLE:
        store = RedisSessionStore(redis_url, ttl)
        if store.available:
            return store
        log.warning("Redis unavailable — falling back to in-process store.")
    else:
        log.warning("redis-py not installed — using in-process store.")
    return InProcessSessionStore(ttl)
