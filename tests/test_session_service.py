"""
tests/test_session_service.py  —  Unit tests for the in-process session store.

Run with:
    pytest tests/test_session_service.py -v
"""
import numpy as np
import pytest
from app.services.session_service import InProcessSessionStore


@pytest.fixture
def store():
    return InProcessSessionStore(ttl=1800)


def make_embs(n=5, d=64):
    return np.random.rand(n, d).astype("float32")


class TestInProcessSessionStore:
    def test_create_and_exists(self, store):
        store.create_session("sid-1")
        assert store.session_exists("sid-1")

    def test_nonexistent_session(self, store):
        assert not store.session_exists("ghost-sid")

    def test_delete_session(self, store):
        store.create_session("sid-del")
        store.delete_session("sid-del")
        assert not store.session_exists("sid-del")

    def test_store_and_get_document(self, store):
        sid = "sid-doc"
        store.create_session(sid)
        chunks = ["clause one", "clause two", "clause three"]
        embs = make_embs(3)
        store.store_document(sid, "test.pdf", chunks, embs)
        got_chunks, got_embs = store.get_document(sid)
        assert got_chunks == chunks
        assert got_embs.shape == embs.shape

    def test_append_document(self, store):
        sid = "sid-append"
        store.create_session(sid)
        chunks1 = ["a", "b"]
        chunks2 = ["c", "d"]
        store.store_document(sid, "file1.pdf", chunks1, make_embs(2))
        store.append_document(sid, "file2.pdf", chunks2, make_embs(2))
        got_chunks, got_embs = store.get_document(sid)
        assert len(got_chunks) == 4
        assert got_embs.shape[0] == 4

    def test_get_files(self, store):
        sid = "sid-files"
        store.create_session(sid)
        store.store_document(sid, "contract.pdf", ["x"], make_embs(1))
        files = store.get_files(sid)
        assert "contract.pdf" in files

    def test_ttl_positive_for_fresh_session(self, store):
        sid = "sid-ttl"
        store.create_session(sid)
        ttl = store.ttl_seconds(sid)
        assert ttl > 0

    def test_get_nonexistent_returns_none(self, store):
        chunks, embs = store.get_document("no-such-session")
        assert chunks is None
        assert embs is None
