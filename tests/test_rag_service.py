"""
tests/test_rag_service.py  —  Unit tests for the RAG pipeline.

Run with:
    pytest tests/test_rag_service.py -v
"""
import pytest
from app.services.rag_service import RAGService


class TestDocTypeDetection:
    def test_legal_doc_detected(self):
        text = "The parties agree to this contract. Indemnification and liability clauses apply. The agreement is governed by law."
        doc_type = RAGService.detect_doc_type(text)
        assert doc_type == "legal"

    def test_default_for_short_text(self):
        text = "Hello world."
        doc_type = RAGService.detect_doc_type(text)
        assert doc_type == "default"


class TestAdaptiveChunking:
    @pytest.fixture
    def rag(self):
        # Minimal RAG instance without real services
        return RAGService(embedding_service=None, session_store=None)

    def test_chunks_produced(self, rag):
        text = " ".join(["word"] * 500)
        chunks = rag.chunk_text(text, filename="test.txt")
        assert len(chunks) > 0

    def test_chunks_not_empty(self, rag):
        text = "This is a legal contract. The vendor shall indemnify the client. Payment is due in 30 days."
        chunks = rag.chunk_text(text, filename="test.pdf")
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_overlap_creates_more_chunks(self, rag):
        # A long document should produce multiple overlapping chunks
        text = " ".join(["legal"] * 1000)
        chunks = rag.chunk_text(text, filename="long.pdf")
        assert len(chunks) > 2
