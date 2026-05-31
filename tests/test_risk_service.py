"""
tests/test_risk_service.py  —  Unit tests for the NLP risk analysis engine.

Run with:
    pytest tests/test_risk_service.py -v
"""
import pytest
from app.services.risk_service import RiskService


@pytest.fixture(scope="module")
def risk_service():
    """Instantiate service without an embedding model (regex-only mode)."""
    return RiskService(embedding_service=None)


class TestRiskServiceRegexDetection:
    def test_indemnification_detected(self, risk_service):
        chunks = ["The vendor shall indemnify and hold harmless the client from all claims."]
        result = risk_service.analyze(chunks)
        labels = [f["label"] for f in result["findings"]]
        assert "Indemnification" in labels

    def test_non_compete_detected(self, risk_service):
        chunks = ["Employee agrees to a non-compete clause for 24 months after termination."]
        result = risk_service.analyze(chunks)
        labels = [f["label"] for f in result["findings"]]
        assert "Non-Compete" in labels

    def test_safe_definitions_clause_skipped(self, risk_service):
        chunks = ["For the purposes of this agreement, 'Party' means any signatory herein defined."]
        result = risk_service.analyze(chunks)
        # Definitions clause — should not generate high risk findings
        high = [f for f in result["findings"] if f["level"] == "high"]
        assert len(high) == 0

    def test_empty_chunks_returns_zero_score(self, risk_service):
        result = risk_service.analyze([])
        assert result["risk_score"] == 0
        assert result["findings"] == []

    def test_risk_score_bounded(self, risk_service):
        chunks = [
            "The vendor shall indemnify the client for unlimited liability.",
            "Non-compete and liquidated damages apply. Automatic renewal clause.",
            "Unilateral amendment rights reserved. Arbitration is mandatory.",
        ]
        result = risk_service.analyze(chunks)
        assert 0 <= result["risk_score"] <= 100

    def test_output_structure(self, risk_service):
        chunks = ["Arbitration shall be the exclusive remedy for all disputes."]
        result = risk_service.analyze(chunks)
        assert "risk_score" in result
        assert "risk_level" in result
        assert "findings" in result
        assert "summary" in result
        assert result["risk_level"] in ("High", "Medium", "Low")


class TestRiskServiceFindingFields:
    def test_finding_has_required_fields(self, risk_service):
        chunks = ["The contract shall automatically renew unless notice is given."]
        result = risk_service.analyze(chunks)
        for finding in result["findings"]:
            assert "label" in finding
            assert "level" in finding
            assert "explanation" in finding
            assert finding["level"] in ("high", "medium", "low")

    def test_duplicate_labels_deduplicated(self, risk_service):
        # Same risk pattern repeated across multiple chunks
        chunks = [
            "The vendor shall indemnify the client.",
            "The contractor shall indemnify and hold harmless all parties.",
        ]
        result = risk_service.analyze(chunks)
        labels = [f["label"] for f in result["findings"]]
        assert len(labels) == len(set(labels)), "Duplicate labels found in findings"
