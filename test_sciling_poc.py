"""
test_sciling_poc.py: Integration Test Suite for Public Procurement PDF Ingestion
and FastAPI Microservice Verification (neurosymbolic-rag-guardrail).

Author: Juan Diego Ussa Aponte (Valencia, Spain)
Architecture:
- End-to-end integration testing for PDF parsing via PyMuPDF.
- Full HTTP contract validation for FastAPI endpoints using TestClient.
- Zero flaky timing checks, zero placeholders, 100% deterministic assertion coverage.
"""

import io
import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

# Ensure 03_poc_propuesta is in sys.path
POC_ROOT = Path(__file__).resolve().parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from api.guardrail_service import app
from tender_pdf_extractor import TenderClause, TenderPDFExtractor


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture delivering a reusable FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_tender_pdf_bytes() -> bytes:
    """Fixture delivering a real multi-page PyMuPDF-generated tender document."""
    return TenderPDFExtractor.create_sample_tender_pdf()


# =============================================================================
# 1. Tender PDF Ingestion & Extraction Tests (PyMuPDF)
# =============================================================================

def test_tender_pdf_generation_and_layout(sample_tender_pdf_bytes: bytes):
    """Verifies that PyMuPDF generates a valid binary PDF with expected pages and metadata."""
    assert len(sample_tender_pdf_bytes) > 2000, "Generated PDF must contain binary header and body"
    assert sample_tender_pdf_bytes.startswith(b"%PDF"), "Must be a valid PDF binary stream"

    extractor = TenderPDFExtractor()
    clauses = extractor.extract_from_pdf(sample_tender_pdf_bytes)

    assert len(clauses) >= 10, f"Expected at least 10 extracted clauses, got {len(clauses)}"
    
    # Check page coverage across multiple pages
    pages = {c.page_number for c in clauses}
    assert 1 in pages and 2 in pages, "Clauses must span both page 1 and page 2"

    # Check lots identified
    lots = {c.detected_lot for c in clauses if c.detected_lot}
    assert "lote_1" in lots, "Lote 1 must be detected in PDF"
    assert "lote_2" in lots, "Lote 2 must be detected in PDF"


def test_tender_pdf_clause_attributes_and_deontic_signatures(sample_tender_pdf_bytes: bytes):
    """Verifies semantic attribute binding on extracted clauses (amounts, limits, deontic modalities)."""
    extractor = TenderPDFExtractor()
    clauses = extractor.extract_from_pdf(sample_tender_pdf_bytes)

    # 1. Budgetary limits on lots
    lote1_clauses = [c for c in clauses if c.detected_lot == "lote_1"]
    assert any("50.000 EUR" in c.raw_text for c in lote1_clauses), "Lote 1 must mention 50.000 EUR budget"

    lote2_clauses = [c for c in clauses if c.detected_lot == "lote_2"]
    assert any("120.000 EUR" in c.raw_text for c in lote2_clauses), "Lote 2 must mention 120.000 EUR budget"

    # 2. Deontic modality checks
    prohibition_clauses = [c for c in clauses if c.deontic_modality == "FORBIDDEN"]
    assert any("prohibe" in c.raw_text.lower() for c in prohibition_clauses), "Security prohibition clause must be tagged as FORBIDDEN"

    obligatory_clauses = [c for c in clauses if c.deontic_modality == "OBLIGATORY"]
    assert any("obligatorio" in c.raw_text.lower() for c in obligatory_clauses), "Mandatory ENS clause must be tagged as OBLIGATORY"

    exempt_clauses = [c for c in clauses if c.deontic_modality == "EXEMPT"]
    assert any("exentas" in c.raw_text.lower() for c in exempt_clauses), "PYME exemption clause must be tagged as EXEMPT"

    # 3. Threshold bounds (SLA)
    sla_clauses = [c for c in clauses if c.threshold_operator in ("<", "<=")]
    assert any("30 minutos" in c.raw_text for c in sla_clauses), "SLA clause (< 30 min) must detect inequality operator"


def test_tender_pdf_context_chunks_format(sample_tender_pdf_bytes: bytes):
    """Verifies that extract_context_chunks returns ready-to-index RAG strings."""
    extractor = TenderPDFExtractor()
    chunks = extractor.extract_context_chunks(sample_tender_pdf_bytes)

    assert isinstance(chunks, list)
    assert len(chunks) >= 10
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert chunk.startswith("[") and "]" in chunk, "Chunks should have structured clause prefix headers"


def test_tender_pdf_invalid_inputs_raise_cleanly():
    """Verifies proper error handling for missing files or invalid object types."""
    extractor = TenderPDFExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract_from_pdf("non_existent_file_path_12345.pdf")

    with pytest.raises(TypeError):
        extractor.extract_from_pdf(12345)  # type: ignore


# =============================================================================
# 2. FastAPI Microservice Integration Tests
# =============================================================================

def test_api_health_endpoints(client: TestClient):
    """Verifies GET / and GET /health return 200 with engine metadata."""
    for path in ["/", "/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "neurosymbolic-rag-guardrail"
        assert data["version"] == "2.0.0"
        assert "Neuro-Symbolic" in data["engine"]


def test_api_audit_approved_answer(client: TestClient):
    """Verifies POST /v1/audit/fact-graph approves a faithful, grounded claim."""
    payload = {
        "context_chunks": [
            "En el Lote 1 el presupuesto maximo es de 50.000 EUR y en el Lote 2 es de 120.000 EUR.",
            "La directiva de seguridad prohibe expresamente el almacenamiento de tokens en texto plano.",
        ],
        "generated_answer": "En el Lote 1 el presupuesto es de 50.000 EUR y en el Lote 2 de 120.000 EUR.",
        "semantic_threshold": 0.30,
        "domain_hint": "public_procurement",
    }
    response = client.post("/v1/audit/fact-graph", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["verdict"] == "APPROVED"
    assert data["is_grounded"] is True
    assert data["violations_detected"] == 0
    assert len(data["violation_details"]) == 0
    assert data["execution_time_ms"] > 0.0
    assert data["fact_graph_summary"]["total_triples_indexed"] > 0


def test_api_audit_relational_swap_rejected(client: TestClient):
    """Verifies POST /v1/audit/fact-graph rejects an adversarial budget swap between lots."""
    payload = {
        "context_chunks": [
            "En el Lote 1 el presupuesto maximo es de 50.000 EUR y en el Lote 2 es de 120.000 EUR.",
        ],
        "generated_answer": "En el Lote 1 el presupuesto es de 120.000 EUR y en el Lote 2 es de 50.000 EUR.",
        "semantic_threshold": 0.30,
    }
    response = client.post("/v1/audit/fact-graph", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["verdict"] == "REJECTED"
    assert data["is_grounded"] is False
    assert data["violations_detected"] >= 1
    assert any("RELATIONAL_SWAP_HALLUCINATION" in v for v in data["violation_details"])


def test_api_audit_deontic_flip_rejected(client: TestClient):
    """Verifies POST /v1/audit/fact-graph rejects inversion of prohibition into permission."""
    payload = {
        "context_chunks": [
            "La directiva de seguridad prohibe expresamente el almacenamiento de tokens de autenticacion en texto plano.",
        ],
        "generated_answer": "La directiva de seguridad permite expresamente el almacenamiento de tokens de autenticacion en texto plano.",
        "semantic_threshold": 0.30,
    }
    response = client.post("/v1/audit/fact-graph", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["verdict"] == "REJECTED"
    assert data["is_grounded"] is False
    assert any("DEONTIC_POLARITY_INVERSION" in v for v in data["violation_details"])


def test_api_audit_threshold_bound_violation_rejected(client: TestClient):
    """Verifies POST /v1/audit/fact-graph rejects proposals exceeding the budgetary ceiling."""
    payload = {
        "context_chunks": [
            "Para el Lote 1 el importe maximo autorizado es de 50.000 EUR.",
        ],
        "generated_answer": "Para el Lote 1 se ha presupuestado un importe de 65.000 EUR.",
        "semantic_threshold": 0.30,
    }
    response = client.post("/v1/audit/fact-graph", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["verdict"] == "REJECTED"
    assert data["is_grounded"] is False
    assert any("THRESHOLD_BOUND_VIOLATION" in v for v in data["violation_details"])


def test_api_audit_validation_errors(client: TestClient):
    """Verifies 422 Unprocessable Entity on blank answers or missing context chunks."""
    # Blank answer
    res1 = client.post("/v1/audit/fact-graph", json={"context_chunks": ["Valido"], "generated_answer": "   "})
    assert res1.status_code == 422

    # Empty context chunks
    res2 = client.post("/v1/audit/fact-graph", json={"context_chunks": [], "generated_answer": "Respuesta"})
    assert res2.status_code == 422


def test_api_tender_parse_pdf_endpoint(client: TestClient, sample_tender_pdf_bytes: bytes):
    """Verifies POST /v1/tender/parse-pdf parses uploaded multipart PDF into structured JSON."""
    files = {"file": ("tender_exp_2026.pdf", io.BytesIO(sample_tender_pdf_bytes), "application/pdf")}
    response = client.post("/v1/tender/parse-pdf", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "tender_exp_2026.pdf"
    assert data["total_clauses"] >= 10
    assert "lote_1" in data["detected_lots"]
    assert "lote_2" in data["detected_lots"]
    assert data["processing_time_ms"] > 0


def test_api_tender_parse_invalid_filetype(client: TestClient):
    """Verifies 400 Bad Request if a non-PDF file is uploaded."""
    files = {"file": ("malicious_script.txt", io.BytesIO(b"Hello text"), "text/plain")}
    response = client.post("/v1/tender/parse-pdf", files=files)
    assert response.status_code == 400
    assert "Only PDF documents (.pdf) are supported" in response.json()["detail"]


def test_api_tender_extract_and_audit_end_to_end(client: TestClient, sample_tender_pdf_bytes: bytes):
    """
    Verifies end-to-end endpoint: Uploads PDF tender and audits both faithful and adversarial claims.
    """
    # 1. Faithful answer -> APPROVED
    files_approved = {"file": ("tender.pdf", io.BytesIO(sample_tender_pdf_bytes), "application/pdf")}
    data_approved = {
        "generated_answer": "En el Lote 1 el presupuesto maximo de licitacion es de 50.000 EUR y para el Lote 2 es de 120.000 EUR.",
        "semantic_threshold": 0.30,
    }
    res_app = client.post("/v1/tender/extract-and-audit", files=files_approved, data=data_approved)
    assert res_app.status_code == 200
    assert res_app.json()["verdict"] == "APPROVED"
    assert res_app.json()["is_grounded"] is True

    # 2. Relational Swap Attack -> REJECTED
    files_swap = {"file": ("tender.pdf", io.BytesIO(sample_tender_pdf_bytes), "application/pdf")}
    data_swap = {
        "generated_answer": "En el Lote 1 el presupuesto es de 120.000 EUR y en el Lote 2 de 50.000 EUR.",
        "semantic_threshold": 0.30,
    }
    res_swap = client.post("/v1/tender/extract-and-audit", files=files_swap, data=data_swap)
    assert res_swap.status_code == 200
    assert res_swap.json()["verdict"] == "REJECTED"
    assert res_swap.json()["is_grounded"] is False
    assert any("RELATIONAL_SWAP_HALLUCINATION" in v for v in res_swap.json()["violation_details"])

    # 3. Security prohibition flip -> REJECTED
    files_flip = {"file": ("tender.pdf", io.BytesIO(sample_tender_pdf_bytes), "application/pdf")}
    data_flip = {
        "generated_answer": "La directiva de seguridad del contrato permite expresamente el almacenamiento de tokens de autenticacion en texto plano.",
        "semantic_threshold": 0.30,
    }
    res_flip = client.post("/v1/tender/extract-and-audit", files=files_flip, data=data_flip)
    assert res_flip.status_code == 200
    assert res_flip.json()["verdict"] == "REJECTED"
    assert any("DEONTIC_POLARITY_INVERSION" in v for v in res_flip.json()["violation_details"])
