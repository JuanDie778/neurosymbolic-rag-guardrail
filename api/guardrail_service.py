"""
guardrail_service.py: Production FastAPI Microservice for Neuro-Symbolic RAG Guardrail
and Public Procurement Fact-Graph Verification (neurosymbolic-rag-guardrail).

Author: Juan Diego Ussa Aponte (Valencia, Spain)
Architecture:
- High-throughput, sub-5ms deterministic audit microservice on FastAPI.
- Endpoints:
    * GET  /health                       -> Service status and hardware/runtime telemetry.
    * POST /v1/audit/fact-graph          -> Core endpoint: audits LLM responses against context chunks.
    * POST /v1/tender/parse-pdf          -> Ingests real PDF tender documents and returns structured clauses.
    * POST /v1/tender/extract-and-audit  -> End-to-end ingestion and auditing of PDF tenders vs generated claims.
- Zero external LLM-as-a-judge latency: instant, reproducible, explainable compliance verdicts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure parent directory is in sys.path for robust imports across runners
POC_ROOT = Path(__file__).resolve().parent.parent
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from neurosymbolic_rag_guardrail import NeuroSymbolicRAGGuardrail
from symbolic_fact_graph import SymbolicFactGraph
from tender_pdf_extractor import TenderClause, TenderPDFExtractor


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class AuditFactGraphRequest(BaseModel):
    """Payload for auditing generated answers against knowledge context chunks."""

    context_chunks: List[str] = Field(
        ...,
        min_length=1,
        description="Ground truth context chunks retrieved from vector store, ERP, or PDF clauses.",
        examples=[[
            "En el Lote 1 el presupuesto maximo es de 50.000 EUR y en el Lote 2 es de 120.000 EUR.",
            "La directiva de seguridad prohibe expresamente el almacenamiento de tokens en texto plano.",
        ]],
    )
    generated_answer: str = Field(
        ...,
        min_length=1,
        description="Generated candidate answer or summary produced by LLM/Agent to be verified.",
        examples=["En el Lote 1 el presupuesto es de 50.000 EUR y en el Lote 2 de 120.000 EUR."],
    )
    semantic_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for unanchored sentence extrapolation check.",
    )
    domain_hint: Optional[str] = Field(
        default=None,
        description="Optional domain label (e.g., 'public_procurement', 'biomedical', 'infrastructure').",
    )


class AuditFactGraphResponse(BaseModel):
    """Forensic verification verdict emitted by the Neuro-Symbolic Guardrail."""

    verdict: str = Field(..., description="Binary decision: 'APPROVED' or 'REJECTED'.")
    is_grounded: bool = Field(..., description="True if no factual, deontic, or numerical violations exist.")
    violations_detected: int = Field(..., description="Count of discrete violations identified.")
    violation_details: List[str] = Field(..., description="Forensic list describing every detected violation.")
    mean_semantic_similarity: float = Field(..., description="Average semantic alignment score across answer clauses.")
    execution_time_ms: float = Field(..., description="Total guardrail verification time in milliseconds on CPU.")
    domain: Optional[str] = Field(default=None, description="Domain context evaluated.")
    fact_graph_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Symbolic fact-graph telemetry (triples, bound operators, entities)."
    )


class TenderParseResponse(BaseModel):
    """Extracted clauses and metadata from uploaded tender PDF."""

    filename: str
    total_clauses: int
    clauses: List[Dict[str, Any]]
    detected_lots: List[str]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """System health check and engine metadata."""

    status: str
    service: str
    version: str
    author: str
    engine: str
    default_semantic_threshold: float


# -----------------------------------------------------------------------------
# FastAPI Application Factory
# -----------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Instantiates and configures the production FastAPI guardrail application."""
    app = FastAPI(
        title="neurosymbolic-rag-guardrail API",
        version="2.0.0",
        description=(
            "High-throughput Neuro-Symbolic RAG Guardrail Microservice. "
            "Evaluates LLM responses for relational entity swaps, deontic polarity flips, "
            "and threshold bound violations in sub-5ms on CPU."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for frontend and agentic orchestration integrations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize reusable stateless components
    pdf_extractor = TenderPDFExtractor()

    @app.get("/", response_model=HealthResponse, tags=["Monitoring"])
    @app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
    def health_check() -> HealthResponse:
        """Returns service availability, version, and architecture metadata."""
        return HealthResponse(
            status="healthy",
            service="neurosymbolic-rag-guardrail",
            version="2.0.0",
            author="Juan Diego Ussa Aponte",
            engine="Neuro-Symbolic Fact-Graph & Deontic Logic Engine v2.0",
            default_semantic_threshold=0.30,
        )

    @app.post(
        "/v1/audit/fact-graph",
        response_model=AuditFactGraphResponse,
        status_code=status.HTTP_200_OK,
        tags=["Auditing"],
        summary="Audit generated response against retrieved context chunks",
    )
    def audit_fact_graph(payload: AuditFactGraphRequest) -> AuditFactGraphResponse:
        """
        Executes neuro-symbolic factual verification:
        1. Builds a local Symbolic Fact Graph from `context_chunks`.
        2. Binds entities, quantities, units, and deontic polarities.
        3. Audits `generated_answer` for relational swaps, modal inversions, and numerical drift.
        4. Emits a deterministic APPROVED/REJECTED verdict in < 3ms.
        """
        t0 = time.perf_counter()

        # Sanitize input strings
        clean_chunks = [c.strip() for c in payload.context_chunks if c and c.strip()]
        if not clean_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one non-empty context chunk is required for fact-graph auditing.",
            )

        clean_answer = payload.generated_answer.strip()
        if not clean_answer:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The generated_answer to verify cannot be blank.",
            )

        # Execute Guardrail verification
        guardrail = NeuroSymbolicRAGGuardrail(semantic_alignment_threshold=payload.semantic_threshold)
        eval_result = guardrail.evaluate_response(clean_chunks, clean_answer)

        # Extract symbolic fact-graph telemetry summary
        ctx_graph = SymbolicFactGraph(clean_chunks)
        summary = {
            "total_triples_indexed": len(ctx_graph.triples),
            "subjects_indexed": sorted(list({t.primary_subject for t in ctx_graph.triples if t.primary_subject})),
            "deontic_clauses_indexed": len(ctx_graph.clause_signatures),
            "threshold_bounds_detected": [
                {
                    "subject": t.primary_subject,
                    "operator": t.quantity.bound_operator,
                    "quantity": t.quantity.raw_text,
                }
                for t in ctx_graph.triples
                if t.quantity and t.quantity.bound_operator and t.quantity.bound_operator != "=="
            ],
        }

        exec_time = (time.perf_counter() - t0) * 1000.0

        return AuditFactGraphResponse(
            verdict=eval_result["verdict"],
            is_grounded=eval_result["is_grounded"],
            violations_detected=eval_result["violations_detected"],
            violation_details=eval_result["violation_details"],
            mean_semantic_similarity=eval_result["mean_semantic_similarity"],
            execution_time_ms=round(exec_time, 3),
            domain=payload.domain_hint,
            fact_graph_summary=summary,
        )

    @app.post(
        "/v1/tender/parse-pdf",
        response_model=TenderParseResponse,
        status_code=status.HTTP_200_OK,
        tags=["Public Procurement / BidBrief"],
        summary="Extract structured tender clauses and lots from a PDF file",
    )
    async def parse_tender_pdf(file: UploadFile = File(...)) -> TenderParseResponse:
        """Parses an uploaded public procurement PDF tender and returns structured clauses."""
        t0 = time.perf_counter()
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{file.filename}'. Only PDF documents (.pdf) are supported.",
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF file is empty.",
            )

        try:
            clauses = pdf_extractor.extract_from_pdf(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Failed to parse PDF document: {str(exc)}",
            ) from exc

        lots = sorted(list({c.detected_lot for c in clauses if c.detected_lot}))
        elapsed = (time.perf_counter() - t0) * 1000.0

        return TenderParseResponse(
            filename=file.filename,
            total_clauses=len(clauses),
            clauses=[c.to_dict() for c in clauses],
            detected_lots=lots,
            processing_time_ms=round(elapsed, 3),
        )

    @app.post(
        "/v1/tender/extract-and-audit",
        response_model=AuditFactGraphResponse,
        status_code=status.HTTP_200_OK,
        tags=["Public Procurement / BidBrief"],
        summary="End-to-end: Ingest PDF tender and audit generated answer in one call",
    )
    async def extract_and_audit_tender(
        file: UploadFile = File(...),
        generated_answer: str = Form(..., description="The agent response to audit against the PDF tender."),
        semantic_threshold: float = Form(default=0.30),
    ) -> AuditFactGraphResponse:
        """
        End-to-end workflow:
        1. Reads uploaded PDF tender directly in-memory via PyMuPDF.
        2. Extracts clean RAG context chunks.
        3. Audits `generated_answer` against the extracted tender facts.
        """
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{file.filename}'. Only PDF files are supported.",
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded PDF is empty.")

        try:
            context_chunks = pdf_extractor.extract_context_chunks(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Could not extract context chunks from PDF: {str(exc)}",
            ) from exc

        audit_req = AuditFactGraphRequest(
            context_chunks=context_chunks,
            generated_answer=generated_answer,
            semantic_threshold=semantic_threshold,
            domain_hint="public_procurement_tender",
        )
        return audit_fact_graph(audit_req)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    print("==========================================================================")
    print("STARTING NEURO-SYMBOLIC RAG GUARDRAIL MICROSERVICE (FastAPI)")
    print("Host: 127.0.0.1:8000 | Docs: http://127.0.0.1:8000/docs")
    print("==========================================================================")
    uvicorn.run("guardrail_service:app", host="127.0.0.1", port=8000, reload=False)
