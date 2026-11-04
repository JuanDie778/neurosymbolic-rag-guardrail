"""
tender_pdf_extractor.py: Production-Grade PDF Parsing and Structured Clause Extraction
for Public Procurement and Enterprise Contracts (Sectorial Open-Source Library: neurosymbolic-rag-guardrail).

Author: Juan Diego Ussa Aponte (Valencia, Spain)
Architecture:
- High-fidelity PDF layout and text stream parsing using PyMuPDF (pymupdf / fitz).
- Domain-aware segmentation for Public Procurement Tenders (PCAP/PPT) and SLA specifications.
- In-line extraction of lots, budgetary ceilings, execution windows, and deontic constraints.
- Direct bridging to SymbolicFactGraph and NeuroSymbolicRAGGuardrail context chunks.
- Built-in generator of synthetic, multi-page compliant tender PDFs for zero-dependency test suites.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import pymupdf  # PyMuPDF 1.24+ recommended import
except ImportError:
    try:
        import fitz as pymupdf  # type: ignore # Fallback for legacy installs
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for tender_pdf_extractor. Please install it via: pip install pymupdf"
        ) from exc

try:
    from symbolic_fact_graph import (
        DeonticPolarityAnalyzer,
        UnitNormalizer,
        strip_accents,
    )
except ImportError:
    # If running from another working directory, adjust sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from symbolic_fact_graph import (
        DeonticPolarityAnalyzer,
        UnitNormalizer,
        strip_accents,
    )


@dataclass(frozen=True)
class TenderClause:
    """Structured representation of an extracted public procurement clause."""

    clause_id: str
    section_title: str
    raw_text: str
    page_number: int
    detected_lot: Optional[str] = None
    detected_quantities: List[str] = field(default_factory=list)
    deontic_modality: Optional[str] = None
    threshold_operator: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "section_title": self.section_title,
            "raw_text": self.raw_text,
            "page_number": self.page_number,
            "detected_lot": self.detected_lot,
            "detected_quantities": list(self.detected_quantities),
            "deontic_modality": self.deontic_modality,
            "threshold_operator": self.threshold_operator,
        }


class TenderPDFExtractor:
    """
    Production-grade PDF parsing engine for public procurement tenders and legal specs.
    Extracts text blocks, recognizes section hierarchy, isolates clauses, and links
    symbolic fact attributes (lots, currencies, SLAs, deontic modalities).
    """

    def __init__(self, min_clause_length: int = 20):
        self.min_clause_length = min_clause_length
        self.unit_normalizer = UnitNormalizer()
        self.deontic_analyzer = DeonticPolarityAnalyzer()

    def _open_document(self, source: Union[str, Path, bytes, io.BytesIO]) -> pymupdf.Document:
        """Opens a PyMuPDF document safely from a filepath, raw bytes, or stream."""
        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            if not path_obj.exists():
                raise FileNotFoundError(f"Tender PDF document not found at: {path_obj}")
            return pymupdf.open(str(path_obj))
        elif isinstance(source, bytes):
            return pymupdf.open(stream=source, filetype="pdf")
        elif isinstance(source, io.BytesIO):
            return pymupdf.open(stream=source.getvalue(), filetype="pdf")
        else:
            raise TypeError(f"Unsupported PDF source type: {type(source)}")

    def extract_from_pdf(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[TenderClause]:
        """
        Parses the PDF document and returns a structured list of TenderClause objects.
        Analyzes headers, lots, numeric quantities, and deontic constraints.
        """
        doc = self._open_document(source)
        clauses: List[TenderClause] = []
        clause_counter = 1
        current_section = "GENERAL"

        section_patterns = [
            (re.compile(r"^(?:CL[ÁA]USULA|\d+[\.\)]|SECCI[ÓO]N)\s*[:\.]?\s*(.+)$", re.IGNORECASE), 1),
            (re.compile(r"^(?:OBJETO|PRESUPUESTO|LOTES|PLAZO|REQUISITOS|SLA|ACUERDOS)\b.*$", re.IGNORECASE), 0),
        ]

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1

                # Extract text blocks: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type == 0 is regular text, 1 is image
                blocks = page.get_text("blocks")

                for block in blocks:
                    if len(block) < 5 or block[4] is None:
                        continue
                    block_text = str(block[4]).strip()
                    if not block_text:
                        continue

                    # Check if block is a section header
                    first_line = block_text.splitlines()[0].strip()
                    matched_header = False
                    for pattern, group_idx in section_patterns:
                        match = pattern.match(first_line)
                        if match and len(first_line) < 120:
                            current_section = match.group(group_idx).strip() if group_idx > 0 else first_line
                            matched_header = True
                            break

                    # Split block into coherent clause sentences
                    raw_paragraphs = [p.strip() for p in block_text.split("\n\n") if len(p.strip()) >= self.min_clause_length]
                    if not raw_paragraphs and len(block_text) >= self.min_clause_length:
                        raw_paragraphs = [block_text]

                    for paragraph in raw_paragraphs:
                        # Clean multiple whitespace and newlines inside the clause
                        cleaned_text = " ".join(paragraph.split())
                        if len(cleaned_text) < self.min_clause_length:
                            continue

                        # Extract metadata for this clause
                        lot_match = re.search(r"\blote\s*([0-9a-zA-Z]+)\b", cleaned_text, re.IGNORECASE)
                        detected_lot = f"lote_{lot_match.group(1).lower()}" if lot_match else None

                        # Extract quantities and bounds
                        quantities_found = []
                        bound_operators = []
                        norm_quantities = self.unit_normalizer.extract_quantities(cleaned_text)
                        for q in norm_quantities:
                            quantities_found.append(q.raw_text)
                            if q.bound_operator and q.bound_operator != "==":
                                bound_operators.append(q.bound_operator)

                        # Extract deontic modalities
                        deontic_sig = self.deontic_analyzer.analyze_clause(cleaned_text)
                        deontic_mod = deontic_sig.deontic_state if deontic_sig else None
                        op = bound_operators[0] if bound_operators else None

                        cid = f"CL-{clause_counter:03d}"
                        if detected_lot:
                            cid = f"{detected_lot.upper()}-{clause_counter:02d}"

                        clause = TenderClause(
                            clause_id=cid,
                            section_title=current_section,
                            raw_text=cleaned_text,
                            page_number=page_num,
                            detected_lot=detected_lot,
                            detected_quantities=quantities_found,
                            deontic_modality=deontic_mod,
                            threshold_operator=op,
                        )
                        clauses.append(clause)
                        clause_counter += 1

        finally:
            doc.close()

        return clauses

    def extract_context_chunks(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[str]:
        """
        Extracts clean, structured text chunks suitable for direct injection into
        NeuroSymbolicRAGGuardrail and SymbolicFactGraph indices.
        """
        clauses = self.extract_from_pdf(source)
        chunks: List[str] = []
        for c in clauses:
            # Format chunk with semantic clarity
            prefix = f"[{c.clause_id} - {c.section_title}] " if c.section_title != "GENERAL" else f"[{c.clause_id}] "
            chunks.append(f"{prefix}{c.raw_text}")
        return chunks

    @staticmethod
    def create_sample_tender_pdf(output_path: Optional[Union[str, Path]] = None) -> bytes:
        """
        Generates an authentic, multi-page public procurement tender document (PPT/PCAP)
        in PDF format using PyMuPDF. Useful for self-contained testing, CI, and local demos.
        """
        doc = pymupdf.open()

        # Page 1: Administrative context, Lots, and Deadlines
        page1 = doc.new_page(width=595, height=842)  # Standard A4 size in points
        
        # Header banner
        page1.draw_rect(pymupdf.Rect(40, 30, 555, 75), color=(0.12, 0.25, 0.45), fill=(0.12, 0.25, 0.45))
        page1.insert_text(
            pymupdf.Point(50, 58),
            "PLIEGO DE PRESCRIPCIONES TÉCNICAS Y CLÁUSULAS ADMINISTRATIVAS",
            fontsize=12,
            color=(1, 1, 1),
            fontname="helv",
        )

        y = 100
        page1.insert_text(
            pymupdf.Point(45, y),
            "EXPEDIENTE: EXP-2026-SCILING-09 | INFRAESTRUCTURA DE IA Y LICITACIONES",
            fontsize=10,
            color=(0.3, 0.3, 0.3),
            fontname="helv",
        )

        y += 30
        page1.insert_text(pymupdf.Point(45, y), "CLÁUSULA 1: OBJETO DEL CONTRATO", fontsize=11, color=(0.1, 0.2, 0.4), fontname="helv")
        y += 18
        p1_text = (
            "El presente contrato tiene por objeto la contratacion del suministro de infraestructura de computacion "
            "avanzada para modelos de lenguaje y el despliegue de la plataforma de analisis documental automatizado."
        )
        page1.insert_textbox(pymupdf.Rect(45, y, 550, y + 40), p1_text, fontsize=9.5, fontname="helv")

        y += 50
        page1.insert_text(pymupdf.Point(45, y), "CLÁUSULA 2: LOTES Y PRESUPUESTO BASE DE LICITACIÓN", fontsize=11, color=(0.1, 0.2, 0.4), fontname="helv")
        y += 18
        p2_text_lote1 = (
            "En el Lote 1 el presupuesto maximo de licitacion es de 50.000 EUR (cincuenta mil euros), IVA excluido, "
            "destinado exclusivamente a la adquisicion de servidores GPU de inferencia de baja latencia."
        )
        page1.insert_textbox(pymupdf.Rect(45, y, 550, y + 40), p2_text_lote1, fontsize=9.5, fontname="helv")

        y += 50
        p2_text_lote2 = (
            "En el Lote 2 el presupuesto maximo de licitacion es de 120.000 EUR (ciento veinte mil euros), IVA excluido, "
            "destinado a licencias de software, orquestacion multiagente y soporte tecnico especializado durante el periodo contratado."
        )
        page1.insert_textbox(pymupdf.Rect(45, y, 550, y + 45), p2_text_lote2, fontsize=9.5, fontname="helv")

        y += 55
        page1.insert_text(pymupdf.Point(45, y), "CLÁUSULA 3: PLAZOS Y CONDICIONES DE EJECUCIÓN", fontsize=11, color=(0.1, 0.2, 0.4), fontname="helv")
        y += 18
        p3_text = (
            "El plazo maximo de entrega para los equipos del Lote 1 es de 2 meses (equivalente a 60 dias naturales) "
            "contados a partir de la firma del acta de inicio de ejecucion."
        )
        page1.insert_textbox(pymupdf.Rect(45, y, 550, y + 35), p3_text, fontsize=9.5, fontname="helv")

        y += 45
        p3_text_b = (
            "El plazo de ejecucion y soporte para el despliegue del Lote 2 tendra una duracion estipulada de 12 meses."
        )
        page1.insert_textbox(pymupdf.Rect(45, y, 550, y + 25), p3_text_b, fontsize=9.5, fontname="helv")

        # Page 2: Security, Deontic Constraints, and SLAs
        page2 = doc.new_page(width=595, height=842)
        page2.draw_rect(pymupdf.Rect(40, 30, 555, 60), color=(0.12, 0.25, 0.45), fill=(0.12, 0.25, 0.45))
        page2.insert_text(
            pymupdf.Point(50, 48),
            "PRESCRIPCIONES TÉCNICAS: REQUISITOS DEÓNTICOS Y NIVELES DE SERVICIO",
            fontsize=11,
            color=(1, 1, 1),
            fontname="helv",
        )

        y = 80
        page2.insert_text(pymupdf.Point(45, y), "CLÁUSULA 4: SEGURIDAD, POLÍTICAS Y REQUISITOS OBLIGATORIOS", fontsize=11, color=(0.1, 0.2, 0.4), fontname="helv")
        y += 18
        p4_text_ens = (
            "Es estrictamente obligatorio para todos los proveedores licitadores acreditar certificacion en el "
            "Esquema Nacional de Seguridad (ENS) nivel Medio o Superior vigente en la fecha de presentacion de la oferta."
        )
        page2.insert_textbox(pymupdf.Rect(45, y, 550, y + 40), p4_text_ens, fontsize=9.5, fontname="helv")

        y += 50
        p4_text_sec = (
            "La directiva de seguridad del contrato prohibe expresamente el almacenamiento de credenciales o "
            "tokens de autenticacion en texto plano en cualquier log, repositorio o base de datos de produccion."
        )
        page2.insert_textbox(pymupdf.Rect(45, y, 550, y + 40), p4_text_sec, fontsize=9.5, fontname="helv")

        y += 50
        p4_text_exempt = (
            "Las pequenas y medianas empresas (PYMES) de base tecnologica e innovadora estan expresamente exentas "
            "de presentar garantia provisional en la fase de licitacion conforme al articulo 106 de la LCSP."
        )
        page2.insert_textbox(pymupdf.Rect(45, y, 550, y + 40), p4_text_exempt, fontsize=9.5, fontname="helv")

        y += 55
        page2.insert_text(pymupdf.Point(45, y), "CLÁUSULA 5: ACUERDOS DE NIVEL DE SERVICIO (SLA)", fontsize=11, color=(0.1, 0.2, 0.4), fontname="helv")
        y += 18
        p5_text_sla1 = (
            "El tiempo maximo de respuesta ante cualquier incidencia tecnica de severidad critica debera ser inferior a 30 minutos."
        )
        page2.insert_textbox(pymupdf.Rect(45, y, 550, y + 30), p5_text_sla1, fontsize=9.5, fontname="helv")

        y += 40
        p5_text_sla2 = (
            "La disponibilidad operativa mensual de la plataforma de inferencia vLLM en produccion debera ser superior al 99.5%."
        )
        page2.insert_textbox(pymupdf.Rect(45, y, 550, y + 30), p5_text_sla2, fontsize=9.5, fontname="helv")

        # Save or return bytes
        pdf_bytes = doc.tobytes()
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(pdf_bytes)

        doc.close()
        return pdf_bytes


def main():
    """CLI demonstration and verification of the TenderPDFExtractor."""
    print("==========================================================================")
    print("TENDER PDF EXTRACTOR: HIGH-FIDELITY PUBLIC PROCUREMENT PARSER")
    print("Library: neurosymbolic-rag-guardrail | Domain: BidBrief / Public Tenders")
    print("==========================================================================")

    # 1. Generate compliant synthetic tender PDF
    print("[*] Generating synthetic multi-page tender PDF with PyMuPDF...")
    pdf_bytes = TenderPDFExtractor.create_sample_tender_pdf()
    print(f"[+] Successfully generated sample PDF ({len(pdf_bytes)} bytes).")

    # 2. Extract structured clauses
    extractor = TenderPDFExtractor()
    clauses = extractor.extract_from_pdf(pdf_bytes)
    print(f"[+] Extracted {len(clauses)} structured clauses across pages:")

    for c in clauses:
        print(f"  - [{c.clause_id}] (Page {c.page_number} | {c.section_title})")
        print(f"    Text: {c.raw_text[:75]}...")
        if c.detected_lot:
            print(f"    * Lot: {c.detected_lot}")
        if c.detected_quantities:
            print(f"    * Quantities: {c.detected_quantities}")
        if c.deontic_modality:
            print(f"    * Deontic Modality: {c.deontic_modality}")
        if c.threshold_operator:
            print(f"    * Threshold Operator: {c.threshold_operator}")
        print()

    # 3. Extract RAG context chunks
    chunks = extractor.extract_context_chunks(pdf_bytes)
    print(f"[+] Converted PDF into {len(chunks)} ready-to-index RAG context chunks.")
    print("==========================================================================")


if __name__ == "__main__":
    main()
