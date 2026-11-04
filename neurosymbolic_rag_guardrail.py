"""
Sciling PoC: Deterministic Neuro-Symbolic Fact-Graph & Deontic Logic RAG Guardrail.
Author: Juan Diego Ussa Aponte (Valencia, Spain)
Target Systems: Sciling Core (Multi-Agent Enterprise RAG) & BidBrief (Public Procurement Auditing)
Architecture: Pre-indexed Hybrid Lexical/Morphological Vector Space + First-Order Symbolic Fact Graph
              + Dimensional Unit Normalizer + Standard Deontic Logic Polarity Verifier (< 2 ms CPU).

Problem Statement:
Enterprise LLM pipelines in high-stakes domains (public procurement tenders in BidBrief, clinical/gerontological
monitoring in PREDATICS, and multi-agent data orchestration in Sciling Core) face five critical failure modes
that naive bag-of-words or bag-of-numbers checks fail to catch:
1. Relational Swap Hallucination: Context states 'Server A -> 14ms, Server B -> 92ms', and the LLM swaps them
   ('Server A -> 92ms, Server B -> 14ms'). Naive set subtraction (ans_nums - ctx_nums) returns empty set and
   falsely approves the hallucination.
2. Deontic & Negation Polarity Inversion: Context states 'prohíbe almacenar tokens' or 'obligatorio ENS', and
   the LLM outputs 'permite almacenar tokens' or 'exento de ENS'. Lexical cosine similarity exceeds 0.85,
   blinding purely vector-based guardrails.
3. Dimensional & Unit Equivalence False Positives: Context states '1.5 segundos', '50.000€', or '0.5g', and
   the LLM paraphrases as '1500ms', '50k€', or '500mg'. Naive regex flags a false positive.
4. Threshold & Inequality Bound Violations: Context sets a tender ceiling ('máximo 50.000€' or '< 30 minutos'),
   and the LLM validates an out-of-bounds figure ('65.000€') or flips the operator ('> 30 minutos').
5. Entity Fabrication & Unanchored Speculation: Introducing ungrounded proper nouns or speculative claims.

Using an external "LLM-as-a-judge" adds >1500 ms latency, recurring API/GPU cost, and non-deterministic
recursive hallucinations. This engine performs complete neuro-symbolic verification in ~0.5 - 1.5 ms on CPU.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from symbolic_fact_graph import (
    DeonticPolarityAnalyzer,
    EntityQuantityBinder,
    NumericalQuantity,
    SymbolicFactGraph,
    UnitNormalizer,
    strip_accents,
)


@dataclass
class _PreIndexedContext:
    """Cached neuro-symbolic representation of reference context chunks to avoid re-fitting per query."""
    context_tuple: Tuple[str, ...]
    fact_graph: SymbolicFactGraph
    all_context_tokens: FrozenSet[str]
    vectorizer: TfidfVectorizer
    context_tfidf_matrix: Any


class NeuroSymbolicRAGGuardrail:
    """
    Deterministic Neuro-Symbolic Guardrail combining:
    - Pre-indexed morphological + symbolic TF-IDF vector alignment.
    - First-order entity-attribute-quantity graph verification (SymbolicFactGraph).
    - Dimensional unit normalization and inequality bound algebra (UnitNormalizer).
    - Standard Deontic Logic (SDL) polarity verification (DeonticPolarityAnalyzer).
    """

    def __init__(self, semantic_alignment_threshold: float = 0.30):
        self.sim_threshold = semantic_alignment_threshold
        self._index_cache: Dict[Tuple[str, ...], _PreIndexedContext] = {}

    @staticmethod
    def _build_semantic_representation(
        text: str,
        precomputed: Optional[Tuple[List[NumericalQuantity], Set[str], List[str]]] = None,
    ) -> str:
        """
        Transforms raw text into a canonical neuro-symbolic feature string combining:
        - Normalized content words (stopwords removed, accents stripped).
        - Morphological root stems for inflection invariance (e.g., 'almacenar' == 'almacenamiento').
        - Canonical dimensional quantity tokens (e.g., '1.5 segundos' and '1500ms' -> '__qty_time_1.5_s__').
        - Canonical deontic state tokens (e.g., 'prohíbe' and 'no permite' -> '__deontic_forbidden__').
        """
        features: List[str] = []

        if precomputed is not None:
            quantities, entities, deontic_states = precomputed
            for q in quantities:
                features.append(q.canonical_tag())
                features.append(f"__dim_{q.dimension}__")
            for e in entities:
                features.append(e)
            for ds in deontic_states:
                features.append(f"__deontic_{ds.lower()}__")
        else:
            quantities = UnitNormalizer.extract_quantities(text)
            for q in quantities:
                features.append(q.canonical_tag())
                features.append(f"__dim_{q.dimension}__")

            clauses = EntityQuantityBinder.split_into_clauses(text)
            for cl in clauses:
                ents = EntityQuantityBinder.extract_clause_entities(cl)
                for e in ents:
                    features.append(e)
                sig = DeonticPolarityAnalyzer.analyze_clause(cl, ents)
                if sig.deontic_state != "AFFIRMATIVE":
                    features.append(f"__deontic_{sig.deontic_state.lower()}__")

        # 3. Extract content words and morphological stems
        raw_words = re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_\-]{2,}\b", text)
        for w in raw_words:
            norm = strip_accents(w.lower())
            if norm in DeonticPolarityAnalyzer.SPANISH_STOPWORDS or norm.isdigit():
                continue
            if re.match(r"^\d+(?:[.,]\d+)?[a-z%€$]*$", norm):
                continue
            features.append(norm)
            if len(norm) >= 5:
                features.append(DeonticPolarityAnalyzer.stem_word(norm))

        return " ".join(features) if features else strip_accents(text.lower())

    def index_context(self, context_chunks: List[str]) -> _PreIndexedContext:
        """Pre-indexes reference context chunks into a cached SymbolicFactGraph and TF-IDF matrix."""
        ctx_key = tuple(context_chunks)
        if ctx_key in self._index_cache:
            return self._index_cache[ctx_key]

        fact_graph = SymbolicFactGraph(context_chunks)
        all_tokens = frozenset(
            strip_accents(t.lower())
            for t in re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_\-]+\b", fact_graph.full_context)
        )

        # Index both full chunks and elementary clauses for fine-grained sentence alignment
        reference_units = list(context_chunks)
        for cl in fact_graph.context_clauses:
            if cl not in reference_units:
                reference_units.append(cl)

        norm_refs = [self._build_semantic_representation(u) for u in reference_units]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        ctx_matrix = vectorizer.fit_transform(norm_refs)
        # Warm up SciPy CSR matrix multiplication kernel during indexing
        _ = (ctx_matrix[:1] @ ctx_matrix.T).toarray()

        indexed = _PreIndexedContext(
            context_tuple=ctx_key,
            fact_graph=fact_graph,
            all_context_tokens=all_tokens,
            vectorizer=vectorizer,
            context_tfidf_matrix=ctx_matrix,
        )
        self._index_cache[ctx_key] = indexed
        return indexed

    def extract_numerical_facts(self, text: str) -> Set[str]:
        """
        Extracts canonical numerical quantities and raw normalized representations.
        Preserves backward-compatible interface while normalizing equivalent units.
        """
        quantities = UnitNormalizer.extract_quantities(text)
        results: Set[str] = set()
        for q in quantities:
            results.add(q.canonical_tag())
            results.add(q.raw_text.lower())
        return results

    def extract_named_entities(self, text: str, known_entities: Optional[Set[str]] = None) -> Set[str]:
        """Extracts proper nouns, technical acronyms, and indexed designator entities across clauses."""
        entities: Set[str] = set()
        for cl in EntityQuantityBinder.split_into_clauses(text):
            entities.update(EntityQuantityBinder.extract_clause_entities(cl, known_entities))
        return entities

    def evaluate_response(self, context_chunks: List[str], generated_answer: str) -> Dict[str, Any]:
        """
        Executes full neuro-symbolic verification of `generated_answer` against `context_chunks`:
        1. Input sanity validation (non-empty context and answer).
        2. Symbolic Fact Graph & Deontic Polarity audit (Relational Swaps, Threshold Bounds,
           Unit-Normalized Numerical Grounding, Deontic Square Contradictions).
        3. Named Entity Grounding audit (detecting fabricated proper nouns/acronyms).
        4. Pre-indexed Morphological + Symbolic Vector Alignment per sentence.
        """
        t0 = time.perf_counter()

        if not context_chunks or not any(c.strip() for c in context_chunks):
            exec_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "verdict": "REJECTED",
                "is_grounded": False,
                "violations_detected": 1,
                "violation_details": ["EMPTY_CONTEXT: No reference context chunks provided for grounding."],
                "violation_types": ["EMPTY_CONTEXT"],
                "mean_semantic_similarity": 0.0,
                "execution_time_ms": round(exec_ms, 3),
                "symbolic_triples_verified": 0,
            }

        if not generated_answer or not generated_answer.strip():
            exec_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "verdict": "REJECTED",
                "is_grounded": False,
                "violations_detected": 1,
                "violation_details": ["EMPTY_RESPONSE: Generated answer contains no verifiable claims."],
                "violation_types": ["EMPTY_RESPONSE"],
                "mean_semantic_similarity": 0.0,
                "execution_time_ms": round(exec_ms, 3),
                "symbolic_triples_verified": 0,
            }

        idx_ctx = self.index_context(context_chunks)
        fact_graph = idx_ctx.fact_graph

        violations: List[str] = []

        # 1. Single-pass Symbolic Fact Graph & Deontic Polarity Check
        (
            graph_violations,
            answer_entities,
            answer_quantities,
            answer_deontic_states,
        ) = fact_graph.verify_answer_with_metadata(generated_answer)
        violations.extend(graph_violations)

        # 2. Symbolic Named Entity Fabrication Check
        grounded_entity_vocab = set(fact_graph.known_entities) | set(idx_ctx.all_context_tokens)
        ungrounded_entities = sorted(answer_entities - grounded_entity_vocab)
        if ungrounded_entities:
            violations.append(
                f"ENTITY_FABRICATION: Entities {set(ungrounded_entities)} absent from source documents."
            )

        # 3. Pre-indexed Vector Semantic Alignment across sentences (L2-normalized sparse dot product)
        answer_sentences = [
            s.strip() for s in re.split(r"(?<!\d)[.!?]+(?!\d)|\n+", generated_answer) if len(s.strip()) > 3
        ]
        sentence_alignments: List[float] = []

        if answer_sentences:
            if len(answer_sentences) == 1:
                norm_ans_sents = [
                    self._build_semantic_representation(
                        answer_sentences[0],
                        precomputed=(answer_quantities, answer_entities, answer_deontic_states),
                    )
                ]
            else:
                norm_ans_sents = [self._build_semantic_representation(s) for s in answer_sentences]
            ans_vectors = idx_ctx.vectorizer.transform(norm_ans_sents)
            sim_matrix = (ans_vectors @ idx_ctx.context_tfidf_matrix.T).toarray()

            for idx, s in enumerate(answer_sentences):
                best_sim = float(np.max(sim_matrix[idx])) if sim_matrix.shape[1] > 0 else 0.0
                sentence_alignments.append(best_sim)
                if best_sim < self.sim_threshold:
                    violations.append(
                        f"UNANCHORED_SENTENCE: '{s[:60]}...' semantic score {best_sim:.2f} < {self.sim_threshold:.2f}"
                    )

        exec_time_ms = (time.perf_counter() - t0) * 1000.0
        mean_semantic_score = float(np.mean(sentence_alignments)) if sentence_alignments else 0.0
        is_approved = len(violations) == 0

        violation_types = []
        for v in violations:
            v_type = v.split(":", 1)[0].strip()
            if v_type not in violation_types:
                violation_types.append(v_type)

        return {
            "verdict": "APPROVED" if is_approved else "REJECTED",
            "is_grounded": is_approved,
            "violations_detected": len(violations),
            "violation_details": violations,
            "violation_types": violation_types,
            "mean_semantic_similarity": round(mean_semantic_score, 3),
            "execution_time_ms": round(exec_time_ms, 3),
            "symbolic_triples_verified": len(answer_quantities),
        }


def compute_classification_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes rigorous binary classification metrics for Guardrail Hallucination Detection:
    - Positive Class (1): Hallucinated / Ungrounded response -> Expected 'REJECTED'
    - Negative Class (0): Faithful / Grounded response      -> Expected 'APPROVED'
    Computes TP, TN, FP, FN, Precision, Recall, F1-Score, FPR, FNR, Accuracy, and
    Matthews Correlation Coefficient (MCC).
    """
    tp = sum(1 for r in results if r["expected"] == "REJECTED" and r["actual"] == "REJECTED")
    tn = sum(1 for r in results if r["expected"] == "APPROVED" and r["actual"] == "APPROVED")
    fp = sum(1 for r in results if r["expected"] == "APPROVED" and r["actual"] == "REJECTED")
    fn = sum(1 for r in results if r["expected"] == "REJECTED" and r["actual"] == "APPROVED")

    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / float(total)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / float(fn + tp) if (fn + tp) > 0 else 0.0

    mcc_denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    latencies = [r["latency_ms"] for r in results]
    p50_ms = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99_ms = float(np.percentile(latencies, 99)) if latencies else 0.0
    mean_ms = float(np.mean(latencies)) if latencies else 0.0

    return {
        "total_cases": total,
        "true_positives_rejected": tp,
        "true_negatives_approved": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mcc": round(mcc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "latency_mean_ms": round(mean_ms, 3),
        "latency_p50_ms": round(p50_ms, 3),
        "latency_p95_ms": round(p95_ms, 3),
        "latency_p99_ms": round(p99_ms, 3),
    }


def get_sciling_benchmark_suite() -> List[Dict[str, Any]]:
    """
    Returns a 24-case adversarial evaluation benchmark spanning Sciling's three core operational domains:
    1. BidBrief (Public Procurement & Tender Auditing)
    2. PREDATICS / Bio-Clinical (Biomedical NLP & Gerontological Monitoring)
    3. Sciling Core (Multi-Agent LLM Orchestration & Enterprise Security)
    """
    bidbrief_context = [
        "En el Lote 1 el presupuesto máximo de licitación es de 50.000€ y en el Lote 2 el presupuesto máximo es de 120.000€.",
        "El plazo de ejecución del contrato será de 2 meses y la garantía definitiva exigida es del 5% del importe de adjudicación.",
        "Es obligatorio presentar certificado ENS de nivel alto; las PYMEs están exentas de constituir garantía provisional.",
        "El tiempo de respuesta del soporte técnico ante incidencias críticas deberá ser inferior a 30 minutos.",
    ]

    bioclinical_context = [
        "La pauta terapéutica establece una dosis de Metformina de 500mg cada 12 horas, mientras que la dosis de Atorvastatina es de 20mg diarios.",
        "Está estrictamente contraindicado administrar Ibuprofeno en pacientes geriátricos con insuficiencia renal severa.",
        "El modelo Sciling-Bio-Llama alcanzó un F1-score del 92.4% en el benchmark clínico MedQA con 8000 muestras evaluadas.",
        "El agente predictivo de gerontología activa una alerta temprana cuando el umbral de riesgo geriátrico supera el 80%.",
    ]

    sciling_core_context = [
        "El servidor de inferencia vLLM opera con una latencia de 14ms por token utilizando cuantización AWQ en GPUs NVIDIA A100, mientras que el motor TGI registra una latencia de 92ms.",
        "La directiva de seguridad prohíbe explícitamente el almacenamiento de tokens de autenticación en texto plano en logs de producción.",
        "El orquestador multiagente Sciling Core procesa documentos heterogéneos con un tiempo de sincronización máximo de 1.5 segundos y 16GB de memoria VRAM reservada.",
    ]

    return [
        # --- DOMAIN 1: BidBrief (Public Procurement & Tenders) ---
        {
            "id": "BB-01",
            "domain": "BidBrief (Public Procurement)",
            "category": "Exact Grounding",
            "context": bidbrief_context,
            "answer": "En el Lote 1 el presupuesto máximo de licitación es de 50.000€ con una garantía definitiva exigida del 5%.",
            "expected": "APPROVED",
        },
        {
            "id": "BB-02",
            "domain": "BidBrief (Public Procurement)",
            "category": "Unit Conversion Equivalence (2 meses -> 60 días, 50.000€ -> 50k€)",
            "context": bidbrief_context,
            "answer": "El plazo de ejecución del contrato será de 60 días y en el Lote 1 el presupuesto máximo es de 50k€.",
            "expected": "APPROVED",
        },
        {
            "id": "BB-03",
            "domain": "BidBrief (Public Procurement)",
            "category": "Deontic Equivalence (exentas <=> no están obligadas)",
            "context": bidbrief_context,
            "answer": "Las PYMEs no están obligadas a constituir garantía provisional en la licitación.",
            "expected": "APPROVED",
        },
        {
            "id": "BB-04",
            "domain": "BidBrief (Public Procurement)",
            "category": "Inequality Bound Satisfaction (45.000€ <= 50.000€)",
            "context": bidbrief_context,
            "answer": "En el Lote 1 una propuesta con presupuesto de 45.000€ cumple el límite de licitación.",
            "expected": "APPROVED",
        },
        {
            "id": "BB-05",
            "domain": "BidBrief (Public Procurement)",
            "category": "Relational Swap Attack (Lote 1 <-> Lote 2 Budgets)",
            "context": bidbrief_context,
            "answer": "En el Lote 1 el presupuesto máximo de licitación es de 120.000€ y en el Lote 2 el presupuesto máximo es de 50.000€.",
            "expected": "REJECTED",
            "expected_violation_type": "RELATIONAL_SWAP_HALLUCINATION",
        },
        {
            "id": "BB-06",
            "domain": "BidBrief (Public Procurement)",
            "category": "Deontic Polarity Flip (obligatorio -> exento)",
            "context": bidbrief_context,
            "answer": "El licitador está exento de presentar certificado ENS de nivel alto.",
            "expected": "REJECTED",
            "expected_violation_type": "DEONTIC_POLARITY_INVERSION",
        },
        {
            "id": "BB-07",
            "domain": "BidBrief (Public Procurement)",
            "category": "Budget Threshold Violation (65.000€ > máximo 50.000€)",
            "context": bidbrief_context,
            "answer": "En el Lote 1 una propuesta con presupuesto de 65.000€ cumple el límite de licitación.",
            "expected": "REJECTED",
            "expected_violation_type": "THRESHOLD_BOUND_VIOLATION",
        },
        {
            "id": "BB-08",
            "domain": "BidBrief (Public Procurement)",
            "category": "Inequality Operator Flip (inferior a 30 min -> superior a 30 min)",
            "context": bidbrief_context,
            "answer": "El tiempo de respuesta del soporte técnico ante incidencias críticas deberá ser superior a 30 minutos.",
            "expected": "REJECTED",
            "expected_violation_type": "THRESHOLD_BOUND_VIOLATION",
        },
        # --- DOMAIN 2: PREDATICS / Bio-Clinical ---
        {
            "id": "BC-01",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Exact Clinical Grounding",
            "context": bioclinical_context,
            "answer": "El modelo Sciling-Bio-Llama alcanzó un F1-score del 92.4% en el benchmark clínico MedQA con 8000 muestras evaluadas.",
            "expected": "APPROVED",
        },
        {
            "id": "BC-02",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Unit Conversion Equivalence (500mg -> 0.5g)",
            "context": bioclinical_context,
            "answer": "La pauta terapéutica establece una dosis de Metformina de 0.5g cada 12 horas.",
            "expected": "APPROVED",
        },
        {
            "id": "BC-03",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Deontic Equivalence (contraindicado <=> no se permite)",
            "context": bioclinical_context,
            "answer": "No se permite administrar Ibuprofeno en pacientes geriátricos con insuficiencia renal severa.",
            "expected": "APPROVED",
        },
        {
            "id": "BC-04",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Inequality Bound Satisfaction (85% > 80% geriatric risk threshold)",
            "context": bioclinical_context,
            "answer": "El agente predictivo de gerontología activa una alerta temprana con un riesgo geriátrico del 85%.",
            "expected": "APPROVED",
        },
        {
            "id": "BC-05",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Relational Swap Attack (Metformina 20mg <-> Atorvastatina 500mg)",
            "context": bioclinical_context,
            "answer": "La pauta terapéutica establece una dosis de Metformina de 20mg cada 12 horas y la dosis de Atorvastatina es de 500mg diarios.",
            "expected": "REJECTED",
            "expected_violation_type": "RELATIONAL_SWAP_HALLUCINATION",
        },
        {
            "id": "BC-06",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Deontic Polarity Flip (contraindicado -> recomienda)",
            "context": bioclinical_context,
            "answer": "Se recomienda administrar Ibuprofeno en pacientes geriátricos con insuficiencia renal severa.",
            "expected": "REJECTED",
            "expected_violation_type": "DEONTIC_POLARITY_INVERSION",
        },
        {
            "id": "BC-07",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Numerical Hallucination (92.4% -> 98.5%)",
            "context": bioclinical_context,
            "answer": "El modelo Sciling-Bio-Llama alcanzó un F1-score del 98.5% en el benchmark clínico MedQA.",
            "expected": "REJECTED",
            "expected_violation_type": "CRITICAL_NUMERICAL_HALLUCINATION",
        },
        {
            "id": "BC-08",
            "domain": "PREDATICS / Bio-Clinical",
            "category": "Unanchored Clinical Extrapolation",
            "context": bioclinical_context,
            "answer": "Los cirujanos cardiovasculares sugieren realizar una tomografía axial computarizada cada semestre.",
            "expected": "REJECTED",
            "expected_violation_type": "UNANCHORED_SENTENCE",
        },
        # --- DOMAIN 3: Sciling Core (Multi-Agent Infrastructure & Security) ---
        {
            "id": "SC-01",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Exact Infrastructure Grounding",
            "context": sciling_core_context,
            "answer": "El servidor vLLM utiliza cuantización AWQ en GPUs NVIDIA A100 alcanzando una latencia de 14ms por token.",
            "expected": "APPROVED",
        },
        {
            "id": "SC-02",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Unit Conversion Equivalence (1.5 segundos -> 1500ms)",
            "context": sciling_core_context,
            "answer": "El orquestador multiagente Sciling Core procesa documentos heterogéneos con un tiempo de sincronización máximo de 1500ms.",
            "expected": "APPROVED",
        },
        {
            "id": "SC-03",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Deontic Equivalence (prohíbe <=> no permite) + Discourse Marker",
            "context": sciling_core_context,
            "answer": "Actualmente la directiva de seguridad no permite el almacenamiento de tokens de autenticación en texto plano en logs de producción.",
            "expected": "APPROVED",
        },
        {
            "id": "SC-04",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Relational Swap Attack (vLLM 92ms <-> TGI 14ms)",
            "context": sciling_core_context,
            "answer": "El servidor de inferencia vLLM opera con una latencia de 92ms por token y el motor TGI registra una latencia de 14ms.",
            "expected": "REJECTED",
            "expected_violation_type": "RELATIONAL_SWAP_HALLUCINATION",
        },
        {
            "id": "SC-05",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Deontic Polarity Flip (prohíbe -> permite plaintext tokens)",
            "context": sciling_core_context,
            "answer": "La directiva de seguridad permite explícitamente el almacenamiento de tokens de autenticación en texto plano en logs de producción.",
            "expected": "REJECTED",
            "expected_violation_type": "DEONTIC_POLARITY_INVERSION",
        },
        {
            "id": "SC-06",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Numerical Hallucination (14ms -> 45ms)",
            "context": sciling_core_context,
            "answer": "El servidor vLLM opera con una latencia de 45ms por token en GPUs NVIDIA A100.",
            "expected": "REJECTED",
            "expected_violation_type": "CRITICAL_NUMERICAL_HALLUCINATION",
        },
        {
            "id": "SC-07",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Entity Fabrication (Google TPUv4 & OpenAI Whisper)",
            "context": sciling_core_context,
            "answer": "El servidor vLLM fue entrenado utilizando Google TPUv4 y optimizado con OpenAI Whisper.",
            "expected": "REJECTED",
            "expected_violation_type": "ENTITY_FABRICATION",
        },
        {
            "id": "SC-08",
            "domain": "Sciling Core (Multi-Agent LLM)",
            "category": "Unanchored Infrastructure Speculation",
            "context": sciling_core_context,
            "answer": "Los administradores de sistemas recomiendan migrar toda la infraestructura a Kubernetes antes del próximo trimestre.",
            "expected": "REJECTED",
            "expected_violation_type": "UNANCHORED_SENTENCE",
        },
    ]


def run_benchmark(export_json_path: Optional[str] = None) -> Dict[str, Any]:
    """Executes the 24-case Sciling Neuro-Symbolic Guardrail benchmark and exports telemetry."""
    print("==========================================================================================")
    print("SCILING BENCHMARK: NEURO-SYMBOLIC FACT-GRAPH & DEONTIC LOGIC GUARDRAIL (v2.0)")
    print("Domains: BidBrief (Public Tenders) | PREDATICS (Bio-Clinical) | Sciling Core (Multi-Agent)")
    print("==========================================================================================")

    guardrail = NeuroSymbolicRAGGuardrail(semantic_alignment_threshold=0.30)
    suite = get_sciling_benchmark_suite()

    # Warm-up context indices once (simulating document ingestion phase in RAG pipeline)
    unique_contexts = {tuple(case["context"]): case["context"] for case in suite}
    for ctx_list in unique_contexts.values():
        guardrail.index_context(ctx_list)

    print(f"[*] Pre-indexed {len(unique_contexts)} domain knowledge bases into Symbolic Fact Graphs.")
    print(f"[*] Evaluating {len(suite)} adversarial & grounded test cases on CPU...\n")

    case_records: List[Dict[str, Any]] = []
    passed_count = 0

    for case in suite:
        eval_res = guardrail.evaluate_response(case["context"], case["answer"])
        verdict_match = eval_res["verdict"] == case["expected"]
        expected_vtype = case.get("expected_violation_type")
        vtype_match = True
        if expected_vtype:
            vtype_match = expected_vtype in eval_res["violation_types"]

        passed = verdict_match and vtype_match
        if passed:
            passed_count += 1
            status_tag = "[PASS]"
        else:
            status_tag = "[FAIL]"

        print(
            f"{status_tag} [{case['id']}] {case['domain']} — {case['category']}\n"
            f"       Verdict: {eval_res['verdict']:8s} (Expected: {case['expected']:8s}) | "
            f"Sim: {eval_res['mean_semantic_similarity']:.2f} | Latency: {eval_res['execution_time_ms']:.3f} ms"
        )
        if eval_res["violations_detected"] > 0:
            for v in eval_res["violation_details"]:
                print(f"       ! {v}")

        case_records.append({
            "id": case["id"],
            "domain": case["domain"],
            "category": case["category"],
            "expected": case["expected"],
            "actual": eval_res["verdict"],
            "passed": passed,
            "latency_ms": eval_res["execution_time_ms"],
            "mean_semantic_similarity": eval_res["mean_semantic_similarity"],
            "violation_types": eval_res["violation_types"],
            "violation_details": eval_res["violation_details"],
        })

    metrics = compute_classification_metrics(case_records)

    print("\n==========================================================================================")
    print("SCIENTIFIC EVALUATION METRICS (PRHLT / PRODUCTION AUDIT STANDARD)")
    print("==========================================================================================")
    print(f"[+] Test Suite Pass Rate          : {passed_count}/{len(suite)} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"[+] Matthews Correlation (MCC)    : {metrics['mcc']:.4f}")
    print(f"[+] F1-Score / Precision / Recall : {metrics['f1_score']:.4f} / {metrics['precision']:.4f} / {metrics['recall']:.4f}")
    print(f"[+] Confusion Matrix (TP/TN/FP/FN): TP={metrics['true_positives_rejected']} | TN={metrics['true_negatives_approved']} | FP={metrics['false_positives']} | FN={metrics['false_negatives']}")
    print(f"[+] False Positive / Negative Rate: FPR={metrics['false_positive_rate']:.4f} | FNR={metrics['false_negative_rate']:.4f}")
    print(f"[+] CPU Latency (Mean / P50 / P99): {metrics['latency_mean_ms']:.3f} ms / {metrics['latency_p50_ms']:.3f} ms / {metrics['latency_p99_ms']:.3f} ms")
    speedup = 1500.0 / max(0.001, metrics["latency_mean_ms"])
    print(f"[+] Speedup vs LLM-as-a-Judge     : > {speedup:,.0f}x faster than 1500 ms neural judge baseline")
    print("==========================================================================================")

    telemetry_payload = {
        "engine": "Sciling Neuro-Symbolic Fact-Graph & Deontic Guardrail v2.0",
        "author": "Juan Diego Ussa Aponte",
        "metrics": metrics,
        "cases": case_records,
    }

    if export_json_path is None:
        export_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_telemetry.json")

    with open(export_json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_payload, f, indent=2, ensure_ascii=False)

    print(f"[V] Forensic telemetry exported to: {export_json_path}")
    return telemetry_payload


if __name__ == "__main__":
    run_benchmark()
