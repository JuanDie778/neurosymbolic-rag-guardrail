"""
Automated Pytest Verification Suite for Sciling Neuro-Symbolic Fact-Graph & Deontic Guardrail.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Tests cover:
1. Dimensional Unit Normalizer (time, currency, mass, percentage, ES/EN separators, calendar units).
2. Inequality & Threshold Bound Algebra (<, <=, >, >=, operator flips, point compliance vs violation).
3. Standard Deontic Logic (SDL) Square of Opposition (Prohibition <=> ~Permission, Obligation vs Exemption, double negation).
4. Predicate-Argument Relational Swap Detection (Server A/B, Lote 1/2, Clinical Dosages).
5. Edge Cases (empty context, empty answer, numberless text, interval ranges 'entre X y Y', capitalization invariance).
6. End-to-End 24-Case Multi-Domain Benchmark (MCC == 1.00, Latency < 5 ms on CPU).
"""

import os
import sys
import pytest

# Ensure local module resolution regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symbolic_fact_graph import (
    DeonticPolarityAnalyzer,
    EntityQuantityBinder,
    SymbolicFactGraph,
    UnitNormalizer,
)
from neurosymbolic_rag_guardrail import (
    NeuroSymbolicRAGGuardrail,
    compute_classification_metrics,
    get_sciling_benchmark_suite,
)


# ============================================================================
# 1. DIMENSIONAL UNIT NORMALIZATION & ES/EN NUMBER FORMATTING TESTS
# ============================================================================

def test_unit_normalizer_time_equivalences():
    """Verifies cross-unit equivalence across seconds, milliseconds, minutes, days, months, and years."""
    q_sec = UnitNormalizer.extract_quantities("El límite es de 1.5 segundos")[0]
    q_ms = UnitNormalizer.extract_quantities("El límite es de 1500 ms")[0]
    q_ms_comma = UnitNormalizer.extract_quantities("El límite es de 1,500ms")[0]
    q_wrong = UnitNormalizer.extract_quantities("El límite es de 450 ms")[0]

    assert UnitNormalizer.are_equivalent(q_sec, q_ms)
    assert UnitNormalizer.are_equivalent(q_sec, q_ms_comma)
    assert not UnitNormalizer.are_equivalent(q_sec, q_wrong)

    # Public procurement calendar conversions (2 meses == 60 días, 1 año == 12 meses)
    q_months = UnitNormalizer.extract_quantities("Plazo de 2 meses")[0]
    q_days = UnitNormalizer.extract_quantities("Plazo de 60 días")[0]
    q_year = UnitNormalizer.extract_quantities("Duración de 1 año")[0]
    q_12m = UnitNormalizer.extract_quantities("Duración de 12 meses")[0]

    assert UnitNormalizer.are_equivalent(q_months, q_days)
    assert UnitNormalizer.are_equivalent(q_year, q_12m)


def test_unit_normalizer_currency_and_separators():
    """Verifies Spanish/English thousands/decimal separators and k€/M€ multipliers."""
    q_es = UnitNormalizer.extract_quantities("Presupuesto de 50.000€")[0]
    q_k = UnitNormalizer.extract_quantities("Presupuesto de 50k€")[0]
    q_en = UnitNormalizer.extract_quantities("Presupuesto de 50,000 EUR")[0]
    q_mil = UnitNormalizer.extract_quantities("Presupuesto de 50 mil euros")[0]

    assert q_es.canonical_value == pytest.approx(50000.0)
    assert UnitNormalizer.are_equivalent(q_es, q_k)
    assert UnitNormalizer.are_equivalent(q_es, q_en)
    assert UnitNormalizer.are_equivalent(q_es, q_mil)

    # Millions
    q_m1 = UnitNormalizer.extract_quantities("Contrato de 2.5M€")[0]
    q_m2 = UnitNormalizer.extract_quantities("Contrato de 2.500.000 euros")[0]
    q_m3 = UnitNormalizer.extract_quantities("Contrato de 2,5 millones de euros")[0]
    assert UnitNormalizer.are_equivalent(q_m1, q_m2)
    assert UnitNormalizer.are_equivalent(q_m1, q_m3)


def test_unit_normalizer_mass_and_percentage():
    """Verifies clinical mass conversions (mg/g/mcg) and percentage-to-ratio equivalence."""
    q_mg = UnitNormalizer.extract_quantities("Dosis de 500mg")[0]
    q_g = UnitNormalizer.extract_quantities("Dosis de 0.5g")[0]
    q_g_es = UnitNormalizer.extract_quantities("Dosis de 0,5 gramos")[0]
    assert UnitNormalizer.are_equivalent(q_mg, q_g)
    assert UnitNormalizer.are_equivalent(q_mg, q_g_es)

    q_pct = UnitNormalizer.extract_quantities("Tasa del 15%")[0]
    q_pct_words = UnitNormalizer.extract_quantities("Tasa del 15 por ciento")[0]
    q_ratio = UnitNormalizer.extract_quantities("Ratio de 0.15")[0]
    q_wrong_pct = UnitNormalizer.extract_quantities("Tasa del 45%")[0]

    assert UnitNormalizer.are_equivalent(q_pct, q_pct_words)
    assert UnitNormalizer.are_equivalent(q_pct, q_ratio)
    assert not UnitNormalizer.are_equivalent(q_pct, q_wrong_pct)


def test_entity_designator_numbers_excluded_from_quantities():
    """Ensures entity designators like 'Lote 1', 'Fase 2', 'Cláusula 4' are not extracted as bare numbers."""
    quantities = UnitNormalizer.extract_quantities(
        "En el Lote 1 y la Fase 2 según la Cláusula 4 el importe es de 50.000€."
    )
    assert len(quantities) == 1
    assert quantities[0].canonical_value == pytest.approx(50000.0)


# ============================================================================
# 2. INEQUALITY & THRESHOLD BOUND ALGEBRA TESTS
# ============================================================================

def test_inequality_bound_detection_and_satisfaction():
    """Verifies <, <=, >, >= operator extraction and bound compliance checking."""
    q_max_budget = UnitNormalizer.extract_quantities("El presupuesto máximo del Lote 1 es de 50.000€")[0]
    assert q_max_budget.bound_operator == "<="

    q_valid_offer = UnitNormalizer.extract_quantities("Una oferta de 45.000€")[0]
    q_exact_offer = UnitNormalizer.extract_quantities("Una oferta de 50k€")[0]
    q_invalid_offer = UnitNormalizer.extract_quantities("Una oferta de 65.000€")[0]
    q_flipped_op = UnitNormalizer.extract_quantities("El presupuesto mínimo es de 50.000€")[0]

    assert UnitNormalizer.satisfies_bound(q_max_budget, q_valid_offer)
    assert UnitNormalizer.satisfies_bound(q_max_budget, q_exact_offer)
    assert not UnitNormalizer.satisfies_bound(q_max_budget, q_invalid_offer)
    assert not UnitNormalizer.satisfies_bound(q_max_budget, q_flipped_op)

    # Strict '<' bound
    q_strict_lat = UnitNormalizer.extract_quantities("Latencia inferior a 20ms")[0]
    assert q_strict_lat.bound_operator == "<"
    q_lat_15 = UnitNormalizer.extract_quantities("Latencia de 15ms")[0]
    q_lat_20 = UnitNormalizer.extract_quantities("Latencia de 20ms")[0]
    q_lat_gt20 = UnitNormalizer.extract_quantities("Latencia superior a 20ms")[0]

    assert UnitNormalizer.satisfies_bound(q_strict_lat, q_lat_15)
    assert not UnitNormalizer.satisfies_bound(q_strict_lat, q_lat_20)
    assert not UnitNormalizer.satisfies_bound(q_strict_lat, q_lat_gt20)


# ============================================================================
# 3. STANDARD DEONTIC LOGIC & NEGATION POLARITY TESTS
# ============================================================================

def test_deontic_square_of_opposition_equivalences_and_contradictions():
    """
    Tests von Wright's Deontic Square:
    - FORBIDDEN ('prohíbe') <=> ~PERMITTED ('no permite') [Consistent]
    - FORBIDDEN ('prohíbe') vs PERMITTED ('permite')      [Contradictory]
    - OBLIGATORY ('obligatorio') vs EXEMPT ('exento')     [Contradictory]
    - EXEMPT ('exento') <=> ~OBLIGATORY ('no está obligado') [Consistent]
    """
    sig_prohibe = DeonticPolarityAnalyzer.analyze_clause(
        "La directiva prohíbe almacenar tokens en texto plano"
    )
    sig_no_permite = DeonticPolarityAnalyzer.analyze_clause(
        "La directiva no permite almacenar tokens en texto plano"
    )
    sig_permite = DeonticPolarityAnalyzer.analyze_clause(
        "La directiva permite almacenar tokens en texto plano"
    )

    assert sig_prohibe.deontic_state == "FORBIDDEN"
    assert sig_no_permite.deontic_state == "FORBIDDEN"
    assert sig_permite.deontic_state == "PERMITTED"

    assert not DeonticPolarityAnalyzer.are_contradictory(sig_prohibe, sig_no_permite)
    assert DeonticPolarityAnalyzer.are_contradictory(sig_prohibe, sig_permite)

    sig_oblig = DeonticPolarityAnalyzer.analyze_clause("Es obligatorio presentar garantía definitiva")
    sig_exento = DeonticPolarityAnalyzer.analyze_clause("Está exento de presentar garantía definitiva")
    sig_no_oblig = DeonticPolarityAnalyzer.analyze_clause("No es obligatorio presentar garantía definitiva")

    assert sig_oblig.deontic_state == "OBLIGATORY"
    assert sig_exento.deontic_state == "EXEMPT"
    assert sig_no_oblig.deontic_state == "EXEMPT"

    assert DeonticPolarityAnalyzer.are_contradictory(sig_oblig, sig_exento)
    assert not DeonticPolarityAnalyzer.are_contradictory(sig_exento, sig_no_oblig)


def test_deontic_double_negation():
    """Verifies double negation resolution ('no está prohibido' -> PERMITTED/POSITIVE)."""
    sig_not_forbidden = DeonticPolarityAnalyzer.analyze_clause("No está prohibido el uso de cuantización AWQ")
    assert sig_not_forbidden.polarity == "POSITIVE"
    assert sig_not_forbidden.deontic_state == "PERMITTED"

    sig_not_exempt = DeonticPolarityAnalyzer.analyze_clause("El adjudicatario no está exento de presentar aval")
    assert sig_not_exempt.polarity == "POSITIVE"
    assert sig_not_exempt.deontic_state == "OBLIGATORY"


# ============================================================================
# 4. RELATIONAL SWAP & PREDICATE-ARGUMENT BINDING TESTS
# ============================================================================

def test_relational_swap_detection_on_servers():
    """
    Tests the exact failure mode where naive set subtraction fails:
    Context: Server A -> 14ms, Server B -> 92ms.
    Answer : Server A -> 92ms, Server B -> 14ms.
    """
    guardrail = NeuroSymbolicRAGGuardrail()
    context = ["El servidor A tiene 14ms de latencia y el servidor B tiene 92ms."]

    faithful = "El servidor A tiene 14ms de latencia y el servidor B tiene 92ms."
    swapped = "El servidor A tiene 92ms de latencia y el servidor B tiene 14ms."

    res_ok = guardrail.evaluate_response(context, faithful)
    res_swap = guardrail.evaluate_response(context, swapped)

    assert res_ok["verdict"] == "APPROVED"
    assert res_swap["verdict"] == "REJECTED"
    assert "RELATIONAL_SWAP_HALLUCINATION" in res_swap["violation_types"]


def test_relational_swap_detection_on_tender_lots():
    """Verifies detection of swapped budgets between Lote 1 and Lote 2 in public tenders."""
    guardrail = NeuroSymbolicRAGGuardrail()
    context = ["En el Lote 1 el presupuesto es de 50.000€ y en el Lote 2 el presupuesto es de 120.000€."]

    swapped = "En el Lote 1 el presupuesto es de 120.000€ y en el Lote 2 el presupuesto es de 50.000€."
    res = guardrail.evaluate_response(context, swapped)

    assert res["verdict"] == "REJECTED"
    assert "RELATIONAL_SWAP_HALLUCINATION" in res["violation_types"]


def test_interval_range_not_mistaken_for_two_clauses():
    """Ensures 'entre 10ms y 20ms' stays in a single clause and binds both endpoints to the subject."""
    clauses = EntityQuantityBinder.split_into_clauses(
        "El servidor vLLM mantiene una latencia entre 10ms y 20ms en producción."
    )
    assert len(clauses) == 1


# ============================================================================
# 5. EDGE CASES & DEFENSIVE ROBUSTNESS TESTS
# ============================================================================

def test_empty_inputs_rejected_gracefully():
    """Verifies that empty or whitespace-only context or answer is deterministically rejected."""
    guardrail = NeuroSymbolicRAGGuardrail()
    ctx = ["El servidor vLLM opera a 14ms."]

    res_empty_ans = guardrail.evaluate_response(ctx, "   ")
    assert res_empty_ans["verdict"] == "REJECTED"
    assert "EMPTY_RESPONSE" in res_empty_ans["violation_types"]

    res_empty_ctx = guardrail.evaluate_response([], "El servidor vLLM opera a 14ms.")
    assert res_empty_ctx["verdict"] == "REJECTED"
    assert "EMPTY_CONTEXT" in res_empty_ctx["violation_types"]


def test_qualitative_text_without_numbers():
    """Verifies grounding, deontic flip, and entity fabrication when text contains zero numbers."""
    guardrail = NeuroSymbolicRAGGuardrail()
    ctx = ["La arquitectura Sciling Core coordina agentes sobre documentos PDF y bases SQL."]

    ans_ok = "La arquitectura Sciling Core coordina agentes sobre documentos PDF y bases SQL."
    ans_flip = "La arquitectura Sciling Core prohíbe coordinar agentes sobre documentos PDF y bases SQL."
    ans_fab = "La arquitectura Sciling Core coordina agentes sobre MongoDB y Snowflake."

    assert guardrail.evaluate_response(ctx, ans_ok)["verdict"] == "APPROVED"

    res_flip = guardrail.evaluate_response(ctx, ans_flip)
    assert res_flip["verdict"] == "REJECTED"
    assert "DEONTIC_POLARITY_INVERSION" in res_flip["violation_types"]

    res_fab = guardrail.evaluate_response(ctx, ans_fab)
    assert res_fab["verdict"] == "REJECTED"
    assert "ENTITY_FABRICATION" in res_fab["violation_types"]


def test_mid_sentence_capitalization_of_context_word_not_flagged_as_fabricated():
    """Ensures capitalizing a common noun present in lowercase in the context does not trigger ENTITY_FABRICATION."""
    guardrail = NeuroSymbolicRAGGuardrail()
    ctx = ["El servidor de inferencia opera con cuantización en producción."]
    ans = "El Servidor de Inferencia opera con cuantización en Producción."

    res = guardrail.evaluate_response(ctx, ans)
    assert res["verdict"] == "APPROVED"
    assert "ENTITY_FABRICATION" not in res["violation_types"]


def test_short_single_stem_deontic_contradiction():
    """Ensures short clauses with a single proposition stem ('Se prohíbe el MD5' vs 'Se permite el MD5') are caught."""
    guardrail = NeuroSymbolicRAGGuardrail()
    ctx = ["Se prohíbe el MD5."]
    ans_bad = "Se permite el MD5."
    ans_good = "No se permite el MD5."

    res_bad = guardrail.evaluate_response(ctx, ans_bad)
    assert res_bad["verdict"] == "REJECTED"
    assert "DEONTIC_POLARITY_INVERSION" in res_bad["violation_types"]

    res_good = guardrail.evaluate_response(ctx, ans_good)
    assert res_good["verdict"] == "APPROVED"


# ============================================================================
# 6. FULL 24-CASE BENCHMARK MCC = 1.00 & LATENCY SLA VERIFICATION
# ============================================================================

def test_full_sciling_benchmark_mcc_and_latency_sla():
    """
    Executes the complete 24-case adversarial benchmark across BidBrief, PREDATICS, and Sciling Core.
    Asserts MCC == 1.0000, F1 == 1.0000, FPR == 0.0, FNR == 0.0, and mean CPU latency < 3.5 ms.
    """
    guardrail = NeuroSymbolicRAGGuardrail(semantic_alignment_threshold=0.30)
    suite = get_sciling_benchmark_suite()

    for case in suite:
        guardrail.index_context(case["context"])

    records = []
    for case in suite:
        res = guardrail.evaluate_response(case["context"], case["answer"])
        assert res["verdict"] == case["expected"], f"Case {case['id']} failed: {res}"
        if "expected_violation_type" in case:
            assert case["expected_violation_type"] in res["violation_types"], (
                f"Case {case['id']} missing expected violation {case['expected_violation_type']}: {res}"
            )
        records.append({
            "expected": case["expected"],
            "actual": res["verdict"],
            "latency_ms": res["execution_time_ms"],
        })

    metrics = compute_classification_metrics(records)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["false_negative_rate"] == pytest.approx(0.0)
    assert metrics["latency_mean_ms"] < 3.5
    assert metrics["latency_p50_ms"] < 3.0


def test_relational_swap_detection_on_clinical_dosages():
    """Verifies relational swap detection when clinical drug dosages are inverted across clauses."""
    guardrail = NeuroSymbolicRAGGuardrail()
    context = [
        "La pauta terapéutica establece una dosis de Metformina de 500mg cada 12 horas, "
        "mientras que la dosis de Atorvastatina es de 20mg diarios."
    ]
    swapped = "La dosis de Metformina es de 20mg y la dosis de Atorvastatina es de 500mg."
    res = guardrail.evaluate_response(context, swapped)
    assert res["verdict"] == "REJECTED"
    assert "RELATIONAL_SWAP_HALLUCINATION" in res["violation_types"]


def test_multi_sentence_partial_hallucination_rejected():
    """Ensures a multi-sentence response where sentence 1 is grounded and sentence 2 hallucinates is REJECTED."""
    guardrail = NeuroSymbolicRAGGuardrail()
    context = ["El servidor vLLM opera con una latencia de 14ms por token en GPUs NVIDIA A100."]
    multi_ans = (
        "El servidor vLLM opera con una latencia de 14ms por token en GPUs NVIDIA A100. "
        "Sin embargo, los consultores externos exigen migrar el clúster a Amazon Bedrock mañana."
    )
    res = guardrail.evaluate_response(context, multi_ans)
    assert res["verdict"] == "REJECTED"
    assert res["violations_detected"] >= 1


def test_data_size_and_frequency_unit_conversions():
    """Verifies GB/MB and GHz/MHz dimensional conversions in UnitNormalizer."""
    q_gb = UnitNormalizer.extract_quantities("Memoria de 1.5 GB")[0]
    q_mb = UnitNormalizer.extract_quantities("Memoria de 1500 MB")[0]
    assert UnitNormalizer.are_equivalent(q_gb, q_mb)

    q_ghz = UnitNormalizer.extract_quantities("Reloj a 2.4 GHz")[0]
    q_mhz = UnitNormalizer.extract_quantities("Reloj a 2400 MHz")[0]
    assert UnitNormalizer.are_equivalent(q_ghz, q_mhz)

