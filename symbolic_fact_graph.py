"""
Symbolic Fact Graph, Dimensional Unit Normalizer, Entity-Quantity Binder & Deontic Logic Engine.
Author: Juan Diego Ussa Aponte (Valencia, Spain)
Target Architecture: Sciling Core (Multi-Agent RAG) & BidBrief (Public Procurement Auditing)

Mathematical & Logical Foundations:
1. Dimensional Unit Normalization (UnitNormalizer):
   Maps heterogeneous surface representations (ES/EN number formatting, SI prefixes, currency
   abbreviations, time/mass/data units, and inequality operators <, <=, >, >=) into canonical
   base-unit quantities with epsilon-tolerance and bound-satisfaction algebra.

2. Syntactic-Positional Predicate-Argument Binding (EntityQuantityBinder):
   Extracts local Knowledge Graph triples (Subject, Predicate, Quantity, Polarity, DeonticState)
   at the elementary clause level. Detects Relational Swap Hallucinations where all numbers exist
   in the retrieved context as a bag-of-numbers, but their subject bindings are inverted
   (e.g., Context: A->14ms, B->92ms vs Answer: A->92ms, B->14ms).

3. Standard Deontic Logic & Polarity Verification (DeonticPolarityAnalyzer):
   Models the Deontic Square of Opposition (von Wright, 1951):
   - Forbidden(p) <=> ~Permitted(p) <=> Obligatory(~p)  [e.g., 'prohíbe' == 'no permite']
   - Exempt(p)    <=> ~Obligatory(p)                    [e.g., 'exento' == 'no exige']
   Detects Deontic Polarity Flips (e.g., 'prohíbe' -> 'permite', 'obligatorio' -> 'exento',
   'contraindicado' -> 'recomendado') even when surface lexical cosine similarity exceeds 0.90.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


def strip_accents(text: str) -> str:
    """Removes diacritical marks from Unicode text while preserving base characters."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


@dataclass(frozen=True)
class NumericalQuantity:
    """Canonical representation of a dimensional or scalar quantity with optional inequality bound."""
    raw_text: str
    raw_value: float
    raw_unit: str
    canonical_value: float
    canonical_unit: str
    dimension: str
    bound_operator: str = "=="  # One of: "==", "<=", "<", ">=", ">"
    span: Tuple[int, int] = (0, 0)

    def canonical_tag(self) -> str:
        """Generates a deterministic symbolic token for vector-space grounding injection."""
        if self.dimension == "percentage":
            rounded = f"{self.canonical_value:.4f}".rstrip("0").rstrip(".")
        elif abs(self.canonical_value) >= 100.0 or self.canonical_value == int(self.canonical_value):
            rounded = f"{int(round(self.canonical_value))}"
        else:
            rounded = f"{self.canonical_value:.4f}".rstrip("0").rstrip(".")
        return f"__qty_{self.dimension}_{rounded}_{self.canonical_unit}__"


@dataclass(frozen=True)
class FactTriple:
    """First-Order Knowledge Graph triple binding an entity subject, attribute predicate, and quantity."""
    primary_subject: str
    clause_entities: FrozenSet[str]
    predicate: str
    quantity: NumericalQuantity
    polarity: str  # "POSITIVE" or "NEGATIVE"
    deontic_state: str  # "AFFIRMATIVE", "NEGATED", "OBLIGATORY", "PERMITTED", "FORBIDDEN", "EXEMPT"
    source_clause: str


@dataclass(frozen=True)
class ClauseDeonticSignature:
    """Deontic and negation modality profile of an elementary proposition clause."""
    clause_text: str
    polarity: str  # "POSITIVE" or "NEGATIVE"
    deontic_state: str  # "AFFIRMATIVE", "NEGATED", "OBLIGATORY", "PERMITTED", "FORBIDDEN", "EXEMPT"
    trigger_tokens: Tuple[str, ...]
    content_stems: FrozenSet[str]
    entities: FrozenSet[str]


class UnitNormalizer:
    """
    Normalizes numerical expressions and physical/financial/clinical units into canonical base units.
    Supports Spanish and English decimal/thousands separators, multiplier suffixes (k, M, mil, millones),
    and inequality bound operators (<, <=, >, >=).
    """

    # Base unit definitions: unit_token -> (dimension, canonical_unit, multiplier_to_canonical)
    UNIT_REGISTRY: Dict[str, Tuple[str, str, float]] = {
        # Time -> canonical: seconds ("s")
        "us": ("time", "s", 1e-6),
        "µs": ("time", "s", 1e-6),
        "microsegundo": ("time", "s", 1e-6),
        "microsegundos": ("time", "s", 1e-6),
        "ms": ("time", "s", 1e-3),
        "miliseg": ("time", "s", 1e-3),
        "milisegundo": ("time", "s", 1e-3),
        "milisegundos": ("time", "s", 1e-3),
        "s": ("time", "s", 1.0),
        "sec": ("time", "s", 1.0),
        "secs": ("time", "s", 1.0),
        "seg": ("time", "s", 1.0),
        "segs": ("time", "s", 1.0),
        "segundo": ("time", "s", 1.0),
        "segundos": ("time", "s", 1.0),
        "min": ("time", "s", 60.0),
        "mins": ("time", "s", 60.0),
        "minuto": ("time", "s", 60.0),
        "minutos": ("time", "s", 60.0),
        "h": ("time", "s", 3600.0),
        "hr": ("time", "s", 3600.0),
        "hrs": ("time", "s", 3600.0),
        "hora": ("time", "s", 3600.0),
        "horas": ("time", "s", 3600.0),
        "d": ("time", "s", 86400.0),
        "dia": ("time", "s", 86400.0),
        "dias": ("time", "s", 86400.0),
        "día": ("time", "s", 86400.0),
        "días": ("time", "s", 86400.0),
        "semana": ("time", "s", 604800.0),
        "semanas": ("time", "s", 604800.0),
        "mes": ("time", "s", 2592000.0),
        "meses": ("time", "s", 2592000.0),
        "año": ("time", "s", 31536000.0),
        "años": ("time", "s", 31536000.0),
        "ano": ("time", "s", 31536000.0),
        "anos": ("time", "s", 31536000.0),
        # Currency -> canonical: "eur" or "usd"
        "€": ("currency", "eur", 1.0),
        "eur": ("currency", "eur", 1.0),
        "euro": ("currency", "eur", 1.0),
        "euros": ("currency", "eur", 1.0),
        "k€": ("currency", "eur", 1e3),
        "keur": ("currency", "eur", 1e3),
        "m€": ("currency", "eur", 1e6),
        "meur": ("currency", "eur", 1e6),
        "mm€": ("currency", "eur", 1e6),
        "$": ("currency", "usd", 1.0),
        "usd": ("currency", "usd", 1.0),
        "dolar": ("currency", "usd", 1.0),
        "dolares": ("currency", "usd", 1.0),
        "dólar": ("currency", "usd", 1.0),
        "dólares": ("currency", "usd", 1.0),
        "k$": ("currency", "usd", 1e3),
        "kusd": ("currency", "usd", 1e3),
        "m$": ("currency", "usd", 1e6),
        "musd": ("currency", "usd", 1e6),
        # Percentage -> canonical: dimensionless ratio ("ratio")
        "%": ("percentage", "ratio", 0.01),
        "pct": ("percentage", "ratio", 0.01),
        "porciento": ("percentage", "ratio", 0.01),
        "por ciento": ("percentage", "ratio", 0.01),
        # Mass / Clinical dosage -> canonical: grams ("g")
        "mcg": ("mass", "g", 1e-6),
        "ug": ("mass", "g", 1e-6),
        "µg": ("mass", "g", 1e-6),
        "microgramo": ("mass", "g", 1e-6),
        "microgramos": ("mass", "g", 1e-6),
        "mg": ("mass", "g", 1e-3),
        "miligramo": ("mass", "g", 1e-3),
        "miligramos": ("mass", "g", 1e-3),
        "g": ("mass", "g", 1.0),
        "gr": ("mass", "g", 1.0),
        "gramo": ("mass", "g", 1.0),
        "gramos": ("mass", "g", 1.0),
        "kg": ("mass", "g", 1e3),
        "kilo": ("mass", "g", 1e3),
        "kilos": ("mass", "g", 1e3),
        "kilogramo": ("mass", "g", 1e3),
        "kilogramos": ("mass", "g", 1e3),
        # Data size -> canonical: "bytes"
        "byte": ("data_size", "bytes", 1.0),
        "bytes": ("data_size", "bytes", 1.0),
        "kb": ("data_size", "bytes", 1e3),
        "mb": ("data_size", "bytes", 1e6),
        "gb": ("data_size", "bytes", 1e9),
        "tb": ("data_size", "bytes", 1e12),
        # Frequency -> canonical: "hz"
        "hz": ("frequency", "hz", 1.0),
        "khz": ("frequency", "hz", 1e3),
        "mhz": ("frequency", "hz", 1e6),
        "ghz": ("frequency", "hz", 1e9),
    }

    # Designator nouns whose trailing integer/letter is an entity ID (e.g., "Lote 1", "Fase 2"), not a measurement
    ENTITY_DESIGNATOR_NOUNS: FrozenSet[str] = frozenset({
        "lote", "fase", "anexo", "clausula", "cláusula", "articulo", "artículo",
        "apartado", "seccion", "sección", "capitulo", "capítulo", "servidor",
        "nodo", "bloque", "grupo", "tipo", "nivel", "etapa", "version", "versión",
        "paciente", "caso", "expediente",
    })

    # Bound operator patterns in the clause segment preceding a quantity
    BOUND_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(
            r"(?:<=|≤|\bcomo\s+m[aá]ximo\b|\bm[aá]xim[oa]s?\b|\bhasta\b|\btope\b|\bl[ií]mite\s+m[aá]ximo\b|"
            r"\bno\s+superior\s+a\b|\bno\s+mayor\s+(?:a|de|que)\b|"
            r"\bno\s+(?:puede\s+|podr[aá]\s+|debe\s+|deber[aá]\s+)?(?:superar|exceder)\b|"
            r"\bno\s+(?:superar[aá]|exceder[aá])\b|"
            r"\bsin\s+(?:superar|exceder)\b)",
            re.IGNORECASE,
        ), "<="),
        (re.compile(
            r"(?:<|\binferior(?:es)?\s+a\b|\bmenor(?:es)?\s+(?:a|que|de)\b|\bmenos\s+de\b|\bpor\s+debajo\s+de\b|\bsub-\s*)",
            re.IGNORECASE,
        ), "<"),
        (re.compile(
            r"(?:>=|≥|\bcomo\s+m[ií]nimo\b|\bm[ií]nim[oa]s?\b|\bal\s+menos\b|\bpor\s+lo\s+menos\b|\ba\s+partir\s+de\b|"
            r"\bno\s+inferior(?:es)?\s+a\b|\bno\s+menor(?:es)?\s+(?:a|de|que)\b)",
            re.IGNORECASE,
        ), ">="),
        (re.compile(
            r"(?:>|\bsuperior(?:es)?\s+a\b|\bmayor(?:es)?\s+(?:a|que|de)\b|\bm[aá]s\s+de\b|\bpor\s+encima\s+de\b|"
            r"\bsupera(?:n|ndo)?\b|\bexcede(?:n|ndo)?\b)",
            re.IGNORECASE,
        ), ">"),
    ]

    # Regex capturing numbers with optional thousands/decimal separators, optional multiplier words, and units
    QUANTITY_REGEX = re.compile(
        r"(?<![A-Za-z0-9_/\-áéíóúñÁÉÍÓÚÑ])"
        r"(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
        r"(?:\s*(millones\s+de|mill[oó]n\s+de|miles\s+de|mil))?"
        r"(?:\s*(por\s+ciento|porciento|"
        r"microsegundos?|milisegundos?|miliseg|segundos?|segs?|secs?|minutos?|mins?|horas?|hrs?|d[ií]as?|semanas?|meses|mes|a[ñn]os?|"
        r"microgramos?|miligramos?|kilogramos?|gramos?|kilos?|"
        r"kilobytes?|megabytes?|gigabytes?|terabytes?|bytes?|"
        r"euros?|eur|d[oó]lares?|usd|keur|meur|kusd|musd|"
        r"k€|m€|mm€|k\$|m\$|€|\$|%|pct|"
        r"mcg|µg|ug|mg|kg|gr|g|"
        r"µs|us|ms|s|h|d|"
        r"kb|mb|gb|tb|"
        r"khz|mhz|ghz|hz|"
        r"[km](?=\s*(?:€|\$|eur\b|euros?\b|usd\b|d[oó]lares?\b))"
        r"(?:\s*(?:de\s+)?(?:€|\$|eur|euros?|usd|d[oó]lares?))?"
        r"))?(?![A-Za-z0-9_áéíóúñÁÉÍÓÚÑ])",
        re.IGNORECASE,
    )

    @classmethod
    def parse_number_string(cls, raw_num: str) -> float:
        """
        Parses localized number strings (ES/EN) into float:
        - '50.000' -> 50000.0 (Spanish thousands)
        - '1,500' -> 1500.0 (English thousands)
        - '1.500.000,50' -> 1500000.5 (Spanish thousands + decimal)
        - '1,500,000.50' -> 1500000.5 (English thousands + decimal)
        - '1.5' or '1,5' -> 1.5 (Decimal)
        - '0.15' or '0,15' -> 0.15 (Decimal)
        """
        s = raw_num.strip()
        if "." in s and "," in s:
            if s.rfind(",") > s.rfind("."):
                # Format: 1.250.000,50
                s = s.replace(".", "").replace(",", ".")
            else:
                # Format: 1,250,000.50
                s = s.replace(",", "")
            return float(s)

        if "," in s:
            # Check if comma is thousands separator: e.g. '1,500' or '1,000,000' (leading part != '0')
            parts = s.split(",")
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] != "0"):
                return float(s.replace(",", ""))
            return float(s.replace(",", "."))

        if "." in s:
            parts = s.split(".")
            # Multiple dots -> definitely thousands separator (e.g. '1.500.000')
            if len(parts) > 2:
                return float(s.replace(".", ""))
            # Single dot followed by exactly 3 digits and non-zero integer part: e.g. '50.000', '1.500'
            if len(parts) == 2 and len(parts[1]) == 3 and parts[0] != "0":
                return float(s.replace(".", ""))
            return float(s)

        return float(s)

    @classmethod
    def detect_bound_operator(cls, text: str, start_idx: int, prev_end_idx: int = 0) -> str:
        """Inspects the preceding clause segment for the closest inequality/bound modifier."""
        window_start = max(prev_end_idx, start_idx - 85)
        prefix = text[window_start:start_idx]
        best_op = "=="
        best_end_pos = -1

        for pattern, op in cls.BOUND_PATTERNS:
            for m in pattern.finditer(prefix):
                if m.end() > best_end_pos:
                    best_end_pos = m.end()
                    best_op = op
        return best_op

    @classmethod
    def _is_entity_designator_prefix(cls, text: str, start_idx: int, has_unit: bool) -> bool:
        """
        Determines whether a bare number is actually an entity index (e.g. 'Lote 1', 'Fase 2', 'Cláusula 4').
        Only applies when the number has no measurement unit attached.
        """
        if has_unit:
            return False
        prefix = text[max(0, start_idx - 25):start_idx]
        match = re.search(r"\b([a-zA-ZáéíóúñÁÉÍÓÚÑ]+)\s+$", prefix)
        if not match:
            return False
        preceding_word = match.group(1).lower()
        return preceding_word in cls.ENTITY_DESIGNATOR_NOUNS

    @classmethod
    def extract_quantities(cls, text: str) -> List[NumericalQuantity]:
        """Extracts and normalizes all numerical quantities from text into canonical NumericalQuantity objects."""
        quantities: List[NumericalQuantity] = []
        prev_end_idx = 0
        for match in cls.QUANTITY_REGEX.finditer(text):
            raw_num_str = match.group(1)
            multiplier_word = (match.group(2) or "").strip().lower()
            raw_unit_str = (match.group(3) or "").strip()
            start_idx, end_idx = match.span()

            has_explicit_unit = bool(raw_unit_str) or bool(multiplier_word)
            if cls._is_entity_designator_prefix(text, start_idx, has_explicit_unit):
                continue

            raw_val = cls.parse_number_string(raw_num_str)
            word_mult = 1.0
            if multiplier_word:
                if "mill" in multiplier_word:
                    word_mult = 1e6
                elif "mil" in multiplier_word:
                    word_mult = 1e3

            unit_lower = re.sub(r"\s+", " ", raw_unit_str.lower()).strip()
            dimension = "scalar"
            canonical_unit = "scalar"
            unit_mult = 1.0

            if unit_lower:
                # Handle compound multiplier + currency like 'm€', 'm eur', 'k€', 'm de euros'
                compound_curr = re.match(
                    r"^([km])\s*(?:de\s+)?(€|\$|eur|euros?|usd|d[oó]lares?)$",
                    unit_lower,
                )
                if compound_curr:
                    prefix_char = compound_curr.group(1)
                    curr_tok = compound_curr.group(2)
                    mult_factor = 1e3 if prefix_char == "k" else 1e6
                    dimension = "currency"
                    canonical_unit = "usd" if curr_tok in ("$", "usd") or "lar" in curr_tok else "eur"
                    unit_mult = mult_factor
                elif unit_lower in cls.UNIT_REGISTRY:
                    dimension, canonical_unit, unit_mult = cls.UNIT_REGISTRY[unit_lower]
                else:
                    # Fallback normalized lookup without accents
                    unit_no_acc = strip_accents(unit_lower)
                    if unit_no_acc in cls.UNIT_REGISTRY:
                        dimension, canonical_unit, unit_mult = cls.UNIT_REGISTRY[unit_no_acc]

            canonical_val = raw_val * word_mult * unit_mult
            bound_op = cls.detect_bound_operator(text, start_idx, prev_end_idx)
            prev_end_idx = end_idx

            quantities.append(
                NumericalQuantity(
                    raw_text=match.group(0).strip(),
                    raw_value=raw_val * word_mult,
                    raw_unit=unit_lower,
                    canonical_value=canonical_val,
                    canonical_unit=canonical_unit,
                    dimension=dimension,
                    bound_operator=bound_op,
                    span=(start_idx, end_idx),
                )
            )
        return quantities

    @classmethod
    def are_equivalent(
        cls,
        q1: NumericalQuantity,
        q2: NumericalQuantity,
        rel_tol: float = 1e-3,
        abs_tol: float = 1e-6,
    ) -> bool:
        """
        Checks whether two NumericalQuantity instances represent the same physical/financial/numerical magnitude.
        Supports:
        - Cross-unit equivalence in the same dimension (e.g., 1.5 s == 1500 ms, 0.5 g == 500 mg, 2.5M€ == 2.500.000 EUR).
        - Calendar equivalence (e.g., 1 año == 12 meses == 365 días).
        - Percentage-to-decimal ratio equivalence (e.g., 15% == 0.15).
        """
        # Case 1: Same dimension and same canonical unit
        if q1.dimension == q2.dimension and q1.canonical_unit == q2.canonical_unit:
            if math.isclose(q1.canonical_value, q2.canonical_value, rel_tol=rel_tol, abs_tol=abs_tol):
                return True
            # Calendar year/month equivalence (1 año = 365d vs 12 meses = 360d)
            if q1.dimension == "time":
                u1, u2 = strip_accents(q1.raw_unit), strip_accents(q2.raw_unit)
                if (u1.startswith("ano") and u2.startswith("mes")) or (u2.startswith("ano") and u1.startswith("mes")):
                    years = q1.raw_value if u1.startswith("ano") else q2.raw_value
                    months = q2.raw_value if u1.startswith("ano") else q1.raw_value
                    if math.isclose(years * 12.0, months, rel_tol=rel_tol, abs_tol=abs_tol):
                        return True
            return False

        # Case 2: Percentage vs dimensionless decimal ratio (e.g. 15% == 0.15)
        dims = {q1.dimension, q2.dimension}
        if dims == {"percentage", "scalar"}:
            q_pct = q1 if q1.dimension == "percentage" else q2
            q_scl = q2 if q1.dimension == "percentage" else q1
            if 0.0 <= abs(q_scl.canonical_value) <= 1.0:
                return math.isclose(q_pct.canonical_value, q_scl.canonical_value, rel_tol=rel_tol, abs_tol=abs_tol)

        return False

    @classmethod
    def satisfies_bound(
        cls,
        reference_bound: NumericalQuantity,
        candidate: NumericalQuantity,
        rel_tol: float = 1e-3,
        abs_tol: float = 1e-6,
    ) -> bool:
        """
        Evaluates whether `candidate` satisfies the inequality constraint defined by `reference_bound`.
        Also validates operator compatibility if `candidate` itself carries a bound operator.
        """
        # Ensure comparable dimensions
        if not (
            (reference_bound.dimension == candidate.dimension and reference_bound.canonical_unit == candidate.canonical_unit)
            or ({reference_bound.dimension, candidate.dimension} == {"percentage", "scalar"})
        ):
            return False

        ref_val = reference_bound.canonical_value
        cand_val = candidate.canonical_value
        ref_op = reference_bound.bound_operator
        cand_op = candidate.bound_operator

        # If candidate also specifies an inequality operator, operators must not point in opposite directions
        if cand_op != "==":
            upper_ops = {"<", "<="}
            lower_ops = {">", ">="}
            if (ref_op in upper_ops and cand_op in lower_ops) or (ref_op in lower_ops and cand_op in upper_ops):
                return False
            # If both define the same bound direction, their threshold values must match
            if ref_op in upper_ops and cand_op in upper_ops:
                return cls.are_equivalent(reference_bound, candidate, rel_tol=rel_tol, abs_tol=abs_tol)
            if ref_op in lower_ops and cand_op in lower_ops:
                return cls.are_equivalent(reference_bound, candidate, rel_tol=rel_tol, abs_tol=abs_tol)

        if ref_op == "==":
            return cls.are_equivalent(reference_bound, candidate, rel_tol=rel_tol, abs_tol=abs_tol)
        if ref_op == "<=":
            return cand_val <= ref_val * (1.0 + rel_tol) + abs_tol
        if ref_op == "<":
            return cand_val < ref_val * (1.0 - 1e-9)
        if ref_op == ">=":
            return cand_val >= ref_val * (1.0 - rel_tol) - abs_tol
        if ref_op == ">":
            return cand_val > ref_val * (1.0 + 1e-9)
        return False


class DeonticPolarityAnalyzer:
    """
    Analyzes deontic modality and negation polarity at the elementary clause level.
    Implements Standard Deontic Logic (SDL) equivalences and contradiction rules:
    - FORBIDDEN(p)  <=> ~PERMITTED(p) <=> OBLIGATORY(~p)
    - EXEMPT(p)     <=> ~OBLIGATORY(p)
    - AFFIRMATIVE(p) contradicts NEGATED(p)
    """

    # Verbs/adjectives expressing intrinsic prohibition, exclusion, or clinical contraindication
    PROHIBITIVE_TERMS: FrozenSet[str] = frozenset({
        "prohibe", "prohiben", "prohibido", "prohibida", "prohibidos", "prohibidas", "prohibicion",
        "veda", "vedado", "vedada", "impide", "imposibilita", "excluye", "excluido", "excluida",
        "excluyentes", "excluyente", "inadmisible", "incompatible", "incompatibles", "ilegal",
        "invalido", "invalida", "nulo", "nula", "rechaza", "rechazado", "rechazada", "deniega",
        "denegado", "bloquea", "restringe", "restringido", "contraindicado", "contraindicada",
        "contraindicados", "contraindicadas", "contraindicacion", "desaconseja", "desaconsejado",
        "penaliza", "penalizado", "incumple", "viola", "vulnera",
    })

    # Terms expressing exemption or waiver of an obligation
    EXEMPT_TERMS: FrozenSet[str] = frozenset({
        "exento", "exenta", "exentos", "exentas", "exime", "eximido", "eximida",
        "dispensado", "dispensada", "opcional", "facultativo", "facultativa", "libre",
    })

    # Terms expressing strict obligation or requirement
    OBLIGATORY_TERMS: FrozenSet[str] = frozenset({
        "obligatorio", "obligatoria", "obligatorios", "obligatorias", "obligado", "obligada",
        "obligados", "obligadas", "obliga", "obligacion", "exige", "exigen", "exigido", "exigida",
        "requiere", "requieren", "requerido", "requerida", "requisito", "debe", "deben", "debera",
        "deberan", "imprescindible", "imperativo", "forzoso", "necesario", "necesaria",
    })

    # Terms expressing permission, recommendation, or authorization
    PERMISSIVE_TERMS: FrozenSet[str] = frozenset({
        "permite", "permiten", "permitido", "permitida", "permitidos", "permitidas",
        "autoriza", "autorizan", "autorizado", "autorizada", "admite", "admiten", "admitido",
        "admitida", "aprueba", "aprobado", "aprobada", "habilita", "habilitado", "faculta",
        "recomienda", "recomendado", "recomendada", "aconseja", "aconsejado", "indica", "indicado",
        "indicada", "puede", "pueden", "podra", "podran", "valido", "valida", "compatible",
        "compatibles", "acepta", "aceptado", "aceptada", "cumple", "respeta", "satisface",
    })

    # Explicit negation operators
    NEGATION_TOKENS: FrozenSet[str] = frozenset({
        "no", "nunca", "jamas", "ningun", "ninguna", "ninguno", "sin", "tampoco", "carece", "ausencia",
    })

    SPANISH_STOPWORDS: FrozenSet[str] = frozenset({
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en",
        "con", "por", "para", "sobre", "entre", "hacia", "hasta", "desde", "durante", "mediante",
        "segun", "y", "o", "e", "u", "pero", "sino", "aunque", "mientras", "que", "como", "cuando",
        "donde", "quien", "cual", "cuyo", "es", "son", "esta", "estan", "fue", "sido", "ser",
        "tiene", "tienen", "se", "su", "sus", "este", "esta", "estos", "estas", "ese", "esa",
        "esos", "esas", "todo", "toda", "todos", "todas", "cada", "mismo", "misma", "mas", "menos",
        "muy", "ya", "tambien", "asi", "actualmente", "explicitamente", "forma", "manera", "caso",
        "tipo", "parte", "uso", "nivel", "sistema", "modelo", "datos",
    })

    @classmethod
    def stem_word(cls, word: str) -> str:
        """Applies deterministic Spanish morphological root truncation for robust proposition alignment."""
        w = strip_accents(word.lower())
        if len(w) >= 7:
            return w[:6]
        if len(w) >= 5:
            return w[:5]
        return w

    @classmethod
    def extract_content_stems(cls, clause: str) -> FrozenSet[str]:
        """Extracts normalized content word stems excluding stopwords and deontic/modal operators."""
        tokens = re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_\-]{3,}\b", clause)
        modal_all = (
            cls.PROHIBITIVE_TERMS
            | cls.EXEMPT_TERMS
            | cls.OBLIGATORY_TERMS
            | cls.PERMISSIVE_TERMS
            | cls.NEGATION_TOKENS
            | cls.SPANISH_STOPWORDS
        )
        stems = set()
        for tok in tokens:
            norm = strip_accents(tok.lower())
            if norm in modal_all or norm.isdigit():
                continue
            stems.add(cls.stem_word(norm))
        return frozenset(stems)

    @classmethod
    def analyze_clause(cls, clause: str, entities: Optional[FrozenSet[str]] = None) -> ClauseDeonticSignature:
        """
        Computes the canonical deontic state and polarity of a clause.
        Handles double negations and Deontic Square equivalences:
        - 'no permite' / 'no debe' / 'prohíbe' / 'contraindicado' -> FORBIDDEN (NEGATIVE)
        - 'exento' / 'no obligatorio' / 'no exige' / 'no requiere' -> EXEMPT (NEGATIVE)
        - 'no prohíbe' / 'permite' -> PERMITTED (POSITIVE)
        - 'obligatorio' / 'exige' / 'debe' -> OBLIGATORY (POSITIVE)
        """
        raw_tokens = re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]+\b", clause)
        norm_tokens = [strip_accents(t.lower()) for t in raw_tokens]

        triggers: List[str] = []
        has_negation = False
        negated_indices: Set[int] = set()

        for i, tok in enumerate(norm_tokens):
            if tok in cls.NEGATION_TOKENS:
                has_negation = True
                triggers.append(raw_tokens[i])
                # Negation scopes over the next 1..4 tokens in the verbal phrase
                for j in range(i + 1, min(len(norm_tokens), i + 5)):
                    negated_indices.add(j)

        def is_financial_tax_exclusion(pos: int) -> bool:
            tok = norm_tokens[pos]
            if tok in {"excluye", "excluido", "excluida", "excluyentes", "excluyente"}:
                tax_terms = {"iva", "impuesto", "impuestos", "tasa", "tasas", "arancel", "aranceles", "tributo", "tributos"}
                if pos > 0 and norm_tokens[pos - 1] in tax_terms:
                    return True
                if pos < len(norm_tokens) - 1 and norm_tokens[pos + 1] in tax_terms:
                    return True
                if pos < len(norm_tokens) - 2 and norm_tokens[pos + 1] in {"el", "la", "los", "las"} and norm_tokens[pos + 2] in tax_terms:
                    return True
            return False

        prohibitive_hits = [
            i for i, t in enumerate(norm_tokens)
            if t in cls.PROHIBITIVE_TERMS and not is_financial_tax_exclusion(i)
        ]
        exempt_hits = [i for i, t in enumerate(norm_tokens) if t in cls.EXEMPT_TERMS]
        obligatory_hits = [i for i, t in enumerate(norm_tokens) if t in cls.OBLIGATORY_TERMS]
        permissive_hits = [i for i, t in enumerate(norm_tokens) if t in cls.PERMISSIVE_TERMS]

        deontic_state = "AFFIRMATIVE"
        polarity = "POSITIVE"

        if prohibitive_hits:
            idx = prohibitive_hits[0]
            triggers.append(raw_tokens[idx])
            if idx in negated_indices:
                # Double negation: 'no está prohibido' -> PERMITTED (POSITIVE)
                deontic_state = "PERMITTED"
                polarity = "POSITIVE"
            else:
                deontic_state = "FORBIDDEN"
                polarity = "NEGATIVE"
        elif exempt_hits:
            idx = exempt_hits[0]
            triggers.append(raw_tokens[idx])
            if idx in negated_indices:
                # 'no está exento' -> OBLIGATORY (POSITIVE)
                deontic_state = "OBLIGATORY"
                polarity = "POSITIVE"
            else:
                deontic_state = "EXEMPT"
                polarity = "NEGATIVE"
        elif obligatory_hits:
            idx = obligatory_hits[0]
            triggers.append(raw_tokens[idx])
            if idx in negated_indices:
                # Check if 'no debe' / 'no deberá' (which in Spanish means FORBIDDEN) vs 'no es obligatorio' / 'no requiere' (EXEMPT)
                tok = norm_tokens[idx]
                if tok in {"debe", "deben", "debera", "deberan"}:
                    deontic_state = "FORBIDDEN"
                else:
                    deontic_state = "EXEMPT"
                polarity = "NEGATIVE"
            else:
                deontic_state = "OBLIGATORY"
                polarity = "POSITIVE"
        elif permissive_hits:
            idx = permissive_hits[0]
            triggers.append(raw_tokens[idx])
            if idx in negated_indices:
                # ~Permitted(p) <=> Forbidden(p) (e.g., 'no permite', 'no se recomienda', 'no cumple')
                deontic_state = "FORBIDDEN"
                polarity = "NEGATIVE"
            else:
                deontic_state = "PERMITTED"
                polarity = "POSITIVE"
        elif has_negation:
            deontic_state = "NEGATED"
            polarity = "NEGATIVE"

        content_stems = cls.extract_content_stems(clause)
        return ClauseDeonticSignature(
            clause_text=clause.strip(),
            polarity=polarity,
            deontic_state=deontic_state,
            trigger_tokens=tuple(triggers),
            content_stems=content_stems,
            entities=entities or frozenset(),
        )

    @classmethod
    def are_contradictory(cls, ctx_sig: ClauseDeonticSignature, ans_sig: ClauseDeonticSignature) -> bool:
        """
        Determines whether two deontic signatures over the same proposition are logically contradictory.
        Note:
        - FORBIDDEN ('prohíbe') and FORBIDDEN ('no permite' / 'no debe') have the SAME state & polarity -> NOT contradictory.
        - EXEMPT ('exento') and EXEMPT ('no es obligatorio') have the SAME state & polarity -> NOT contradictory.
        - FORBIDDEN vs PERMITTED / OBLIGATORY / AFFIRMATIVE -> CONTRADICTORY.
        - OBLIGATORY vs EXEMPT / FORBIDDEN / NEGATED -> CONTRADICTORY.
        - AFFIRMATIVE vs NEGATED -> CONTRADICTORY.
        """
        if ctx_sig.polarity != ans_sig.polarity:
            return True
        return False


class EntityQuantityBinder:
    """
    Segments text into elementary clauses and binds each numerical quantity to its local subject
    entities and attribute predicate, forming a local Symbolic Fact Graph.
    """

    CLAUSE_SPLIT_CONJUNCTIONS = re.compile(
        r"(?:\s*;\s*|\s*,\s*(?:mientras\s+que|en\s+tanto\s+que|aunque|pero|sino\s+que|donde)\s+|\s+pero\s+|\s+mientras\s+que\s+)",
        re.IGNORECASE,
    )

    DESIGNATOR_ENTITY_REGEX = re.compile(
        r"\b(servidor|lote|fase|anexo|cl[aá]usula|art[ií]culo|apartado|secci[oó]n|cap[ií]tulo|"
        r"nodo|bloque|grupo|tipo|nivel|etapa|paciente|modelo|cl[uú]ster|motor)\s+"
        r"(?:de\s+[a-zA-ZáéíóúñÁÉÍÓÚÑ]+\s+)?"
        r"([A-Za-z0-9][A-Za-z0-9_\-]*)\b",
        re.IGNORECASE,
    )

    PREDICATE_NOUNS: FrozenSet[str] = frozenset({
        "latencia", "presupuesto", "importe", "coste", "costo", "precio", "valor", "plazo",
        "duracion", "tiempo", "dosis", "concentracion", "frecuencia", "f1", "score", "precision",
        "exactitud", "recall", "muestras", "pacientes", "usuarios", "nodos", "garantia",
        "penalizacion", "solvencia", "facturacion", "temperatura", "frecuencia", "riesgo",
        "umbral", "limite", "tope", "capacidad", "memoria", "velocidad", "rendimiento",
    })

    NON_ENTITY_WORDS: FrozenSet[str] = frozenset({
        "de", "del", "en", "con", "por", "para", "sin", "sobre", "entre", "hasta", "desde",
        "el", "la", "los", "las", "un", "una", "unos", "unas", "que", "es", "son", "tiene",
        "tienen", "opera", "presenta", "alcanza", "alcanzo", "utiliza", "registra", "exige",
        "permite", "prohibe", "requiere", "cuenta", "supera", "cumple", "maximo", "maxima",
        "minimo", "minima", "inferior", "superior", "total", "anual", "mensual", "diario",
        "actualmente", "asimismo", "ademas", "seguridad", "produccion", "inferencia",
        "licitacion", "adjudicatario", "administracion", "pliego", "contrato", "oferta",
        "propuesta", "directiva", "sistema", "administradores", "sistemas", "pacientes",
        "tratamiento", "ensayo", "clinico", "estudio", "informe", "documento",
    })

    @classmethod
    def split_into_clauses(cls, text: str) -> List[str]:
        """
        Splits text into sentences and further splits coordinated multi-quantity sentences into
        elementary clauses (e.g., 'El servidor A tiene 14ms y el servidor B tiene 92ms').
        """
        # Protect decimal/thousands dots inside numbers before sentence splitting
        sentences = [s.strip() for s in re.split(r"(?<!\d)[.!?]+(?!\d)|\n+", text) if len(s.strip()) > 2]
        clauses: List[str] = []

        for sent in sentences:
            subparts = cls.CLAUSE_SPLIT_CONJUNCTIONS.split(sent)
            for part in subparts:
                part = part.strip()
                if not part:
                    continue
                # Check if ' y ' connects two independent clauses that BOTH contain a number or modal verb
                y_splits = cls._split_coordinated_y_clauses(part)
                clauses.extend(y_splits)

        return clauses

    @classmethod
    def _split_coordinated_y_clauses(cls, clause: str) -> List[str]:
        """
        Splits a clause on ' y ' ONLY when both sides contain a numerical quantity and the right side
        introduces a new subject/determiner/preposition (avoiding splitting 'entre 10ms y 20ms').
        """
        # Look for ' y ' followed by an article, preposition, or capitalized subject/designator
        pattern = re.compile(
            r"\s+y\s+(?=(?:el|la|los|las|en\s+el|en\s+la|para\s+el|para\s+la|un|una|"
            r"servidor|lote|fase|modelo|motor|cl[uú]ster|paciente|[A-ZÁÉÍÓÚÑ][a-záéíóúñA-Z0-9_\-]+)\b)",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(clause))
        if not matches:
            return [clause]

        result: List[str] = []
        last_idx = 0
        for m in matches:
            left_candidate = clause[last_idx:m.start()].strip()
            right_remainder = clause[m.end():].strip()
            # Ensure both left and right contain at least one digit, and left doesn't end with 'entre X'
            if (
                re.search(r"\d", left_candidate)
                and re.search(r"\d", right_remainder)
                and not re.search(r"\bentre\s+\d+(?:[.,]\d+)?\s*[a-zA-Z%€$]*$", left_candidate, re.IGNORECASE)
            ):
                result.append(left_candidate)
                last_idx = m.end()

        result.append(clause[last_idx:].strip())
        return [c for c in result if c]

    @classmethod
    def extract_clause_entities(cls, clause: str, known_entities: Optional[Set[str]] = None) -> FrozenSet[str]:
        """
        Extracts canonical entity identifiers from a clause, including:
        1. Designator + ID pairs ('servidor A' -> 'servidor_a', 'Lote 1' -> 'lote_1', 'servidor vLLM' -> 'vllm').
        2. Technical acronyms and mixed-case identifiers ('vLLM', 'A100', 'AWQ', 'TPUv4', 'Sciling-Bio-Llama').
        3. Capitalized proper nouns ('Metformina', 'Atorvastatina', 'MedQA', 'Kubernetes').
        """
        entities: Set[str] = set()

        # 1. Designator + Identifier (e.g. 'servidor A', 'Lote 1', 'Fase 2', 'servidor de inferencia vLLM')
        for m in cls.DESIGNATOR_ENTITY_REGEX.finditer(clause):
            designator = strip_accents(m.group(1).lower())
            ident_raw = m.group(2)
            ident_norm = strip_accents(ident_raw.lower())
            if ident_norm in cls.NON_ENTITY_WORDS:
                continue
            # Single letter ('A', 'B') or integer ('1', '2') -> composite 'servidor_a', 'lote_1'
            if len(ident_norm) == 1 or ident_norm.isdigit():
                entities.add(f"{designator}_{ident_norm}")
            else:
                entities.add(ident_norm)

        # 2. Standalone Named Entities & Acronyms
        tokens = re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_\-]+\b", clause)
        for idx, token in enumerate(tokens):
            norm_tok = strip_accents(token.lower())
            if norm_tok in cls.NON_ENTITY_WORDS or norm_tok.isdigit():
                continue
            # Skip pure numbers with units (e.g. '14ms', '500mg', '50k')
            if re.match(r"^\d+(?:[.,]\d+)?(?:ms|s|min|h|d|mg|g|kg|mcg|ug|kb|mb|gb|tb|hz|khz|mhz|ghz|k|m|eur|usd|pct)?$", norm_tok):
                continue

            # Acronyms & mixed alphanumeric ('vLLM', 'AWQ', 'TPUv4', 'A100', 'Bio-Llama', 'F1')
            if norm_tok not in {"f1", "c2", "p50", "p95", "p99"} and re.search(
                r"[A-Z]{2,}|[a-z]+[A-Z]|[A-Z]+[0-9]|[0-9]+[A-Z]|\b[A-Z][a-z]+-[A-Z]", token
            ):
                entities.add(norm_tok)
            # Mid-clause capitalized proper nouns
            elif idx > 0 and token[0].isupper() and len(token) > 2:
                if norm_tok not in UnitNormalizer.ENTITY_DESIGNATOR_NOUNS:
                    entities.add(norm_tok)
            # Sentence-initial capitalized proper nouns (if not a common function/discourse word or designator)
            elif idx == 0 and token[0].isupper() and len(token) > 3:
                if (
                    norm_tok not in DeonticPolarityAnalyzer.SPANISH_STOPWORDS
                    and norm_tok not in UnitNormalizer.ENTITY_DESIGNATOR_NOUNS
                    and norm_tok not in cls.PREDICATE_NOUNS
                ):
                    if known_entities is not None and norm_tok in known_entities:
                        entities.add(norm_tok)
            # Known entities from context matched case-insensitively
            if known_entities is not None and norm_tok in known_entities:
                entities.add(norm_tok)

        return frozenset(entities)

    @classmethod
    def infer_predicate(cls, clause: str, qty: NumericalQuantity) -> str:
        """Finds the closest domain attribute noun (e.g., 'latencia', 'presupuesto', 'dosis') to the quantity."""
        q_center = (qty.span[0] + qty.span[1]) / 2.0
        best_pred = qty.dimension
        best_dist = float("inf")

        for m in re.finditer(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+\b", clause):
            tok_norm = strip_accents(m.group(0).lower())
            if tok_norm in cls.PREDICATE_NOUNS:
                t_center = (m.start() + m.end()) / 2.0
                dist = abs(q_center - t_center)
                if dist < best_dist:
                    best_dist = dist
                    best_pred = tok_norm

        return best_pred

    @classmethod
    def infer_primary_subject(
        cls,
        clause: str,
        clause_entities: FrozenSet[str],
        predicate: str,
        qty: NumericalQuantity,
    ) -> str:
        """Selects the primary subject entity closest to the start of the clause or quantity."""
        if not clause_entities:
            return predicate

        # Prefer designator_id entities (e.g. 'servidor_a', 'lote_1') or the entity appearing earliest before the quantity
        designator_ents = [e for e in clause_entities if "_" in e]
        if designator_ents:
            return sorted(designator_ents)[0]

        clause_norm = strip_accents(clause.lower())
        best_ent = sorted(clause_entities)[0]
        best_pos = len(clause_norm) + 1
        for ent in clause_entities:
            pos = clause_norm.find(ent)
            if 0 <= pos < best_pos:
                best_pos = pos
                best_ent = ent
        return best_ent


class SymbolicFactGraph:
    """
    Constructs a local First-Order Fact Graph from text chunks and verifies whether a candidate
    answer's triples and deontic propositions are entailed by or contradict the source graph.
    """

    def __init__(self, context_chunks: List[str]):
        self.context_chunks = context_chunks
        self.full_context = " ".join(context_chunks)
        self.context_clauses: List[str] = []
        for chunk in context_chunks:
            self.context_clauses.extend(EntityQuantityBinder.split_into_clauses(chunk))

        # First pass to collect all context entities
        prelim_entities: Set[str] = set()
        for cl in self.context_clauses:
            prelim_entities.update(EntityQuantityBinder.extract_clause_entities(cl))
        self.known_entities: FrozenSet[str] = frozenset(prelim_entities)

        # Build clause deontic signatures and fact triples
        self.clause_signatures: List[ClauseDeonticSignature] = []
        self.triples: List[FactTriple] = []
        self.quantities: List[NumericalQuantity] = []

        for cl in self.context_clauses:
            cl_ents = EntityQuantityBinder.extract_clause_entities(cl, set(self.known_entities))
            sig = DeonticPolarityAnalyzer.analyze_clause(cl, cl_ents)
            self.clause_signatures.append(sig)

            cl_quantities = UnitNormalizer.extract_quantities(cl)
            for q in cl_quantities:
                pred = EntityQuantityBinder.infer_predicate(cl, q)
                subj = EntityQuantityBinder.infer_primary_subject(cl, cl_ents, pred, q)
                self.quantities.append(q)
                self.triples.append(
                    FactTriple(
                        primary_subject=subj,
                        clause_entities=cl_ents,
                        predicate=pred,
                        quantity=q,
                        polarity=sig.polarity,
                        deontic_state=sig.deontic_state,
                        source_clause=cl,
                    )
                )

    @staticmethod
    def _subject_overlap(ans_triple: FactTriple, ctx_triple: FactTriple) -> bool:
        """Checks whether an answer triple and a context triple refer to the same entity/subject."""
        if ans_triple.clause_entities and ctx_triple.clause_entities:
            overlap = ans_triple.clause_entities & ctx_triple.clause_entities
            if overlap:
                # Ensure they don't have conflicting indexed designators (e.g. 'lote_1' vs 'lote_2' or 'servidor_a' vs 'servidor_b')
                ans_indexed = {e for e in ans_triple.clause_entities if "_" in e}
                ctx_indexed = {e for e in ctx_triple.clause_entities if "_" in e}
                if ans_indexed and ctx_indexed and not (ans_indexed & ctx_indexed):
                    return False
                return True
            return False
        return ans_triple.primary_subject == ctx_triple.primary_subject

    def verify_answer(self, generated_answer: str) -> List[str]:
        """
        Executes symbolic First-Order Logic and Deontic verification over `generated_answer`:
        1. Relational Swap Detection (Predicate-Argument Binding verification).
        2. Threshold / Inequality Bound Compliance (<, <=, >, >=).
        3. Canonical Numerical Grounding (with Dimensional Unit Normalization).
        4. Deontic & Negation Polarity Consistency (Square of Opposition).
        """
        violations, _, _, _ = self.verify_answer_with_metadata(generated_answer)
        return violations

    def verify_answer_with_metadata(
        self, generated_answer: str
    ) -> Tuple[List[str], Set[str], List[NumericalQuantity], List[str]]:
        """Single-pass symbolic verification returning (violations, entities, quantities, deontic_states)."""
        violations: List[str] = []
        all_ans_entities: Set[str] = set()
        all_ans_quantities: List[NumericalQuantity] = []
        all_deontic_states: List[str] = []
        answer_clauses = EntityQuantityBinder.split_into_clauses(generated_answer)

        for cl in answer_clauses:
            cl_ents = EntityQuantityBinder.extract_clause_entities(cl, set(self.known_entities))
            all_ans_entities.update(cl_ents)
            ans_sig = DeonticPolarityAnalyzer.analyze_clause(cl, cl_ents)
            if ans_sig.deontic_state != "AFFIRMATIVE":
                all_deontic_states.append(ans_sig.deontic_state)
            cl_quantities = UnitNormalizer.extract_quantities(cl)
            all_ans_quantities.extend(cl_quantities)

            # 1. Verify each numerical claim in the clause
            for q_ans in cl_quantities:
                pred = EntityQuantityBinder.infer_predicate(cl, q_ans)
                subj = EntityQuantityBinder.infer_primary_subject(cl, cl_ents, pred, q_ans)
                ans_triple = FactTriple(
                    primary_subject=subj,
                    clause_entities=cl_ents,
                    predicate=pred,
                    quantity=q_ans,
                    polarity=ans_sig.polarity,
                    deontic_state=ans_sig.deontic_state,
                    source_clause=cl,
                )
                triple_violation = self._verify_single_triple(ans_triple)
                if triple_violation:
                    violations.append(triple_violation)

            # 2. Verify Deontic & Negation Polarity against aligned context clauses
            deontic_violation = self._verify_clause_deontic_polarity(ans_sig)
            if deontic_violation:
                violations.append(deontic_violation)

        return violations, all_ans_entities, all_ans_quantities, all_deontic_states

    def _verify_single_triple(self, ans_triple: FactTriple) -> Optional[str]:
        """Verifies a single answer FactTriple against the context FactGraph."""
        q_ans = ans_triple.quantity

        # Find context triples with compatible dimension
        dim_compatible_ctx = [
            t for t in self.triples
            if t.quantity.dimension == q_ans.dimension
            or {t.quantity.dimension, q_ans.dimension} == {"percentage", "scalar"}
        ]

        if not dim_compatible_ctx:
            return (
                f"CRITICAL_NUMERICAL_HALLUCINATION: Quantity '{q_ans.raw_text}' "
                f"({q_ans.canonical_value:g} {q_ans.canonical_unit}) has no matching dimension in context."
            )

        # Partition compatible context triples into subject-matched, unscoped/global, and other-entity
        subject_matched_ctx = [t for t in dim_compatible_ctx if self._subject_overlap(ans_triple, t)]
        global_unscoped_ctx = [t for t in dim_compatible_ctx if not t.clause_entities]

        # Step A: Check if exact quantity matches a subject-matched context triple (or global triple if no subject conflict)
        candidate_matches = subject_matched_ctx if subject_matched_ctx else global_unscoped_ctx
        for ctx_t in candidate_matches:
            if UnitNormalizer.are_equivalent(ctx_t.quantity, q_ans):
                if ctx_t.quantity.bound_operator != "==" and q_ans.bound_operator != "==":
                    if not UnitNormalizer.satisfies_bound(ctx_t.quantity, q_ans):
                        return (
                            f"THRESHOLD_BOUND_VIOLATION: Operator flip for subject '{ans_triple.primary_subject}' "
                            f"(Answer: '{q_ans.bound_operator} {q_ans.raw_text}' vs "
                            f"Context: '{ctx_t.quantity.bound_operator} {ctx_t.quantity.raw_text}')."
                        )
                return None

        # Step B: Check for RELATIONAL_SWAP_HALLUCINATION
        # Did the answer take a quantity that belongs to a DIFFERENT explicit entity in the context?
        other_entity_matches = [
            t for t in dim_compatible_ctx
            if t.clause_entities and not self._subject_overlap(ans_triple, t) and UnitNormalizer.are_equivalent(t.quantity, q_ans)
        ]
        if other_entity_matches and (subject_matched_ctx or ans_triple.clause_entities):
            wrong_owner = other_entity_matches[0]
            expected_str = (
                f" (expected '{subject_matched_ctx[0].quantity.raw_text}')"
                if subject_matched_ctx
                else ""
            )
            return (
                f"RELATIONAL_SWAP_HALLUCINATION: Quantity '{q_ans.raw_text}' bound to subject "
                f"'{ans_triple.primary_subject}'{expected_str}, but in context '{q_ans.raw_text}' "
                f"belongs to '{wrong_owner.primary_subject}'."
            )

        # Step C: Check Inequality / Threshold Bounds on subject-matched (or predicate-matched / global) context triples
        bound_candidates = subject_matched_ctx if subject_matched_ctx else [
            t for t in dim_compatible_ctx if t.predicate == ans_triple.predicate or not t.clause_entities
        ]
        for ctx_t in bound_candidates:
            if ctx_t.quantity.bound_operator != "==":
                if UnitNormalizer.satisfies_bound(ctx_t.quantity, q_ans):
                    return None
                else:
                    return (
                        f"THRESHOLD_BOUND_VIOLATION: Quantity '{q_ans.raw_text}' "
                        f"({q_ans.canonical_value:g} {q_ans.canonical_unit}) for '{ans_triple.primary_subject}' "
                        f"violates context constraint '{ctx_t.quantity.bound_operator} {ctx_t.quantity.raw_text}' "
                        f"({ctx_t.quantity.canonical_value:g} {ctx_t.quantity.canonical_unit})."
                    )

        # Step D: If answer clause omitted explicit entities, check if any context triple with same dimension has equivalent value
        if not ans_triple.clause_entities:
            for ctx_t in dim_compatible_ctx:
                if UnitNormalizer.are_equivalent(ctx_t.quantity, q_ans):
                    if ctx_t.quantity.bound_operator != "==" and q_ans.bound_operator != "==":
                        if not UnitNormalizer.satisfies_bound(ctx_t.quantity, q_ans):
                            return (
                                f"THRESHOLD_BOUND_VIOLATION: Operator flip '{q_ans.bound_operator} {q_ans.raw_text}' "
                                f"contradicts context bound '{ctx_t.quantity.bound_operator} {ctx_t.quantity.raw_text}'."
                            )
                    return None

        return (
            f"CRITICAL_NUMERICAL_HALLUCINATION: Quantity '{q_ans.raw_text}' "
            f"({q_ans.canonical_value:g} {q_ans.canonical_unit}) not grounded in context."
        )

    def _verify_clause_deontic_polarity(self, ans_sig: ClauseDeonticSignature) -> Optional[str]:
        """
        Finds the best-aligned context clause by proposition content stems and checks for
        Deontic / Negation Polarity Inversion.
        """
        if not ans_sig.content_stems:
            return None

        best_ctx_sig: Optional[ClauseDeonticSignature] = None
        best_overlap_score = 0.0

        for ctx_sig in self.clause_signatures:
            if not ctx_sig.content_stems:
                continue
            # If both clauses specify indexed entities (e.g. 'lote_1' vs 'lote_2'), they must not conflict
            ans_idx_ents = {e for e in ans_sig.entities if "_" in e}
            ctx_idx_ents = {e for e in ctx_sig.entities if "_" in e}
            if ans_idx_ents and ctx_idx_ents and not (ans_idx_ents & ctx_idx_ents):
                continue

            shared = ans_sig.content_stems & ctx_sig.content_stems
            if not shared:
                continue
            denom = min(len(ans_sig.content_stems), len(ctx_sig.content_stems))
            score = len(shared) / float(max(1, denom))
            min_required = 1 if denom == 1 else 2
            if len(shared) >= min_required and score > best_overlap_score:
                best_overlap_score = score
                best_ctx_sig = ctx_sig

        if best_ctx_sig is not None and best_overlap_score >= 0.45:
            if DeonticPolarityAnalyzer.are_contradictory(best_ctx_sig, ans_sig):
                ctx_trig = ", ".join(best_ctx_sig.trigger_tokens) or "affirmative"
                ans_trig = ", ".join(ans_sig.trigger_tokens) or "affirmative"
                return (
                    f"DEONTIC_POLARITY_INVERSION: Proposition polarity flipped from "
                    f"{best_ctx_sig.deontic_state} ({best_ctx_sig.polarity}: '{ctx_trig}') in context to "
                    f"{ans_sig.deontic_state} ({ans_sig.polarity}: '{ans_trig}') in answer."
                )

        return None
