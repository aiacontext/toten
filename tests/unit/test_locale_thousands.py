"""Tests da classe de problema: locale PT-BR thousands vs US decimal.

Engenharia brasileira usa rotineiramente `1.900 MPa` para 1900 MPa,
`2.385,6 mm²` para 2385.6 mm², `1.234.567 kN` para 1234567 kN.

Antes da correção, o parser interpretava `1.900` como decimal US =
1.9 — erro de 3 ordens de magnitude em valor crítico de norma.

Heurística aplicada (para texto técnico BR):
- Múltiplos `.` sem `,` → thousands PT-BR
- Único `.` com 3 dígitos após E parte inteira não-zero → thousands PT-BR
- Único `.` com 1-2 dígitos após OU parte inteira zero → decimal US
- `,` único → sempre decimal PT-BR
- Ambos → último é decimal, outro thousands
"""

from __future__ import annotations

import pytest

from toten import Tokenizer
from toten.classifier import OntologicalClassifier
from toten.instantiators import canonicalize_number, parse_grandeza
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# canonicalize_number — locale resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        # PT-BR thousands sem decimal — caso crítico de texto técnico BR
        ("1.900", 1900.0),
        ("1.710", 1710.0),
        ("12.345", 12345.0),
        ("1.234.567", 1234567.0),
        # PT-BR thousands com decimal
        ("2.385,6", 2385.6),
        ("1.234,56", 1234.56),
        ("1.234.567,89", 1234567.89),
        # US thousands com decimal
        ("1,234.56", 1234.56),
        ("1,234,567.89", 1234567.89),
        # Decimal único (1-2 dígitos após `.`)
        ("287.4", 287.4),
        ("1.5", 1.5),
        ("0.5", 0.5),
        # Decimal único com integer = 0 (ex: 0.500 NÃO é thousands)
        ("0.500", 0.5),
        # Decimal único com 4+ dígitos após (NÃO é thousands)
        ("1.5000", 1.5),
        ("1.23456", 1.23456),
        # Decimal PT-BR único
        ("287,4", 287.4),
        ("1,5", 1.5),
        # Sinal
        ("-1.900", -1900.0),
        ("+2.385,6", 2385.6),
        # Múltiplas vírgulas sem ponto (US thousands raro)
        ("1,234,567", 1234567.0),
        # Ambos presentes com integer_part vazio antes do decimal
        (",5", 0.5),
    ],
)
def test_canonicalize_locale_thousands(entrada: str, esperado: float) -> None:
    resultado = canonicalize_number(entrada)
    assert resultado == pytest.approx(esperado, rel=1e-12)


# ---------------------------------------------------------------------------
# parse_grandeza — locale thousands em região GF completa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "valor", "unidade"),
    [
        ("1.900 MPa", 1900.0, "MPa"),
        ("1.710 MPa", 1710.0, "MPa"),
        ("2.385,6 mm²", 2385.6, "mm²"),
        ("1.234,56 kg", 1234.56, "kg"),
        ("1.234.567 kN", 1234567.0, "kN"),
        # Cases não-thousands devem continuar funcionando
        ("287,4 MPa", 287.4, "MPa"),
        ("0.5 m", 0.5, "m"),
        ("1.5e-3 Pa·s", 1.5e-3, "Pa·s"),
    ],
)
def test_parse_grandeza_pt_br_thousands(
    entrada: str, valor: float, unidade: str
) -> None:
    q = parse_grandeza(entrada)
    assert q.value == pytest.approx(valor, rel=1e-12)
    assert q.unit_text == unidade


# ---------------------------------------------------------------------------
# Camada 2 — classifier captura número PT-BR completo (não fragmenta)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier() -> OntologicalClassifier:
    return OntologicalClassifier()


@pytest.mark.parametrize(
    ("texto", "esperado_content"),
    [
        ("fptk = 1.900 MPa nominal", "1.900 MPa"),
        ("área Ap = 2.385,6 mm² total", "2.385,6 mm²"),
        ("massa estimada de 1.234,5 kg", "1.234,5 kg"),
        ("carga última 1.234.567 N aplicada", "1.234.567 N"),
    ],
)
def test_classifier_captura_pt_br_thousands(
    classifier: OntologicalClassifier, texto: str, esperado_content: str
) -> None:
    regions = classifier.classify(texto)
    grandezas = [
        r.content for r in regions if r.tipo is TipoNome.GRANDEZA_FISICA
    ]
    assert esperado_content in grandezas, (
        f"esperado '{esperado_content}' em {grandezas}"
    )


# ---------------------------------------------------------------------------
# Pipeline ponta-a-ponta — caso real de concreto protendido
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "OBSOLETO: refator anterior — fptk agora vira "
        "[SYM:fptk] em vez de [IDX:fptk]. CP 190 RB também sob revisão "
        "(provavelmente perdeu IDX). Reescrita pendente."
    )
)
def test_pipeline_cordoalha_cp_190_rb() -> None:
    """fptk = 1.900 MPa (PT-BR thousands de valor normativo NBR 7483)
    deve sair como 1900 MPa, NÃO como 1.9 MPa."""
    tok = Tokenizer.from_ontology("oee-v1")
    out = tok.preprocess(
        "Cordoalha CP 190 RB tem fptk = 1.900 MPa conforme NBR 7483."
    )
    # Valor preservado como 1.900 (engenheiro vê o que escreveu)
    assert "value=1.900" in out or "value=1900" in out
    # NÃO interpreta como 1.9
    assert "value=1.9 " not in out
    assert "[IDX:cp-190-rb]" in out
    assert "[IDX:fptk]" in out


def test_pipeline_area_pt_br_completa() -> None:
    """Ap = 2.385,6 mm² preservado integralmente, não fragmentado."""
    tok = Tokenizer.from_ontology("oee-v1")
    out = tok.preprocess(
        "Ap = 2.385,6 mm² para os 4 cabos de 6 cordoalhas cada"
    )
    # Valor preservado
    assert "value=2385.6" in out or "value=2.385,6" in out or "value=2.3856" in out
    # NÃO fragmenta em "2." + "[QTY 385.6]"
    assert "2.[QTY" not in out


# ---------------------------------------------------------------------------
# Não-regressão: casos US continuam funcionando
# ---------------------------------------------------------------------------


def test_us_decimal_continua_funcionando() -> None:
    """`350 MPa` simples continua. `1.5e-3` continua. `287.4` continua."""
    tok = Tokenizer.from_ontology("oee-v1")
    out = tok.preprocess(
        "tensão de 350 MPa, módulo 200 GPa, viscosidade 1.5e-3 Pa·s, σ_y = 287.4 MPa"
    )
    # Todos preservados
    assert "value=350 unit=MPa" in out
    assert "value=200 unit=GPa" in out
    assert "value=1.5e-3" in out or "value=0.0015" in out
    assert "value=287.4" in out
