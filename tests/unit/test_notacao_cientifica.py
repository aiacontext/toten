"""Tests da classe de problema: notação científica em variantes Unicode/caret.

Engenharia BR/internacional usa `1,5 × 10⁻³` muito mais que `1.5e-3` em
texto formal. O framework deve aceitar as variantes equivalentes:

- `1.5e-3` (programming style)
- `1,5e-3` (PT-BR locale)
- `1,5 × 10⁻³` (Unicode superscript com ×, x, ·, *)
- `1.5 × 10^-3` (caret ASCII)
- `3.0 × 10^(-3)` (caret com parens)

Antes da correção, só `[eE]` era aceito. `× 10ⁿ` Unicode ficava
invisível ao classifier, perdendo grandezas legítimas como
`1,616 × 10⁻³⁵ m` (comprimento de Planck).
"""

from __future__ import annotations

import pytest

from toten import Tokenizer
from toten.classifier import OntologicalClassifier
from toten.instantiators import canonicalize_number, parse_grandeza
from toten.instantiators.quantity import _normalize_scientific_notation
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# _normalize_scientific_notation — normalização determinística
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("1,5 × 10⁻³", "1,5e-3"),
        ("1.5 × 10⁻³", "1.5e-3"),
        ("2 × 10⁵", "2e5"),
        ("2 × 10⁺⁵", "2e+5"),
        ("3.0 · 10⁻¹⁵", "3.0e-15"),
        ("4 x 10⁹", "4e9"),
        ("5 * 10⁻⁶", "5e-6"),
        # Caret ASCII
        ("1.5 × 10^-3", "1.5e-3"),
        ("2 × 10^5", "2e5"),
        ("3.0 × 10^(-3)", "3.0e-3"),
        ("4 × 10^(+15)", "4e+15"),
        # Casos sem mudança
        ("1.5e-3", "1.5e-3"),
        ("350", "350"),
    ],
)
def test_normalize_scientific(entrada: str, esperado: str) -> None:
    assert _normalize_scientific_notation(entrada) == esperado


# ---------------------------------------------------------------------------
# canonicalize_number — Unicode e caret integrados
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        # Unicode superscript
        ("1.5 × 10⁻³", 1.5e-3),
        ("1,5 × 10⁻³", 1.5e-3),
        ("1,616 × 10⁻³⁵", 1.616e-35),
        ("2 × 10⁵", 2.0e5),
        ("2 × 10⁺⁵", 2.0e5),
        ("3.0 · 10⁻¹⁵", 3.0e-15),
        # Caret ASCII
        ("2 × 10^5", 2.0e5),
        ("1.5 × 10^-3", 1.5e-3),
        ("3.0 × 10^(-3)", 3.0e-3),
        # Sinal Unicode na mantissa
        ("−1,5 × 10⁻³", -1.5e-3),
    ],
)
def test_canonicalize_number_variantes_cientificas(
    entrada: str, esperado: float
) -> None:
    assert canonicalize_number(entrada) == pytest.approx(esperado, rel=1e-12)


# ---------------------------------------------------------------------------
# parse_grandeza — variantes na entrada de uma região GF completa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "valor_esperado", "unidade"),
    [
        ("1,616 × 10⁻³⁵ m", 1.616e-35, "m"),
        ("5,29 × 10⁻¹¹ m", 5.29e-11, "m"),
        ("1.380649 × 10⁻²³ J/K", 1.380649e-23, "J/K"),
        ("2 × 10⁵ Pa", 2e5, "Pa"),
        ("3.0 × 10^-3 m/s", 3.0e-3, "m/s"),
    ],
)
def test_parse_grandeza_notacao_cientifica_unicode(
    entrada: str, valor_esperado: float, unidade: str
) -> None:
    q = parse_grandeza(entrada)
    assert q.value == pytest.approx(valor_esperado, rel=1e-12)
    assert q.unit_text == unidade


# ---------------------------------------------------------------------------
# Camada 2 — classifier detecta grandezas em notação Unicode
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier() -> OntologicalClassifier:
    return OntologicalClassifier()


@pytest.mark.parametrize(
    ("texto", "esperado_content"),
    [
        ("comprimento de Planck = 1,616 × 10⁻³⁵ m", "1,616 × 10⁻³⁵ m"),
        ("raio de Bohr a_0 = 5,29 × 10⁻¹¹ m", "5,29 × 10⁻¹¹ m"),
        ("pressão de 2 × 10⁵ Pa atuante", "2 × 10⁵ Pa"),
        ("velocidade 3.0 × 10^5 m/s no fluido", "3.0 × 10^5 m/s"),
    ],
)
def test_classifier_detecta_grandeza_em_notacao_unicode(
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
# Pipeline ponta-a-ponta — Modo B para notação científica formal
# ---------------------------------------------------------------------------


def test_pipeline_planck_unicode() -> None:
    """Caso 8 do piloto v2 — comprimento de Planck em notação BR/internacional
    formal."""
    tok = Tokenizer.from_ontology("oee-v1")
    out = tok.preprocess(
        "O comprimento de Planck é l_P = 1,616 × 10⁻³⁵ m, "
        "valor extremo fundamental."
    )
    # Grandeza detectada e canonizada com dim de comprimento
    assert "unit=m" in out
    assert "dim=[0,1,0,0,0,0,0]" in out
    # Valor normalizado preserva magnitude exata
    assert "value=1.616e-35" in out


def test_pipeline_multiplas_grandezas_cientificas() -> None:
    tok = Tokenizer.from_ontology("oee-v1")
    out = tok.preprocess(
        "Compare l_P = 1,616 × 10⁻³⁵ m com a_0 = 5,29 × 10⁻¹¹ m."
    )
    assert "value=1.616e-35" in out
    assert "value=5.29e-11" in out


# ---------------------------------------------------------------------------
# Não-regressão: × isolado entre dois números não casa como científica
# ---------------------------------------------------------------------------


def test_x_entre_numeros_nao_dispara_cientifica(
    classifier: OntologicalClassifier,
) -> None:
    """`5 m × 3 m` é dimensão (área), não notação científica. O regex
    `× 10` é específico — `× 3` não casa."""
    regions = classifier.classify("dimensão de 5 m × 3 m")
    grandezas = [
        r.content for r in regions if r.tipo is TipoNome.GRANDEZA_FISICA
    ]
    # Ambos `5 m` e `3 m` devem ser GFs separadas (não juntadas como sci)
    assert "5 m" in grandezas
    assert "3 m" in grandezas
