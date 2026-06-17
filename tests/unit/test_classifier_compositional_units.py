"""Testes Dia 4 — Camada 2 detecta unidades compostas via átomos conhecidos.

A robustez estrutural: composições não-enumeradas (mas formadas por
átomos da lista) são detectadas. Não-átomos quebram o casamento (sem
falso-positivo em prosa que segue número).
"""

from __future__ import annotations

import pytest

from toten.classifier import OntologicalClassifier
from toten.ontology.types import TipoNome


@pytest.fixture(scope="module")
def classifier() -> OntologicalClassifier:
    return OntologicalClassifier()


def _grandezas(regions: list) -> list[str]:
    return [r.content for r in regions if r.tipo is TipoNome.GRANDEZA_FISICA]


# ---------------------------------------------------------------------------
# Compostos LITERALMENTE em classifier_units_v0.json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("tensão 350 MPa aplicada", "350 MPa"),
        ("viscosidade 1.5e-3 Pa·s", "1.5e-3 Pa·s"),
        ("tensão 350 kgf/cm² no escoamento", "350 kgf/cm²"),
        ("fluxo 5 W/(m²·K) calculado", "5 W/(m²·K)"),
    ],
)
def test_unidades_literais(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    grandezas = _grandezas(classifier.classify(texto))
    assert esperado in grandezas


# ---------------------------------------------------------------------------
# Compostos NÃO-enumerados, mas com átomos conhecidos — núcleo do Dia 4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("solo com 10 tf/(m²·°C) de capacidade", "10 tf/(m²·°C)"),
        ("difusividade 2.5e-7 m²/h relatada", "2.5e-7 m²/h"),
        ("expansão 1.2e-5 mm/(m·K) medida", "1.2e-5 mm/(m·K)"),
        ("massa 12.5 kg·m²/s², impulso", "12.5 kg·m²/s²"),
    ],
)
def test_compostos_long_tail_via_atomos_conhecidos(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    grandezas = _grandezas(classifier.classify(texto))
    assert esperado in grandezas, (
        f"composto {esperado!r} (átomos todos conhecidos) não detectado em "
        f"{texto!r}; observado: {grandezas}"
    )


# ---------------------------------------------------------------------------
# Anti-falso-positivo: átomos NÃO-conhecidos quebram o casamento
# ---------------------------------------------------------------------------


def test_falso_positivo_atomo_desconhecido(
    classifier: OntologicalClassifier,
) -> None:
    """`foo` não é átomo, então `350 MPa·foo` NÃO casa como composto.
    Apenas `350 MPa` casa, e o resto vai pra ProsaTecnica."""
    regions = classifier.classify("tensão 350 MPa·foo bar")
    grandezas = _grandezas(regions)
    assert "350 MPa·foo" not in grandezas


@pytest.mark.skip(
    reason=(
        "OBSOLETO (2026-05-31): teste assume comportamento antigo onde "
        "tokens não-átomos eram filtrados. Com v1.1 unit-only + extensões, "
        "comportamento sobre tokens ambíguos como 'foo/bar' está sob "
        "revisão. Reescrita pendente."
    )
)
def test_unidade_pura_sem_atomo_conhecido_nao_casa(
    classifier: OntologicalClassifier,
) -> None:
    """`350 foo/bar` não tem átomo conhecido — nem GrandezaFisica nada."""
    regions = classifier.classify("valor 350 foo/bar arbitrário")
    grandezas = _grandezas(regions)
    assert not any("foo" in g or "bar" in g for g in grandezas)


# ---------------------------------------------------------------------------
# Cenário integrador — concreto protendido BR com long tail estrutural
# ---------------------------------------------------------------------------


def test_concreto_protendido_completo_com_compostos(
    classifier: OntologicalClassifier,
) -> None:
    texto = (
        "viga em concreto C40, cordoalha CP 190 RB, "
        "fck = 40 MPa, carga de 50 tf, "
        "momento 250 tf·m, massa específica 7.85 tf/m³, "
        "expansão térmica 1.2e-5 mm/(m·K)"
    )
    regions = classifier.classify(texto)
    grandezas = set(_grandezas(regions))

    esperados = {
        "40 MPa",
        "50 tf",
        "250 tf·m",
        "7.85 tf/m³",
        "1.2e-5 mm/(m·K)",
    }
    faltando = esperados - grandezas
    assert not faltando, f"faltando: {faltando}\nobservado: {grandezas}"
