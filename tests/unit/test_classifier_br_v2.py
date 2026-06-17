"""Testes Dia 3.6 — cordoalhas, contexto PT-BR ampliado, unidades BR.

Cobertura cultural BR completa: componentes estruturais (cordoalha, fio,
cabo, armadura, estribo, viga, pilar, laje, sapata, parafuso),
designações compactas (CP 190 RB, AH 500, LE 1860), e unidades técnicas
brasileiras (tf, kgf, kgf/cm², tf·m, kgf·m, tf/m²).
"""

from __future__ import annotations

import pytest

from toten.classifier import OntologicalClassifier
from toten.instantiators import canonicalize_identifier
from toten.ontology.types import TipoNome


@pytest.fixture(scope="module")
def classifier() -> OntologicalClassifier:
    return OntologicalClassifier()


def _ids_contents(regions: list) -> list[str]:
    return [r.content for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]


def _grandezas_contents(regions: list) -> list[str]:
    return [r.content for r in regions if r.tipo is TipoNome.GRANDEZA_FISICA]


# ---------------------------------------------------------------------------
# Cordoalhas / armaduras / fios com designação compacta acrônimo + número
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("cordoalha CP 190 RB de protensão", "CP 190 RB"),
        ("a cordoalha CP 190 utilizada", "CP 190"),
        ("armadura AH 500 distribuída", "AH 500"),
        ("fio MR 250 lubrificado", "MR 250"),
        ("cabo LE 1860 alta resistência", "LE 1860"),
        ("estribo CA 60 espaçado", "CA 60"),
    ],
)
def test_acronym_designation_pt_br(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    ids = _ids_contents(classifier.classify(texto))
    assert esperado in ids, f"esperado '{esperado}' em {ids} para '{texto}'"


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("strand CP 190 RB", "CP 190 RB"),
        ("reinforcement AH 500", "AH 500"),
        ("cable LE 1860 grade", "LE 1860"),
    ],
)
def test_acronym_designation_en(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    ids = _ids_contents(classifier.classify(texto))
    assert esperado in ids, f"esperado '{esperado}' em {ids}"


def test_acronym_sem_contexto_nao_dispara(
    classifier: OntologicalClassifier,
) -> None:
    """`CP 190` solto, fora de contexto técnico, NÃO emite — protege contra
    falso-positivo em prosa."""
    regions = classifier.classify("o capítulo CP 190 do livro")
    ids = _ids_contents(regions)
    assert "CP 190" not in ids


# ---------------------------------------------------------------------------
# Unidades técnicas BR — força, pressão, momento
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("carga de 10 tf aplicada", "10 tf"),
        ("força de 1500 kgf medida", "1500 kgf"),
        ("peso de 2.5 t na estrutura", "2.5 t"),
        ("força de 0.5 kip aplicada", "0.5 kip"),
    ],
)
def test_unidades_forca_br(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    grandezas = _grandezas_contents(classifier.classify(texto))
    assert esperado in grandezas, f"esperado '{esperado}' em {grandezas}"


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("tensão de 350 kgf/cm² no escoamento", "350 kgf/cm²"),
        ("limite de 450 kgf/mm² calculado", "450 kgf/mm²"),
        ("solo com 25 tf/m² de capacidade", "25 tf/m²"),
    ],
)
def test_unidades_pressao_tecnica_br(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    grandezas = _grandezas_contents(classifier.classify(texto))
    assert esperado in grandezas, f"esperado '{esperado}' em {grandezas}"


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("momento de 50 tf·m no engaste", "50 tf·m"),
        ("torque de 250 kgf·m aplicado", "250 kgf·m"),
        ("kgf·cm é unidade pequena para 1500 kgf·cm de momento", "1500 kgf·cm"),
    ],
)
def test_unidades_momento_br(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    grandezas = _grandezas_contents(classifier.classify(texto))
    assert esperado in grandezas, f"esperado '{esperado}' em {grandezas}"


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("densidade de 2500 kgf/m³ adotada", "2500 kgf/m³"),
        ("massa específica 7.85 tf/m³ no aço", "7.85 tf/m³"),
        ("motor de 150 cv potência", "150 cv"),
        ("potência de 200 HP no equipamento", "200 HP"),
    ],
)
def test_outras_unidades_br(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    grandezas = _grandezas_contents(classifier.classify(texto))
    assert esperado in grandezas, f"esperado '{esperado}' em {grandezas}"


# ---------------------------------------------------------------------------
# Cenário integrador realista — concreto protendido brasileiro
# ---------------------------------------------------------------------------


def test_concreto_protendido_completo(
    classifier: OntologicalClassifier,
) -> None:
    texto = (
        "viga em concreto C40 protendida com cordoalha CP 190 RB, "
        "armadura passiva CA-50, fck = 40 MPa, "
        "carga de 50 tf aplicada gerando 250 tf·m"
    )
    regions = classifier.classify(texto)
    ids = set(_ids_contents(regions))
    grandezas = set(_grandezas_contents(regions))

    esperados_ids = {"C40", "CP 190 RB", "CA-50"}
    esperados_grandezas = {"40 MPa", "50 tf", "250 tf·m"}

    faltando_ids = esperados_ids - ids
    faltando_grandezas = esperados_grandezas - grandezas
    assert not faltando_ids, f"identificadores faltando: {faltando_ids}"
    assert not faltando_grandezas, f"grandezas faltando: {faltando_grandezas}"


def test_round_trip_canonical_cordoalha() -> None:
    """Cordoalha CP 190 RB, detectada pela Camada 2, é canonicalizada
    pela Camada 3 para slug determinístico — token atômico, identidade
    preservada."""
    c = OntologicalClassifier()
    regions = c.classify("cordoalha CP 190 RB de protensão")
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    slugs = [canonicalize_identifier(r.content) for r in ids]
    assert "cp-190-rb" in slugs, (
        f"slug 'cp-190-rb' não encontrado em {slugs}"
    )


def test_round_trip_canonical_grandeza_br() -> None:
    """50 tf em texto BR é detectado como GrandezaFisica — sem fragmentação
    em prosa."""
    c = OntologicalClassifier()
    regions = c.classify("uma carga de 50 tf foi aplicada")
    grandezas = _grandezas_contents(regions)
    assert "50 tf" in grandezas
