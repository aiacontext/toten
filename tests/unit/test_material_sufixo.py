"""Tests da classe de problema: identificador de material + sufixo curto
que coincide com símbolo de unidade.

Antes da correção, `SAE 4140 H` virava `SAE [QTY value=4140 unit=H ...]`
porque H é Henry (indutância) e o resolver privilegiava priority (GF=0)
sobre length (IDX maior).

A correção tem dois braços articulados:

1. Resolver com containment override: quando IDX contém estritamente
   GF, IDX vence (mesma start E menor ou igual end, com pelo menos
   uma desigualdade estrita).

2. Padrões de material aceitando sufixo `[A-Z]{1,3}` separado por
   space (AISI, SAE, ASTM, EN, DIN).

Cobertura: SAE 4140 H, AISI 304 L (L coincide com litro), AISI 316 N
(N coincide com Newton), SAE 1045 T (T coincide com Tesla),
AISI 416 F (F coincide com Farad), e regressões nas formas curtas.
"""

from __future__ import annotations

import pytest

from toten import Tokenizer
from toten.classifier import OntologicalClassifier
from toten.ontology.types import TipoNome


@pytest.fixture(scope="module")
def classifier() -> OntologicalClassifier:
    return OntologicalClassifier()


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_ontology("oee-v1")


def _ids(regions: list) -> list[str]:
    return [r.content for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]


def _grandezas(regions: list) -> list[str]:
    return [r.content for r in regions if r.tipo is TipoNome.GRANDEZA_FISICA]


# ---------------------------------------------------------------------------
# Sufixos de material que coincidem com símbolos de unidade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado_ids"),
    [
        # SAE com sufixo H (coincide com Henry)
        ("aço SAE 4140 H normalizado", ["SAE 4140 H"]),
        ("liga SAE 4140 HT temperada", ["SAE 4140 HT"]),
        # AISI com sufixo L (coincide com Litro)
        ("aço AISI 304 L baixo carbono", ["AISI 304 L"]),
        # AISI com sufixo N (coincide com Newton)
        ("aço AISI 316 N nitretado", ["AISI 316 N"]),
        # AISI com sufixo LN (multi-letra)
        ("aço AISI 316 LN", ["AISI 316 LN"]),
        # AISI com sufixo F (coincide com Farad)
        ("aço AISI 416 F free-machining", ["AISI 416 F"]),
        # SAE com sufixo T (coincide com Tesla)
        ("aço SAE 1045 T tratado", ["SAE 1045 T"]),
    ],
)
def test_material_com_sufixo_letra_unica(
    classifier: OntologicalClassifier, texto: str, esperado_ids: list[str]
) -> None:
    """Identificador completo é preservado; a letra-sufixo não vira
    GrandezaFisica acidental (Henry, Litro, Newton, Farad, Tesla)."""
    regions = classifier.classify(texto)
    ids_obs = _ids(regions)
    for esperado in esperado_ids:
        assert esperado in ids_obs, (
            f"esperado IDX '{esperado}' em '{texto}'; observado: {ids_obs}"
        )


@pytest.mark.parametrize(
    ("texto", "letra_sufixo"),
    [
        ("aço SAE 4140 H normalizado", "H"),
        ("aço AISI 304 L", "L"),
        ("aço AISI 316 N", "N"),
        ("aço AISI 416 F", "F"),
        ("aço SAE 1045 T", "T"),
    ],
)
def test_sufixo_nao_vira_grandeza_fisica(
    classifier: OntologicalClassifier, texto: str, letra_sufixo: str
) -> None:
    """Validação simétrica: a letra-sufixo NÃO deve vir como GrandezaFisica
    isolada (containment override descarta o GF candidato)."""
    regions = classifier.classify(texto)
    grandezas_obs = _grandezas(regions)
    # Não pode haver GF cujo content termine na letra-sufixo isolada
    assert not any(g.endswith(f" {letra_sufixo}") for g in grandezas_obs), (
        f"GrandezaFisica espúria com sufixo '{letra_sufixo}' detectada em "
        f"'{texto}': {grandezas_obs}"
    )


# ---------------------------------------------------------------------------
# Regressões: formas curtas ainda funcionam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("aço SAE 4140", "SAE 4140"),
        ("aço AISI 304", "AISI 304"),
        ("aço AISI 1045", "AISI 1045"),
        ("perfil ASTM A36", "ASTM A36"),
        ("aço EN 10025", "EN 10025"),
        ("perfil DIN 17100", "DIN 17100"),
    ],
)
def test_material_curto_continua_funcionando(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    ids_obs = _ids(classifier.classify(texto))
    assert esperado in ids_obs


# ---------------------------------------------------------------------------
# Grandezas legítimas com letras-unidade não devem ser engolidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado_gf"),
    [
        # Henry isolado
        ("indutância de 50 H no circuito", "50 H"),
        # Newton isolado
        ("força de 100 N aplicada", "100 N"),
        # Litro isolado
        ("volume de 2 L armazenado", "2 L"),
        # Tesla
        ("campo magnético de 0.5 T", "0.5 T"),
        # Kelvin
        ("temperatura de 300 K", "300 K"),
    ],
)
def test_grandeza_com_letra_unica_continua_detectada(
    classifier: OntologicalClassifier, texto: str, esperado_gf: str
) -> None:
    """Sem contexto de identificador, a letra única é grandeza legítima."""
    gs_obs = _grandezas(classifier.classify(texto))
    assert esperado_gf in gs_obs, f"GF '{esperado_gf}' não detectada em '{texto}'"


# ---------------------------------------------------------------------------
# Pipeline ponta-a-ponta — Modo B canônico para o piloto
# ---------------------------------------------------------------------------


def test_pipeline_sae_4140_h_modo_b(tok: Tokenizer) -> None:
    """Caso 5 do piloto adversarial: SAE 4140 H deve sair como [IDX]
    completo, sem fragmentação de '4140 H' em GrandezaFisica de Henry."""
    out = tok.preprocess("Compare SAE 4140 H normalizado com SAE 4140")
    assert "[IDX:sae-4140-h]" in out
    assert "[IDX:sae-4140]" in out
    # Não pode haver QTY com unit=H espúria
    assert "unit=H " not in out
    assert "unit=H]" not in out


def test_pipeline_aisi_304_l_modo_b(tok: Tokenizer) -> None:
    """AISI 304 L (baixo carbono) preservado, L não vira litro."""
    out = tok.preprocess("usar aço AISI 304 L na soldagem")
    assert "[IDX:aisi-304-l]" in out
    assert "unit=L " not in out


def test_pipeline_containment_estrita(tok: Tokenizer) -> None:
    """Princípio geral: candidato totalmente contido em outro é
    descartado. Validação em texto onde múltiplos casos co-ocorrem."""
    out = tok.preprocess(
        "viga em AISI 316 N submetida a 50 N de força, "
        "com indutância parasita de 0.5 H"
    )
    # AISI 316 N preservado como IDX completo
    assert "[IDX:aisi-316-n]" in out
    # 50 N como grandeza legítima (sem ser engolido)
    assert "unit=N" in out  # válido em "50 N"
    # 0.5 H como grandeza de Henry (sem ser engolido)
    assert "unit=H" in out  # válido em "0.5 H"


# ---------------------------------------------------------------------------
# Containment override — comportamento puro do resolver
# ---------------------------------------------------------------------------


def test_containment_override_descarta_contido_de_outro_tipo(
    classifier: OntologicalClassifier,
) -> None:
    """SAE 4140 H: IDX(0..10) contém GF(4..10) estritamente. GF descartado."""
    regions = classifier.classify("aço SAE 4140 H")
    ids_obs = _ids(regions)
    gs_obs = _grandezas(regions)
    assert "SAE 4140 H" in ids_obs
    # Nada com '4140 H' como GF isolada
    assert not any("4140" in g for g in gs_obs)


def test_overlap_parcial_mantem_resolucao_por_priority(
    classifier: OntologicalClassifier,
) -> None:
    """Quando não há containment estrito (só overlap parcial), priority
    decide. Validação para garantir que não rompemos esse comportamento."""
    # 'Hardox 500' (IDX estrutural) e '500 MPa' (GF) — overlap parcial em
    # '500'. Por priority, GF (0) vence sobre IDX (2).
    regions = classifier.classify("aço Hardox 500 MPa")
    gs_obs = _grandezas(regions)
    # GF deve emitir '500 MPa'
    assert any("500" in g and "MPa" in g for g in gs_obs)
