"""Testes Dia 3.5 — cobertura BR-específica e padrões estruturais.

Avalia:
- Padrões BR curados (cimentos NBR 16697, aços rebar NBR 7480, classes
  de concreto NBR 8953, fabricantes USIMINAS/Niobrás/Gerdau, símbolos
  técnicos σ_y/f_ck etc., métodos GMAW/NDT/CAD).
- Padrões estruturais (hyphen-internal + context-required cap-word).
- Ausência de falso-positivo: prosa que NÃO contém identificador não
  emite IdentificadorTecnico via padrões estruturais.
"""

from __future__ import annotations

import pytest

from toten.classifier import OntologicalClassifier
from toten.ontology.types import TipoNome


@pytest.fixture(scope="module")
def classifier() -> OntologicalClassifier:
    return OntologicalClassifier()


def _ids(regions: list, esperado: str) -> bool:
    return any(
        r.tipo is TipoNome.IDENTIFICADOR_TECNICO and r.content == esperado
        for r in regions
    )


# ---------------------------------------------------------------------------
# Cimentos NBR 16697
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("o cimento CP-V ARI atinge resistência alta", "CP-V ARI"),
        ("usamos CP-V ARI-RS em ambiente agressivo", "CP-V ARI-RS"),
        ("cimento CP-I é tradicional", "CP-I"),
        ("CP-I-S contém escória", "CP-I-S"),
        ("CP-II-E foi escolhido", "CP-II-E"),
        ("CP-II-Z-32 para alvenaria", "CP-II-Z-32"),
        ("CP-III-40 alto-forno", "CP-III-40"),
        ("CP-IV-32 pozolânico", "CP-IV-32"),
        ("CP-V pré-fabricados", "CP-V"),
    ],
)
def test_cimento_nbr16697(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    assert _ids(classifier.classify(texto), esperado)


# ---------------------------------------------------------------------------
# Aços rebar NBR 7480
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("barra de CA-50 corrugada", "CA-50"),
        ("CA-60 trefilado", "CA-60"),
        ("malha CA-25 lisa", "CA-25"),
        ("rebar CA-50A para uso geral", "CA-50A"),
    ],
)
def test_rebar_nbr7480(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    assert _ids(classifier.classify(texto), esperado)


# ---------------------------------------------------------------------------
# Classes de concreto NBR 8953 (context-bounded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("usar fck = C25 conforme projeto", "C25"),
        ("classe C30 para vigas principais", "C30"),
        ("concreto C25/30 estrutural", "C25/30"),
        ("classe C50 protendido", "C50"),
    ],
)
def test_concreto_classe(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    assert _ids(classifier.classify(texto), esperado)


def test_concreto_classe_sem_contexto_nao_dispara(
    classifier: OntologicalClassifier,
) -> None:
    """`C25` solto (sem fck/concreto/classe precedente) NÃO emite — protege
    contra confusão com °C ou outras siglas."""
    regions = classifier.classify("temperatura C25 ambiente isolado")
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert not any(r.content == "C25" for r in ids)


# ---------------------------------------------------------------------------
# Fabricantes BR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("chapa USI-AR-400 para desgaste", "USI-AR-400"),
        ("USI-SAC-300 estrutural", "USI-SAC-300"),
        ("aço Niobrás de Araxá", "Niobrás"),
        ("a marca Niobrás 320 é alto-resistente", "Niobrás 320"),
        ("Gerdau GG-25 fundido", "Gerdau GG-25"),
    ],
)
def test_fabricantes_br(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    assert _ids(classifier.classify(texto), esperado)


# ---------------------------------------------------------------------------
# Símbolos técnicos com subscrito
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "OBSOLETO: refator ontológico unificado (refator anterior, "
        "'Camada 2 derivada de 6 condições') passou símbolos técnicos com "
        "subscript (σ_y, f_ck, etc.) para ExpressaoSimbolica [SYM:σ_y] em vez "
        "de IdentificadorTecnico [IDX:σy]. Decisão arquitetural deliberada — "
        "preservação literal (princípio do tokenizer 2026-05-20) + P3 mediação. "
        "Teste assume comportamento anterior; precisa ser reescrito para "
        "verificar [SYM:σ_y] OU removido se a nova ontologia tornar a "
        "verificação redundante."
    )
)
@pytest.mark.parametrize(
    "simbolo",
    ["σ_y", "σ_u", "σ_eq", "τ_y", "f_ck", "f_yk", "f_yd"],
)
def test_simbolos_tecnicos(
    classifier: OntologicalClassifier, simbolo: str
) -> None:
    """Símbolos consagrados em normas BR (σ_y, f_ck, etc.) são IDX.

    NOTA: E_s, E_c, E_p, A_c, I_c, A_p, etc. NÃO são IDX — são variáveis
    de engenharia (símbolos com valor instanciado pelo problema). Caem
    como SYM (ExpressaoSimbolica) preservando subscript.
    """
    texto = f"a propriedade {simbolo} foi calculada"
    regions = classifier.classify(texto)
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert any(r.content == simbolo for r in ids), (
        f"{simbolo} não detectado em '{texto}'"
    )


# ---------------------------------------------------------------------------
# Métodos (soldagem, NDT, simulação)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metodo",
    ["GMAW", "GTAW", "SMAW", "FCAW", "TIG", "MIG", "CMT"],
)
def test_metodo_soldagem(
    classifier: OntologicalClassifier, metodo: str
) -> None:
    texto = f"processo {metodo} aplicado no cordão"
    regions = classifier.classify(texto)
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert any(r.content == metodo for r in ids)


@pytest.mark.parametrize(
    "metodo",
    ["NDT", "UT", "RT", "PAUT", "TOFD"],
)
def test_metodo_ndt(classifier: OntologicalClassifier, metodo: str) -> None:
    texto = f"inspeção via {metodo} realizada"
    regions = classifier.classify(texto)
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert any(r.content == metodo for r in ids)


@pytest.mark.parametrize("metodo", ["CFD", "FEM", "FEA", "MEF", "CAD"])
def test_metodo_simulacao(
    classifier: OntologicalClassifier, metodo: str
) -> None:
    texto = f"análise por {metodo} apresentou convergência"
    regions = classifier.classify(texto)
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert any(r.content == metodo for r in ids)


# ---------------------------------------------------------------------------
# Padrões estruturais — hyphen-internal alfanumérico
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("aço Hardox-500 abrasivo", "Hardox-500"),
        ("liga Brem-280 importada", "Brem-280"),
        ("perfil Toscelik-S355 europeu", "Toscelik-S355"),
    ],
)
def test_estrutural_hyphen_alfanumerico(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    assert _ids(classifier.classify(texto), esperado)


# ---------------------------------------------------------------------------
# Padrões estruturais — context-required cap-word + designação
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("aço Hardox 500 abrasivo", "Hardox 500"),
        ("material Bremil 280 nacional", "Bremil 280"),
        ("steel Hardox 500 plate", "Hardox 500"),
        ("brand Bremil 280 product", "Bremil 280"),
    ],
)
def test_estrutural_context_capword(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    assert _ids(classifier.classify(texto), esperado)


def test_estrutural_capword_sem_contexto_nao_dispara(
    classifier: OntologicalClassifier,
) -> None:
    """`Pedro 1840` sem context-word precedente NÃO deve casar — protege
    contra falso-positivo em nomes próprios."""
    regions = classifier.classify("Pedro 1840 escreveu sobre a viga")
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert not any("Pedro" in r.content for r in ids)


def test_estrutural_capword_minimo_6_caracteres() -> None:
    """Cap-words curtas em PT-BR (`Para 1840`, `Sobre 100`, `Antes 50`)
    NÃO casam — preposições/advérbios são falso-positivo comum.
    Padrão exige cap-word com 6+ caracteres totais (≥5 minúsculas)."""
    c = OntologicalClassifier()
    for texto, palavra in [
        ("tipo Para 1840", "Para"),
        ("marca Sobre 100", "Sobre"),
        ("classe Antes 50", "Antes"),
    ]:
        regions = c.classify(texto)
        ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
        assert not any(palavra in r.content for r in ids), (
            f"falso-positivo: '{palavra}' detectado em '{texto}'"
        )


# ---------------------------------------------------------------------------
# Cenário integrador BR
# ---------------------------------------------------------------------------


def test_sentenca_engenharia_br_completa(
    classifier: OntologicalClassifier,
) -> None:
    texto = (
        "viga em concreto C30 com aço CA-50, cimento CP-V ARI, "
        "ensaiada por UT e simulada em FEM"
    )
    regions = classifier.classify(texto)
    ids = {r.content for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO}
    esperados = {"C30", "CA-50", "CP-V ARI", "UT", "FEM"}
    assert esperados.issubset(ids), (
        f"faltando: {esperados - ids}\nobservado: {ids}"
    )
