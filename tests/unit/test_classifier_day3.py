"""Testes Dia 3 — cobertura completa da Camada 2.

Adiciona casos para ConstanteUniversal, IdentificadorTecnico,
OperadorFormal, RelacaoEstrutural e ProsaTecnica (residual), em PT-BR
e EN, e valida resolução de sobreposição via priority da OEE.
"""

from __future__ import annotations

import pytest

from toten.classifier import OntologicalClassifier
from toten.ontology.types import TipoNome


@pytest.fixture(scope="module")
def classifier() -> OntologicalClassifier:
    return OntologicalClassifier()


def _tipos(regions) -> list[TipoNome]:
    return [r.tipo for r in regions]


def _contents(regions) -> list[str]:
    return [r.content for r in regions]


# ---------------------------------------------------------------------------
# IdentificadorTecnico
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "texto",
    [
        "conforme ABNT NBR 12655 a resistência",
        "norma ABNT NBR ISO 9001:2015 aplicada",
        "atende NBR 6118 para concreto",
        "ISO 9001 certificada",
        "tubulação ASME B31.3",
        "consumível AWS A5.18",
        "aço EN 10025-2 estrutural",
    ],
)
def test_identificador_norma(classifier: OntologicalClassifier, texto: str) -> None:
    regions = classifier.classify(texto)
    assert any(r.tipo is TipoNome.IDENTIFICADOR_TECNICO for r in regions)


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("aço AISI 304", "AISI 304"),
        ("AISI 1045 tratado", "AISI 1045"),
        ("perfil ASTM A36 estrutural", "ASTM A36"),
        ("placa ASTM A572-50", "ASTM A572-50"),
        ("eixo SAE 1045 forjado", "SAE 1045"),
        ("eletrodo ER70S-6", "ER70S-6"),
        ("liga Al 6061-T6 anodizada", "Al 6061-T6"),
        ("Ti-6Al-4V grau 5", "Ti-6Al-4V"),
    ],
)
def test_identificador_material(
    classifier: OntologicalClassifier, texto: str, esperado: str
) -> None:
    regions = classifier.classify(texto)
    matches = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert any(r.content == esperado for r in matches), (
        f"esperado '{esperado}' em {_contents(matches)}"
    )


@pytest.mark.parametrize("simbolo", ["Re", "Nu", "Pr", "Fr", "Pe", "Ma", "We", "Bi"])
def test_identificador_adimensional(
    classifier: OntologicalClassifier, simbolo: str
) -> None:
    texto = f"o número de {simbolo} indica regime"
    regions = classifier.classify(texto)
    matches = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert any(r.content == simbolo for r in matches)


def test_identificador_adimensional_nao_dispara_em_palavra() -> None:
    """`Pr` em `Prata` ou `Pe` em `Pedro` NÃO deve casar."""
    c = OntologicalClassifier()
    for texto in ["Prata é metal", "Pedro escreveu", "Reverso de direção"]:
        regions = c.classify(texto)
        ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
        assert ids == [], f"falso positivo em '{texto}': {_contents(ids)}"


# ---------------------------------------------------------------------------
# OperadorFormal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simbolo", ["≤", "≥", "≠", "≈", "≡", "→", "⇒", "∑", "∫", "∂", "∇"])
def test_operador_unicode(classifier: OntologicalClassifier, simbolo: str) -> None:
    texto = f"x {simbolo} y"
    regions = classifier.classify(texto)
    ops = [r for r in regions if r.tipo is TipoNome.OPERADOR_FORMAL]
    assert any(r.content == simbolo for r in ops)


def test_operador_igual_em_equacao(classifier: OntologicalClassifier) -> None:
    regions = classifier.classify("σ_y = 350 MPa")
    ops = [r for r in regions if r.tipo is TipoNome.OPERADOR_FORMAL]
    assert any(r.content == "=" for r in ops)


def test_operador_igual_nao_dispara_em_token_contiguo(
    classifier: OntologicalClassifier,
) -> None:
    """`==` em código não deve produzir OperadorFormal."""
    regions = classifier.classify("if x == 1 then")
    ops = [r for r in regions if r.tipo is TipoNome.OPERADOR_FORMAL]
    assert ops == []


# ---------------------------------------------------------------------------
# ConstanteUniversal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simbolo", ["π", "ℏ", "k_B", "N_A", "σ_SB", "R_g"])
def test_constante_multi_char_sempre_emite(
    classifier: OntologicalClassifier, simbolo: str
) -> None:
    texto = f"considerando {simbolo} no cálculo"
    regions = classifier.classify(texto)
    cs = [r for r in regions if r.tipo is TipoNome.CONSTANTE_UNIVERSAL]
    assert any(r.content == simbolo for r in cs), f"{simbolo} não emitida em '{texto}'"


def test_constante_single_char_nao_auto_capturada(
    classifier: OntologicalClassifier,
) -> None:
    """Decisão ontológica: letras únicas (c, R, G, e)
    NÃO são auto-capturadas como CONST porque em texto técnico de engenharia
    denotam VARIÁVEIS muito mais frequentemente que constantes universais
    (c=amortecimento, R=resistência, G=módulo de cisalhamento). Capturar
    como CONST induz erro físico crítico (valor errado entregue ao LLM).
    Engenheiros que querem constante universal escrevem por nome (`R_g`,
    `velocidade da luz no vácuo`)."""
    regions = classifier.classify("c = 299792458 m/s")
    cs = [r for r in regions if r.tipo is TipoNome.CONSTANTE_UNIVERSAL]
    assert not any(r.content == "c" for r in cs), (
        "letra única 'c' NÃO deve virar CONST automaticamente"
    )


def test_constante_single_char_nao_dispara_em_prosa(
    classifier: OntologicalClassifier,
) -> None:
    """`e` em prosa PT-BR como conjunção NÃO deve emitir ConstanteUniversal."""
    regions = classifier.classify("a viga e a coluna estão alinhadas")
    cs = [r for r in regions if r.tipo is TipoNome.CONSTANTE_UNIVERSAL]
    assert cs == []


# ---------------------------------------------------------------------------
# RelacaoEstrutural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "conector",
    ["portanto", "logo", "então", "dado que", "tal que", "se e somente se"],
)
def test_relacao_pt_br(classifier: OntologicalClassifier, conector: str) -> None:
    texto = f"a hipótese vale; {conector} a conclusão decorre"
    regions = classifier.classify(texto)
    rels = [r for r in regions if r.tipo is TipoNome.RELACAO_ESTRUTURAL]
    assert any(conector in r.content.lower() for r in rels)


@pytest.mark.parametrize(
    "conector",
    ["therefore", "hence", "thus", "given that", "such that", "if and only if"],
)
def test_relacao_en(classifier: OntologicalClassifier, conector: str) -> None:
    texto = f"the hypothesis holds; {conector} the conclusion follows"
    regions = classifier.classify(texto)
    rels = [r for r in regions if r.tipo is TipoNome.RELACAO_ESTRUTURAL]
    assert any(conector in r.content.lower() for r in rels)


def test_relacao_multi_token_prefere_casamento_mais_longo(
    classifier: OntologicalClassifier,
) -> None:
    """`se e somente se` vence `se` isolado (que nem está no lexicon)."""
    regions = classifier.classify("vale x se e somente se y vale")
    rels = [r for r in regions if r.tipo is TipoNome.RELACAO_ESTRUTURAL]
    assert any(r.content.lower() == "se e somente se" for r in rels)


# ---------------------------------------------------------------------------
# ProsaTecnica (residual)
# ---------------------------------------------------------------------------


def test_prosa_emitida_em_gap(classifier: OntologicalClassifier) -> None:
    texto = "a tensão de 350 MPa supera o limite"
    regions = classifier.classify(texto)
    prosas = [r for r in regions if r.tipo is TipoNome.PROSA_TECNICA]
    assert len(prosas) >= 2
    assert any("tensão de" in r.content for r in prosas)
    assert any("supera o limite" in r.content for r in prosas)


def test_prosa_nao_emitida_para_whitespace_isolado(
    classifier: OntologicalClassifier,
) -> None:
    """Gaps com apenas whitespace não geram ProsaTecnica."""
    regions = classifier.classify("350 MPa ≥ 200 MPa")
    prosas = [r for r in regions if r.tipo is TipoNome.PROSA_TECNICA]
    assert prosas == []


def test_texto_inteiramente_prosa(classifier: OntologicalClassifier) -> None:
    texto = "a viga foi analisada quanto à flexão"
    regions = classifier.classify(texto)
    assert len(regions) == 1
    assert regions[0].tipo is TipoNome.PROSA_TECNICA
    assert regions[0].content == texto


# ---------------------------------------------------------------------------
# Resolução de sobreposição (priority por OEE)
# ---------------------------------------------------------------------------


def test_priority_carregada_da_oee(classifier: OntologicalClassifier) -> None:
    p = classifier.priority
    assert p[TipoNome.GRANDEZA_FISICA] < p[TipoNome.CONSTANTE_UNIVERSAL]
    assert p[TipoNome.CONSTANTE_UNIVERSAL] < p[TipoNome.IDENTIFICADOR_TECNICO]
    assert p[TipoNome.IDENTIFICADOR_TECNICO] < p[TipoNome.OPERADOR_FORMAL]
    assert p[TipoNome.OPERADOR_FORMAL] < p[TipoNome.RELACAO_ESTRUTURAL]
    assert p[TipoNome.RELACAO_ESTRUTURAL] < p[TipoNome.PROSA_TECNICA]


@pytest.mark.skip(
    reason=(
        "OBSOLETO: refator anterior — '2π' fica em SYM "
        "(composição mediada) em vez de CONST atômico. Decisão "
        "documentada na docstring do classifier (uso vs menção: "
        "adjacente a marca matemática vira SYM). Reescrita pendente."
    )
)
def test_constante_unicode_apos_digito_emite() -> None:
    """`2π` mantém `π` como ConstanteUniversal — dígito imediatamente antes
    é contexto numérico legítimo, não tem que invalidar o casamento."""
    c = OntologicalClassifier()
    regions = c.classify("perímetro = 2π m")
    cs = [r for r in regions if r.tipo is TipoNome.CONSTANTE_UNIVERSAL]
    assert any(r.content == "π" for r in cs)


def test_spans_nao_se_sobrepoem_apos_resolucao(
    classifier: OntologicalClassifier,
) -> None:
    texto = "para AISI 304, σ_y = 200 MPa, portanto ok"
    regions = classifier.classify(texto)
    for a, b in zip(regions, regions[1:], strict=False):
        assert a.end <= b.start


def test_regions_cobrem_texto_em_ordem(classifier: OntologicalClassifier) -> None:
    texto = "350 MPa é o limite; ≥ 200 MPa, portanto ok"
    regions = classifier.classify(texto)
    starts = [r.start for r in regions]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Cenário integrador realista
# ---------------------------------------------------------------------------


def test_sentenca_engenharia_completa(classifier: OntologicalClassifier) -> None:
    texto = "para AISI 304 a tensão σ_y = 215 MPa, portanto ≥ 200 MPa"
    regions = classifier.classify(texto)
    tipos = _tipos(regions)
    assert TipoNome.IDENTIFICADOR_TECNICO in tipos
    assert TipoNome.OPERADOR_FORMAL in tipos
    assert TipoNome.GRANDEZA_FISICA in tipos
    assert TipoNome.RELACAO_ESTRUTURAL in tipos
    assert TipoNome.PROSA_TECNICA in tipos


def test_sentenca_engenharia_completa_en(classifier: OntologicalClassifier) -> None:
    texto = "for AISI 304 the yield σ_y = 215 MPa, therefore ≥ 200 MPa"
    regions = classifier.classify(texto)
    tipos = _tipos(regions)
    assert TipoNome.IDENTIFICADOR_TECNICO in tipos
    assert TipoNome.OPERADOR_FORMAL in tipos
    assert TipoNome.GRANDEZA_FISICA in tipos
    assert TipoNome.RELACAO_ESTRUTURAL in tipos
    assert TipoNome.PROSA_TECNICA in tipos


def test_priority_obedece_oee_em_caso_concreto(
    classifier: OntologicalClassifier,
) -> None:
    """Garante que GrandezaFisica `300 K` vence Identificador `K`-like."""
    regions = classifier.classify("operação a 300 K em regime estacionário")
    grandezas = [r for r in regions if r.tipo is TipoNome.GRANDEZA_FISICA]
    assert any(r.content == "300 K" for r in grandezas)
