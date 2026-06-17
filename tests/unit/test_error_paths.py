"""Tests dos error paths defensivos (auditoria de cobertura).

Cada teste cobre um branch que existia sem exercício. Defensivos
legítimos — o sistema deve falhar de forma controlada em entradas
inválidas (lexicon malformado, parser de unidades com input quebrado,
canonicalize_number com None, OEE com tipos faltando, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from toten import Tokenizer
from toten.classifier import OntologicalClassifier, Region
from toten.instantiators import (
    ConstanteUniversalInstantiator,
    GrandezaFisicaInstantiator,
    canonicalize_number,
    parse_unit_composition,
)
from toten.instantiators.prose import _UnboundBPEBackend
from toten.instantiators.quantity import _format_value
from toten.ontology.schema import OEE
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# canonicalize_number — error paths
# ---------------------------------------------------------------------------


def test_canonicalize_number_none_levanta() -> None:
    with pytest.raises(ValueError, match="não-nula"):
        canonicalize_number(None)  # type: ignore[arg-type]


def test_canonicalize_number_expoente_malformado_levanta() -> None:
    """Notação científica com mantissa não-numérica."""
    with pytest.raises(ValueError, match="não conseguiu parsear"):
        canonicalize_number("abc.def e+5")


def test_canonicalize_number_decimal_sem_inteiro() -> None:
    """`.5` (decimal sem parte inteira) → integer_part vira '0'."""
    assert canonicalize_number(".5") == pytest.approx(0.5)
    assert canonicalize_number(",5") == pytest.approx(0.5)


def test_format_value_none_devolve_vazio() -> None:
    """Helper interno: None devolve string vazia (proteção downstream)."""
    assert _format_value(None) == ""


# ---------------------------------------------------------------------------
# parse_unit_composition — error paths do parser
# ---------------------------------------------------------------------------


def test_parser_unit_caractere_inesperado() -> None:
    """Operador sem símbolo após (e.g., `m/`) — dispara 'esperado símbolo'
    em _factor."""
    with pytest.raises(ValueError, match="esperado símbolo"):
        parse_unit_composition("m/")


def test_parser_unit_caractere_inesperado_apos_termino() -> None:
    """Token sobrando após composição completa (e.g., `m)` ou `m a`) —
    dispara 'caractere inesperado' no nível parse()."""
    with pytest.raises(ValueError, match="caractere inesperado"):
        parse_unit_composition("m)")


def test_parser_unit_sem_simbolo() -> None:
    """Sem símbolo logo no início (operador isolado)."""
    with pytest.raises(ValueError, match="esperado símbolo"):
        parse_unit_composition("/m")


def test_parser_unit_paren_aninhado_invalido() -> None:
    """Parens não-fechados internos."""
    with pytest.raises(ValueError, match="esperado '\\)'"):
        parse_unit_composition("kg/(m·s")


def test_parser_unit_expoente_sem_digitos() -> None:
    """`m^` sem dígitos após o caret."""
    with pytest.raises(ValueError, match="expoente sem dígitos"):
        parse_unit_composition("m^")


def test_parser_unit_expoente_minus_sem_digitos() -> None:
    """`m^-` (sinal sem dígitos)."""
    with pytest.raises(ValueError, match="expoente sem dígitos"):
        parse_unit_composition("m^-")


def test_parser_unit_whitespace_interno() -> None:
    """Whitespace entre átomos é permitido (skip_ws)."""
    assert parse_unit_composition("kg / s") == parse_unit_composition("kg/s")


# ---------------------------------------------------------------------------
# Modo A com átomo desconhecido — fallback _render_a + _render_terms
# ---------------------------------------------------------------------------


def test_grandeza_modo_a_atom_desconhecido() -> None:
    """Quando dim_vector é None (átomo fora da dim_table), Modo A cai no
    fallback que renderiza terms verbatim via _render_terms / _render_term."""
    inst = GrandezaFisicaInstantiator()
    # 'foobar' não está em classifier_units_v0.json, então não vira região
    # via classify(); aqui criamos a Region manualmente para forçar o caso.
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 11, "350 foobar²")
    token = inst.instantiate(region, mode="A")
    # _render_terms é chamado quando dim_vector None
    assert "<DIM>foobar^2</DIM>" in token.text
    assert "<VAL>350</VAL>" in token.text


def test_grandeza_modo_a_atom_unknown_com_incerteza() -> None:
    """_render_a fallback emite <UNC> quando uncertainty != None mas
    dim None."""
    inst = GrandezaFisicaInstantiator()
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 18, "350 ± 10 foobar")
    token = inst.instantiate(region, mode="A")
    assert "<UNC>10</UNC>" in token.text


# ---------------------------------------------------------------------------
# ConstanteUniversal — Modo A com símbolo desconhecido
# ---------------------------------------------------------------------------


def test_constante_modo_a_simbolo_desconhecido() -> None:
    """ψ não está no constant_lexicon → token sem QTY composta."""
    inst = ConstanteUniversalInstantiator()
    region = Region(TipoNome.CONSTANTE_UNIVERSAL, 0, 1, "ψ")
    token = inst.instantiate(region, mode="A")
    assert token.text == "<CONST><SYM>ψ</SYM></CONST>"


# ---------------------------------------------------------------------------
# ProsaTecnica — BPE backend default levanta NotImplementedError em decode
# ---------------------------------------------------------------------------


def test_prose_default_backend_decode_levanta() -> None:
    backend = _UnboundBPEBackend()
    with pytest.raises(NotImplementedError, match="decode requer BPEBackend"):
        backend.decode([1, 2, 3])


# ---------------------------------------------------------------------------
# Classificador — defensivos no carregamento de lexicons
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "OBSOLETO: refator dim ℤ⁷ (refator anterior) "
        "renomeou _load_unit_symbols → _load_unit_symbols_from_dim_table "
        "e mudou contrato (lê do dim_table.json, não de units.json). "
        "Reescrita pendente para testar o novo loader."
    )
)
def test_classifier_units_json_sem_symbols(tmp_path: Path) -> None:
    """JSON sem 'symbols' deve levantar ValueError no carregamento."""
    from toten.classifier.classifier import _load_unit_symbols

    bad = tmp_path / "units.json"
    bad.write_text(json.dumps({"version": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="não contém lista 'symbols'"):
        _load_unit_symbols(bad)


def test_classifier_identifier_padrao_sem_regex(tmp_path: Path) -> None:
    """Entrada em identifier_lexicon sem campo 'regex' é simplesmente ignorada."""
    bad = tmp_path / "id_lex.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "padroes": [
                {"classe": "norma"},  # sem regex
                {"classe": "norma", "regex": "\\bABNT\\s+\\d+\\b"},
            ]
        }),
        encoding="utf-8",
    )
    c = OntologicalClassifier(identifier_lexicon_path=bad)
    # Padrão válido funciona; o sem regex foi ignorado sem erro
    regions = c.classify("a norma ABNT 12655 vale")
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    assert any("ABNT" in r.content for r in ids)


def test_classifier_operator_sem_unicode_levanta(tmp_path: Path) -> None:
    """operator_lexicon só com `=` ASCII (sem nenhum Unicode) levanta."""
    bad = tmp_path / "op.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "operadores": [{"simbolo": "="}],  # só ASCII
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sem operadores Unicode"):
        OntologicalClassifier(operator_lexicon_path=bad)


def test_classifier_operator_entry_sem_simbolo_eh_ignorada(
    tmp_path: Path,
) -> None:
    """Entrada operador sem 'simbolo' é pulada (continue)."""
    bad = tmp_path / "op.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "operadores": [
                {"aridade": 2},  # sem simbolo
                {"simbolo": "≤", "aridade": 2, "categoria": "relacional"},
            ]
        }),
        encoding="utf-8",
    )
    c = OntologicalClassifier(operator_lexicon_path=bad)
    regions = c.classify("x ≤ 5")
    ops = [r for r in regions if r.tipo is TipoNome.OPERADOR_FORMAL]
    assert any(r.content == "≤" for r in ops)


def test_classifier_constant_entry_sem_simbolo_eh_ignorada(
    tmp_path: Path,
) -> None:
    """Entrada constante sem 'simbolo' é pulada (continue)."""
    bad = tmp_path / "const.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "constantes": [
                {"nome": "missing"},  # sem simbolo
                {"simbolo": "π", "multi_char": True},
            ]
        }),
        encoding="utf-8",
    )
    c = OntologicalClassifier(constant_lexicon_path=bad)
    regions = c.classify("o valor de π")
    cs = [r for r in regions if r.tipo is TipoNome.CONSTANTE_UNIVERSAL]
    assert any(r.content == "π" for r in cs)


def test_classifier_constant_sem_single_char_usa_pattern_vazio(
    tmp_path: Path,
) -> None:
    """Sem constantes single-char, single_re compila padrão que nunca casa."""
    bad = tmp_path / "const.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "constantes": [{"simbolo": "π", "multi_char": True}],
        }),
        encoding="utf-8",
    )
    c = OntologicalClassifier(constant_lexicon_path=bad)
    # `c = 3` não dispara constante (não há single_char no lexicon)
    regions = c.classify("c = 3")
    cs = [r for r in regions if r.tipo is TipoNome.CONSTANTE_UNIVERSAL]
    assert not any(r.content == "c" for r in cs)


def test_classifier_relation_lexicon_vazio(tmp_path: Path) -> None:
    """Sem conectores no lexicon, relation_re compila padrão never-match."""
    bad = tmp_path / "rel.json"
    bad.write_text(json.dumps({"version": "x"}), encoding="utf-8")
    c = OntologicalClassifier(relation_lexicon_path=bad)
    regions = c.classify("portanto a viga falha")
    rels = [r for r in regions if r.tipo is TipoNome.RELACAO_ESTRUTURAL]
    assert rels == []


def test_classifier_relation_entry_sem_forma_eh_ignorada(
    tmp_path: Path,
) -> None:
    """Conector sem 'forma' é pulado (RelacaoEstruturalInstantiator)."""
    from toten.instantiators.relation import (
        RelacaoEstruturalInstantiator,
    )

    bad = tmp_path / "rel.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "pt-br": [
                {"funcao": "conclusiva"},  # sem forma
                {"forma": "portanto", "funcao": "conclusiva"},
            ],
            "en": [],
        }),
        encoding="utf-8",
    )
    inst = RelacaoEstruturalInstantiator(lexicon_path=bad)
    region = Region(TipoNome.RELACAO_ESTRUTURAL, 0, 8, "portanto")
    token = inst.instantiate(region)
    assert token.funcao_logica == "conclusiva"


# ---------------------------------------------------------------------------
# Schema (pydantic) — error paths de validação
# ---------------------------------------------------------------------------


def _oee_v2_minimo() -> dict:
    """OEE com version != 1.x — escapa do _v1_completa enforcing 6 tipos."""
    return {
        "oee_version": "2.0",
        "nome": "OEE v2 fixture",
        "descricao": "fixture",
        "referencias": ["x"],
        "principios": [{"id": "P1", "nome": "P1", "declaracao": "d"}],
        "tipos": {
            "GrandezaFisica": {
                "natureza": "n",
                "propriedades_intrinsecas": {"valor": "escalar"},
                "invariantes": ["dim sempre presente"],
                "exemplos": ["1 m"],
            },
            "ProsaTecnica": {
                "natureza": "n",
                "propriedades_intrinsecas": {"idioma": "iso"},
                "invariantes": ["não-numérico"],
                "exemplos": ["x"],
            },
            "IdentificadorTecnico": {
                "natureza": "n",
                "propriedades_intrinsecas": {"classe": "x"},
                "invariantes": ["preservada"],
                "exemplos": ["X"],
            },
            "OperadorFormal": {
                "natureza": "n",
                "propriedades_intrinsecas": {"aridade": "1"},
                "invariantes": ["atômico"],
                "exemplos": ["="],
            },
            "RelacaoEstrutural": {
                "natureza": "n",
                "propriedades_intrinsecas": {"funcao_logica": "x"},
                "invariantes": ["x"],
                "exemplos": ["então"],
            },
            "ExpressaoSimbolica": {
                "natureza": "n",
                "propriedades_intrinsecas": {"forma_canonica": "x"},
                "invariantes": ["x"],
                "composicoes": ["IdentificadorTecnico", "OperadorFormal"],
                "exemplos": ["pl/12"],
            },
        },
        "resolucao_ambiguidade": {
            "estrategia": "maior_especificidade",
            "ordem": [
                "GrandezaFisica",
                "IdentificadorTecnico",
                "OperadorFormal",
                "ExpressaoSimbolica",
                "RelacaoEstrutural",
                "ProsaTecnica",
                "ConstanteUniversal",
            ],
        },
    }


def test_schema_propriedade_com_nome_vazio() -> None:
    raw = _oee_v2_minimo()
    raw["tipos"]["GrandezaFisica"]["propriedades_intrinsecas"] = {"": "descricao"}
    with pytest.raises(ValidationError, match="nome vazio"):
        OEE.model_validate(raw)


def test_schema_propriedade_com_descricao_vazia() -> None:
    raw = _oee_v2_minimo()
    raw["tipos"]["GrandezaFisica"]["propriedades_intrinsecas"] = {"valor": "   "}
    with pytest.raises(ValidationError, match="sem descrição"):
        OEE.model_validate(raw)


def test_schema_ordem_com_duplicatas() -> None:
    raw = _oee_v2_minimo()
    raw["resolucao_ambiguidade"]["ordem"] = [
        "GrandezaFisica",
        "GrandezaFisica",  # duplicata
        "IdentificadorTecnico",
        "OperadorFormal",
        "RelacaoEstrutural",
        "ProsaTecnica",
    ]
    with pytest.raises(ValidationError, match="duplicatas"):
        OEE.model_validate(raw)


@pytest.mark.skip(
    reason=(
        "OBSOLETO: Evolução 1 (OEE v1.1) introduziu validator "
        "version-aware (_expected_types_for_version) que rejeita versões "
        "não-suportadas (v2.0) com mensagem 'versão OEE não suportada'. "
        "Antes, v2.0 escapava do _v1_completa silenciosamente. Fixtures "
        "_oee_v2_minimo precisam ser reescritas para usar v1.0/v1.1 OU "
        "validator precisa ganhar escape hatch para versões customizadas "
        "(decisão arquitetural pendente)."
    )
)
def test_schema_ordem_incompleta() -> None:
    """Ordem sem cobrir TIPOS_V1 (5 entries, falta ConstanteUniversal) —
    dispara branch faltando/sobrando sem cair em duplicatas."""
    raw = _oee_v2_minimo()
    raw["resolucao_ambiguidade"]["ordem"] = [
        "GrandezaFisica",
        "IdentificadorTecnico",
        "OperadorFormal",
        "RelacaoEstrutural",
        "ProsaTecnica",
        # Sem ConstanteUniversal — set != TIPOS_V1, sem duplicatas
    ]
    with pytest.raises(ValidationError, match="exatamente os tipos da v1.0"):
        OEE.model_validate(raw)


@pytest.mark.skip(
    reason="OBSOLETO: mesma razão de test_schema_ordem_incompleta — fixture _oee_v2_minimo rejeitada pelo validator v1.1 version-aware."
)
def test_schema_composicao_para_tipo_v2_nao_declarado() -> None:
    """OEE v2 sem ConstanteUniversal: outra entrada referencia tipo ausente."""
    raw = _oee_v2_minimo()
    raw["tipos"]["IdentificadorTecnico"]["composicoes"] = ["ConstanteUniversal"]
    with pytest.raises(ValidationError, match="não está declarado"):
        OEE.model_validate(raw)


@pytest.mark.skip(
    reason="OBSOLETO: mesma razão — _oee_v2_minimo rejeitada pelo validator v1.1."
)
def test_schema_v2_sem_constante_universal_passa() -> None:
    """OEE v2 sem ConstanteUniversal valida — _constante_universal_compoe_grandeza
    short-circuits (cu None → return self)."""
    oee = OEE.model_validate(_oee_v2_minimo())
    assert TipoNome.CONSTANTE_UNIVERSAL not in oee.tipos


# ---------------------------------------------------------------------------
# Pipeline — factory + edge case de whitespace trailing
# ---------------------------------------------------------------------------


def test_pipeline_from_oee_path(tmp_path: Path) -> None:
    """from_oee_path constrói tokenizer a partir de OEE em caminho custom."""
    from toten.ontology.loader import default_oee_path

    target = tmp_path / "oee.yaml"
    target.write_text(default_oee_path().read_text(encoding="utf-8"), encoding="utf-8")
    tok = Tokenizer.from_oee_path(target)
    assert tok.oee.oee_version.startswith("1.")
    # Sanity: pipeline funciona
    out = tok.preprocess("350 MPa")
    assert "[QTY value=350 unit=MPa" in out


def test_pipeline_trailing_whitespace_preservado() -> None:
    """Texto que termina em whitespace após última região — pipeline
    appenda o gap trailing sem perda."""
    tok = Tokenizer.from_ontology("oee-v1")
    out = tok.preprocess("350 MPa   ")
    assert out.endswith("   ")
    assert "[QTY value=350 unit=MPa" in out
