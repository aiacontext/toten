"""Tests para aliases de unidade — sistema técnico brasileiro + ASCII.

Cobre `data/unit_aliases_v0.json` e a pré-normalização aplicada por
`parse_unit_composition`. Convenção:

- Abreviações históricas BR: `mt = tf·m` (não `mT = mili-Tesla`).
- Composições com `t`: `t/m → tf/m`, `t·m → tf·m`, etc.
  (`t` standalone permanece massa = 1000 kg).
- Variantes ASCII de superscript: `kgf/cm2 → kgf/cm²`, `m2 → m²`, etc.

Convenção (c) elimina ambiguidade prática em texto de engenharia BR
pré-SI (Süssekind, Pfeil, Schiel) onde `t/m` sempre denota força/comprimento.
"""

from __future__ import annotations

import pytest

from toten.classifier import OntologicalClassifier
from toten.dimensional.algebra import combine_terms
from toten.dimensional.aliases import (
    UnitAliases,
    default_unit_aliases,
    load_unit_aliases,
)
from toten.dimensional.table import default_dim_table
from toten.instantiators import parse_unit_composition
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# Carregamento e estrutura do arquivo
# ---------------------------------------------------------------------------


def test_default_aliases_carrega() -> None:
    a = default_unit_aliases()
    assert isinstance(a, UnitAliases)
    assert a.version
    assert "mt" in a.string_aliases
    assert a.string_aliases["mt"] == "tf·m"


def test_aliases_normalize_passthrough() -> None:
    """Strings não-mapeadas devem retornar idênticas."""
    a = default_unit_aliases()
    assert a.normalize("MPa") == "MPa"
    assert a.normalize("kgf/cm²") == "kgf/cm²"  # já canônica
    assert a.normalize("xyz") == "xyz"


def test_aliases_normalize_strip_externo() -> None:
    """Normalize trabalha sobre exact match — caller strip antes."""
    a = default_unit_aliases()
    assert a.normalize("mt") == "tf·m"
    # Quem chama `parse_unit_composition` já dá .strip(); normalize não strip.
    assert a.normalize(" mt ") == " mt "


# ---------------------------------------------------------------------------
# Abreviações históricas brasileiras
# ---------------------------------------------------------------------------


def test_parse_mt_metro_tonelada() -> None:
    """`mt` (metro·tonelada) parseia como `tf·m` (momento técnico)."""
    terms = parse_unit_composition("mt")
    assert len(terms) == 2
    syms = sorted((t.symbol, t.power) for t in terms)
    assert syms == [("m", 1), ("tf", 1)]


def test_dim_mt_momento_completo() -> None:
    """`13,5 mt` resolve com dim [1,2,-2,...] (momento) e factor 9806.65."""
    terms = parse_unit_composition("mt")
    a = combine_terms(terms, default_dim_table())
    assert a is not None
    assert a.dim == (1, 2, -2, 0, 0, 0, 0)
    assert a.factor == pytest.approx(9806.65)
    assert a.si_unit == "J"  # mesma dim de Joule; render como J é correto


def test_parse_mt_inicio_de_frase_capitalizado() -> None:
    """`Mt` (caso de início de frase) também mapeia para `tf·m`."""
    terms = parse_unit_composition("Mt")
    syms = sorted((t.symbol, t.power) for t in terms)
    assert syms == [("m", 1), ("tf", 1)]


# ---------------------------------------------------------------------------
# Composições com `t` (convenção: força em engenharia BR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit, expected_terms",
    [
        ("t/m",  [("tf", 1), ("m", -1)]),
        ("t·m",  [("tf", 1), ("m", 1)]),
        ("t/m²", [("tf", 1), ("m", -2)]),
        ("t/m³", [("tf", 1), ("m", -3)]),
        ("t/cm²",[("tf", 1), ("cm", -2)]),
    ],
)
def test_composicoes_t_mapeiam_para_tf(unit: str, expected_terms) -> None:
    terms = parse_unit_composition(unit)
    actual = sorted((t.symbol, t.power) for t in terms)
    assert actual == sorted(expected_terms)


def test_t_standalone_permanece_massa() -> None:
    """`t` standalone permanece tonelada métrica (1000 kg, massa)."""
    terms = parse_unit_composition("t")
    assert len(terms) == 1
    assert terms[0].symbol == "t"
    assert terms[0].power == 1
    a = combine_terms(terms, default_dim_table())
    assert a is not None
    assert a.dim == (1, 0, 0, 0, 0, 0, 0)  # massa
    assert a.factor == pytest.approx(1000.0)


def test_t_por_m_dim_forca_linear() -> None:
    """`2 t/m` resolve para força linear (N/m), não massa/comprimento."""
    terms = parse_unit_composition("t/m")
    a = combine_terms(terms, default_dim_table())
    assert a is not None
    assert a.dim == (1, 0, -2, 0, 0, 0, 0)  # força/comprimento (≠ kg/m que seria [1,-1,0,0,0,0,0])
    assert a.factor == pytest.approx(9806.65)


# ---------------------------------------------------------------------------
# Variantes ASCII de superscript
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ascii_form, unicode_form",
    [
        ("kgf/cm2", "kgf/cm²"),
        ("kgf/mm2", "kgf/mm²"),
        ("kgf/m2",  "kgf/m²"),
        ("kgf/m3",  "kgf/m³"),
        ("tf/cm2",  "tf/cm²"),
        ("tf/m2",   "tf/m²"),
        ("tf/m3",   "tf/m³"),
        ("m2",      "m²"),
        ("cm3",     "cm³"),
    ],
)
def test_ascii_equivale_unicode(ascii_form: str, unicode_form: str) -> None:
    """Formas ASCII (`kgf/cm2`) parseiam idênticas às Unicode (`kgf/cm²`)."""
    t_ascii = parse_unit_composition(ascii_form)
    t_unicode = parse_unit_composition(unicode_form)
    assert t_ascii == t_unicode


# ---------------------------------------------------------------------------
# Round-trip via classifier (camada 2 → camada 3)
# ---------------------------------------------------------------------------


def test_classifier_reconhece_mt() -> None:
    """Scanner reconhece `13,5 mt` como QTY e instantiator parseia."""
    clf = OntologicalClassifier()
    regions = clf.classify("M = 13,5 mt na seção crítica")
    qtys = [r for r in regions if r.tipo == TipoNome.GRANDEZA_FISICA]
    assert len(qtys) >= 1
    # localizar a região do "13,5 mt"
    r = next((x for x in qtys if "mt" in x.content), None)
    assert r is not None, f"esperado QTY contendo 'mt' em {qtys}"


def test_classifier_reconhece_kgf_cm2_ascii() -> None:
    """Scanner reconhece variante ASCII `kgf/cm2`."""
    clf = OntologicalClassifier()
    regions = clf.classify("Tensão admissível 2.500 kgf/cm2 (granito)")
    qtys = [r for r in regions if r.tipo == TipoNome.GRANDEZA_FISICA]
    r = next((x for x in qtys if "kgf/cm2" in x.content), None)
    assert r is not None, f"esperado QTY contendo 'kgf/cm2' em {qtys}"


def test_classifier_reconhece_t_por_m_como_qty() -> None:
    """Scanner reconhece `2 t/m` como QTY (força linear via alias)."""
    clf = OntologicalClassifier()
    regions = clf.classify("Carga distribuída q = 2 t/m no apoio esquerdo")
    qtys = [r for r in regions if r.tipo == TipoNome.GRANDEZA_FISICA]
    r = next((x for x in qtys if "t/m" in x.content), None)
    assert r is not None


# ---------------------------------------------------------------------------
# Regressão: composições já suportadas continuam funcionando
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit, expected_factor",
    [
        ("tf·m",     9806.65),
        ("kgf·m",       9.80665),
        ("kgf/cm²", 98066.5),
        ("kgf/mm²", 9806650.0),
        ("tf/m",     9806.65),
    ],
)
def test_regressao_composicoes_existentes(unit: str, expected_factor: float) -> None:
    terms = parse_unit_composition(unit)
    a = combine_terms(terms, default_dim_table())
    assert a is not None
    assert a.factor == pytest.approx(expected_factor)


# ---------------------------------------------------------------------------
# Carregamento explícito por caminho
# ---------------------------------------------------------------------------


def test_load_unit_aliases_arquivo_inexistente(tmp_path) -> None:
    fake = tmp_path / "naoexiste.json"
    with pytest.raises(FileNotFoundError):
        load_unit_aliases(fake)
