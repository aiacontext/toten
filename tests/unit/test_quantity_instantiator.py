"""Testes Dia 4 — GrandezaFisicaInstantiator (Camada 3, tipo central).

Cobre:
- canonicalize_number: locale PT-BR (`287,4`), US (`287.4`), thousands
  (`1.234,5` / `1,234.5`), notação científica, sinal Unicode.
- parse_unit_composition: literais (`MPa`), compostos (`kgf/cm²`,
  `W/(m²·K⁴)`, `m³·K/(s²·A²)`), unidades com superscript Unicode e
  caret (`^-3`), parens, operadores `·` / `*` / `/`.
- Quantity + QuantityToken: Modo B textual, Modo A trinitário,
  invariantes (unit_text não-vazio, ao menos um UnitTerm).
- Round-trip Camada 2 → Camada 3: regiões detectadas pelo classifier
  são parseáveis pelo instantiator.
"""

from __future__ import annotations

import pytest

from toten.classifier import OntologicalClassifier, Region
from toten.instantiators import (
    GrandezaFisicaInstantiator,
    Quantity,
    UnitTerm,
    canonicalize_number,
    parse_grandeza,
    parse_unit_composition,
)
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# canonicalize_number — locale awareness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("350", 350.0),
        ("0", 0.0),
        ("0.5", 0.5),
        ("287.4", 287.4),
        ("287,4", 287.4),
        ("1.234,5", 1234.5),
        ("1,234.5", 1234.5),
        ("12.345,678", 12345.678),
        ("12,345.678", 12345.678),
        ("-50", -50.0),
        ("+50", 50.0),
        ("−50", -50.0),
        ("1.5e-3", 1.5e-3),
        ("1,5e-3", 1.5e-3),
        ("2e8", 2e8),
        ("3.50E+08", 3.50e8),
    ],
)
def test_canonicalize_number_locale(entrada: str, esperado: float) -> None:
    resultado = canonicalize_number(entrada)
    assert resultado == pytest.approx(esperado)


def test_canonicalize_number_rejeita_vazio() -> None:
    with pytest.raises(ValueError, match="não-vazia"):
        canonicalize_number("")
    with pytest.raises(ValueError, match="não-vazia"):
        canonicalize_number("   ")


# ---------------------------------------------------------------------------
# parse_unit_composition — átomos, potências, composição, parens
# ---------------------------------------------------------------------------


def _terms(*pairs: tuple[str, int]) -> tuple[UnitTerm, ...]:
    return tuple(UnitTerm(symbol=s, power=p) for s, p in pairs)


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("MPa", _terms(("MPa", 1))),
        ("kg", _terms(("kg", 1))),
        ("°C", _terms(("°C", 1))),
        ("Pa·s", _terms(("Pa", 1), ("s", 1))),
        ("m/s", _terms(("m", 1), ("s", -1))),
        ("m/s²", _terms(("m", 1), ("s", -2))),
        ("kgf/cm²", _terms(("kgf", 1), ("cm", -2))),
        ("W/(m²·K⁴)", _terms(("W", 1), ("m", -2), ("K", -4))),
        ("J/(kg·K)", _terms(("J", 1), ("kg", -1), ("K", -1))),
        ("tf·m", _terms(("tf", 1), ("m", 1))),
        ("kgf·cm", _terms(("kgf", 1), ("cm", 1))),
        ("m³·K/(s²·A²)", _terms(("m", 3), ("K", 1), ("s", -2), ("A", -2))),
        ("s⁻¹", _terms(("s", -1))),
        ("mol⁻¹", _terms(("mol", -1))),
        ("m·s⁻¹", _terms(("m", 1), ("s", -1))),
        ("kg·m²·s⁻³", _terms(("kg", 1), ("m", 2), ("s", -3))),
    ],
)
def test_parse_unit_composition(
    entrada: str, esperado: tuple[UnitTerm, ...]
) -> None:
    assert parse_unit_composition(entrada) == esperado


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("m^2", _terms(("m", 2))),
        ("m^-3", _terms(("m", -3))),
        ("kg^2/s^3", _terms(("kg", 2), ("s", -3))),
    ],
)
def test_parse_unit_caret_power(
    entrada: str, esperado: tuple[UnitTerm, ...]
) -> None:
    assert parse_unit_composition(entrada) == esperado


def test_parse_unit_composto_desconhecido_via_atomos_conhecidos() -> None:
    """Unidade composta NÃO enumerada literalmente, mas formada por
    átomos conhecidos, é parseável estruturalmente. Este é o ganho de
    robustez do framework para o long tail de unidades."""
    terms = parse_unit_composition("tf/(m²·°C)")
    assert terms == _terms(("tf", 1), ("m", -2), ("°C", -1))


def test_parse_unit_rejeita_vazio() -> None:
    with pytest.raises(ValueError, match="não-vazia"):
        parse_unit_composition("")


def test_parse_unit_rejeita_paren_nao_fechado() -> None:
    with pytest.raises(ValueError, match="esperado '\\)'"):
        parse_unit_composition("kg/(m·s")


# ---------------------------------------------------------------------------
# parse_grandeza — integra número + incerteza + unidade
# ---------------------------------------------------------------------------


def test_parse_grandeza_simples() -> None:
    q = parse_grandeza("350 MPa")
    assert q.value == 350.0
    assert q.uncertainty is None
    assert q.unit_text == "MPa"
    assert q.unit_terms == _terms(("MPa", 1))


def test_parse_grandeza_com_incerteza_unicode() -> None:
    q = parse_grandeza("350 ± 10 MPa")
    assert q.value == 350.0
    assert q.uncertainty == 10.0
    assert q.unit_text == "MPa"


def test_parse_grandeza_com_incerteza_ascii() -> None:
    q = parse_grandeza("350 +/- 10 MPa")
    assert q.value == 350.0
    assert q.uncertainty == 10.0


def test_parse_grandeza_decimal_pt_br() -> None:
    q = parse_grandeza("287,4 MPa")
    assert q.value == pytest.approx(287.4)
    assert q.unit_text == "MPa"


def test_parse_grandeza_notacao_cientifica() -> None:
    q = parse_grandeza("1.5e-3 Pa·s")
    assert q.value == pytest.approx(1.5e-3)
    assert q.unit_terms == _terms(("Pa", 1), ("s", 1))


def test_parse_grandeza_unit_composta_br() -> None:
    q = parse_grandeza("350 kgf/cm²")
    assert q.value == 350.0
    assert q.unit_terms == _terms(("kgf", 1), ("cm", -2))


def test_parse_grandeza_unit_long_tail() -> None:
    """Mesmo composto exótico mas com átomos conhecidos é parseável."""
    q = parse_grandeza("10 tf/(m²·°C)")
    assert q.value == 10.0
    assert q.unit_terms == _terms(("tf", 1), ("m", -2), ("°C", -1))


def test_parse_grandeza_rejeita_vazio() -> None:
    with pytest.raises(ValueError, match="não-vazia"):
        parse_grandeza("")


def test_parse_grandeza_rejeita_sem_unidade() -> None:
    with pytest.raises(ValueError, match="não conseguiu casar"):
        parse_grandeza("350")


def test_parse_grandeza_aceita_sem_espaco_entre_num_e_unit() -> None:
    """Consistência com Camada 2: `540°C`, `350MPa`, `10tf` sem espaço
    entre número e unidade são casos legítimos. Antes da correção, o
    parser de Camada 3 exigia \\s+, rejeitando esses casos."""
    q1 = parse_grandeza("540°C")
    assert q1.value == 540.0
    assert q1.unit_text == "°C"
    assert q1.dim_vector == (0, 0, 0, 0, 1, 0, 0)

    q2 = parse_grandeza("350MPa")
    assert q2.value == 350.0
    assert q2.unit_text == "MPa"

    q3 = parse_grandeza("1.5e-3Pa·s")
    assert q3.value == pytest.approx(1.5e-3)
    assert q3.unit_text == "Pa·s"


# ---------------------------------------------------------------------------
# Quantity — invariantes
# ---------------------------------------------------------------------------


def test_quantity_imutavel() -> None:
    q = Quantity(
        value=350.0,
        unit_text="MPa",
        unit_terms=_terms(("MPa", 1)),
    )
    with pytest.raises(AttributeError):
        q.value = 100.0  # type: ignore[misc]


def test_quantity_rejeita_unit_text_vazio() -> None:
    with pytest.raises(ValueError, match="unit_text"):
        Quantity(value=1.0, unit_text="  ", unit_terms=_terms(("m", 1)))


def test_quantity_rejeita_unit_terms_vazio() -> None:
    with pytest.raises(ValueError, match="UnitTerm"):
        Quantity(value=1.0, unit_text="m", unit_terms=())


# ---------------------------------------------------------------------------
# QuantityToken — Modo A vs Modo B
# ---------------------------------------------------------------------------


@pytest.fixture
def instantiator() -> GrandezaFisicaInstantiator:
    return GrandezaFisicaInstantiator()


def test_instantiate_modo_b_simples(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    """Modo B preserva unidade do engenheiro + anexa dim p/ tipo dimensional."""
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
    token = instantiator.instantiate(region, mode="B")
    assert token.text == "[QTY value=350 unit=MPa dim=[1,-1,-2,0,0,0,0]]"


def test_instantiate_modo_b_si_canonical_em_quantity() -> None:
    """SI canônico fica programaticamente acessível em Quantity para uso
    downstream, mesmo que NÃO apareça no texto Modo B."""
    inst = GrandezaFisicaInstantiator()
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
    token = inst.instantiate(region, mode="B")
    q = token.quantity
    assert q.value == 350.0
    assert q.unit_text == "MPa"
    assert q.si_value == pytest.approx(3.5e8)
    assert q.si_unit == "Pa"
    assert q.factor == pytest.approx(1.0e6)
    assert q.dim_vector == (1, -1, -2, 0, 0, 0, 0)


def test_instantiate_modo_b_com_incerteza(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    """Modo B preserva valor + incerteza no original do engenheiro."""
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 13, "350 ± 10 MPa")
    token = instantiator.instantiate(region, mode="B")
    assert token.text == (
        "[QTY value=350 unit=MPa dim=[1,-1,-2,0,0,0,0] unc=10]"
    )


def test_instantiate_modo_b_locale_pt_br(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    """287,4 MPa (locale PT-BR canonicalizado para 287.4); unit preservada."""
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 9, "287,4 MPa")
    token = instantiator.instantiate(region, mode="B")
    assert "value=287.4" in token.text
    assert "unit=MPa" in token.text
    assert "dim=[1,-1,-2,0,0,0,0]" in token.text


def test_instantiate_modo_b_unit_compost_br(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    """350 kgf/cm² preserva unidade BR; dim assina pressão."""
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 11, "350 kgf/cm²")
    token = instantiator.instantiate(region, mode="B")
    assert token.text == (
        "[QTY value=350 unit=kgf/cm² dim=[1,-1,-2,0,0,0,0]]"
    )


def test_instantiate_modo_a_trinitario(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    # Evolução 4 (v0.7): formato canônico único entre Modo A e Modo B —
    # inteiros até 1e15 compactos (`str(int)`), sem ramo `:.6g` para
    # valores SI exatos. Princípio: nenhuma divergência de formatação
    # entre modos ou instanciadores.
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 13, "350 ± 10 MPa")
    token = instantiator.instantiate(region, mode="A")
    expected = "<QTY><VAL>350000000</VAL><DIM>Pa</DIM><UNC>10000000</UNC></QTY>"
    assert token.text == expected


def test_instantiate_modo_a_sem_incerteza(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
    token = instantiator.instantiate(region, mode="A")
    expected = "<QTY><VAL>350000000</VAL><DIM>Pa</DIM><UNC/></QTY>"
    assert token.text == expected


def test_instantiate_modo_a_unit_compost(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    # 350 kgf/cm² → 350 × 98066.5 = 34323275.0 Pa — inteiro IEEE 754.
    # Modo A consistente com Modo B: inteiros até 1e15 ficam compactos
    # como `str(int)`, sem cair em notação científica.
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 11, "350 kgf/cm²")
    token = instantiator.instantiate(region, mode="A")
    assert "<DIM>Pa</DIM>" in token.text
    assert "<VAL>34323275</VAL>" in token.text


def test_instantiate_rejeita_tipo_errado(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    region = Region(TipoNome.IDENTIFICADOR_TECNICO, 0, 10, "AISI 1045")
    with pytest.raises(TypeError, match="GrandezaFisica"):
        instantiator.instantiate(region)


def test_instantiate_text_atalho(
    instantiator: GrandezaFisicaInstantiator,
) -> None:
    """10 tf preserva unidade do engenheiro em Modo B; Modo A canoniza p/ SI."""
    text_b = instantiator.instantiate_text("10 tf")
    assert text_b == "[QTY value=10 unit=tf dim=[1,1,-2,0,0,0,0]]"
    # Modo A é tokenizer interno (vocab nosso) — canoniza para SI base
    text_a = instantiator.instantiate_text("10 tf", mode="A")
    assert text_a == "<QTY><VAL>98066.5</VAL><DIM>N</DIM><UNC/></QTY>"


# ---------------------------------------------------------------------------
# Round-trip Camada 2 → Camada 3
# ---------------------------------------------------------------------------


def test_round_trip_textos_engenharia_br() -> None:
    """Grandezas detectadas Camada 2 → instanciadas Camada 3 preservando
    o que o engenheiro escreveu; dim_vector assina o tipo dimensional;
    si_value/si_unit ficam acessíveis programaticamente em Quantity para
    uso downstream que precisar de canônico (e.g., evidências experimentais)."""
    classifier = OntologicalClassifier()
    instantiator = GrandezaFisicaInstantiator()

    texto = (
        "Carga de 50 tf gerou tensão de 287,4 MPa, "
        "com módulo E = 200 GPa e viscosidade 1.5e-3 Pa·s, "
        "momento de 250 tf·m, fluxo térmico 5 W/(m²·K)."
    )
    regions = classifier.classify(texto)
    grandezas = [r for r in regions if r.tipo is TipoNome.GRANDEZA_FISICA]
    tokens = [instantiator.instantiate(r) for r in grandezas]
    textos_b = [t.text for t in tokens]

    # Preserva unidade do engenheiro
    assert any("value=50" in t and "unit=tf" in t for t in textos_b)
    assert any("value=287.4" in t and "unit=MPa" in t for t in textos_b)
    assert any("value=200" in t and "unit=GPa" in t for t in textos_b)
    assert any("unit=tf·m" in t for t in textos_b)

    # dim_vector está sempre presente para tipo dimensional
    assert all("dim=[" in t for t in textos_b)

    # SI canônico programaticamente acessível em cada Quantity
    quantities = [t.quantity for t in tokens]
    # 50 tf → si_value=490332.5 N
    assert any(
        q.unit_text == "tf" and q.si_value == pytest.approx(490332.5)
        for q in quantities
    )
    # 287,4 MPa → si_value=2.874e+8 Pa
    assert any(
        q.unit_text == "MPa" and q.si_value == pytest.approx(2.874e8)
        for q in quantities
    )
