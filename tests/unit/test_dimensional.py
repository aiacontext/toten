"""Testes Dia 6 — tabela dimensional + álgebra.

Avalia: carga + validação da dim_table, lookup de átomos, derived_si
lookup, combine_terms para casos canônicos e long tail, render_si_unit,
verify_homogeneity, graceful degradation para átomos desconhecidos.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from toten.dimensional import (
    DEFAULT_DIM_TABLE_PATH,
    DimensionalTable,
    combine_terms,
    load_dim_table,
    render_si_unit,
    verify_homogeneity,
)
from toten.dimensional.table import default_dim_table
from toten.instantiators.quantity import UnitTerm


@pytest.fixture(scope="module")
def table() -> DimensionalTable:
    return default_dim_table()


def _term(symbol: str, power: int) -> UnitTerm:
    return UnitTerm(symbol=symbol, power=power)


# ---------------------------------------------------------------------------
# Carga + validação
# ---------------------------------------------------------------------------


def test_default_dim_table_path_existe() -> None:
    assert DEFAULT_DIM_TABLE_PATH.is_file()


def test_load_dim_table_default(table: DimensionalTable) -> None:
    assert table.version.startswith("0.")
    assert table.si_base_order == ("kg", "m", "s", "A", "K", "mol", "cd")
    assert len(table.atoms) >= 80
    assert len(table.derived_si) >= 15


def test_load_dim_table_path_inexistente(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dim_table(tmp_path / "no_table.json")


def test_dim_table_rejeita_si_base_errada(tmp_path) -> None:
    import json

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "comment": "x",
            "si_base_order": ["x", "m", "s", "A", "K", "mol", "cd"],
            "atoms": {"x": {"dim": [1,0,0,0,0,0,0], "factor": 1.0,
                            "si_canonical": "x", "category": "x"}},
            "derived_si": {},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="si_base_order"):
        load_dim_table(bad)


def test_dim_table_rejeita_derived_chave_invalida(tmp_path) -> None:
    import json

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({
            "version": "x",
            "comment": "x",
            "si_base_order": ["kg", "m", "s", "A", "K", "mol", "cd"],
            "atoms": {"m": {"dim": [0,1,0,0,0,0,0], "factor": 1.0,
                            "si_canonical": "m", "category": "length"}},
            "derived_si": {"1,2,3": "Bad"},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="7 componentes"):
        load_dim_table(bad)


# ---------------------------------------------------------------------------
# Lookup de átomos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "dim", "factor", "si"),
    [
        ("Pa", (1, -1, -2, 0, 0, 0, 0), 1.0, "Pa"),
        ("MPa", (1, -1, -2, 0, 0, 0, 0), 1.0e6, "Pa"),
        ("kgf", (1, 1, -2, 0, 0, 0, 0), 9.80665, "N"),
        ("tf", (1, 1, -2, 0, 0, 0, 0), 9806.65, "N"),
        ("m", (0, 1, 0, 0, 0, 0, 0), 1.0, "m"),
        ("cm", (0, 1, 0, 0, 0, 0, 0), 1.0e-2, "m"),
        ("K", (0, 0, 0, 0, 1, 0, 0), 1.0, "K"),
        ("°C", (0, 0, 0, 0, 1, 0, 0), 1.0, "K"),
        ("W", (1, 2, -3, 0, 0, 0, 0), 1.0, "W"),
        ("cv", (1, 2, -3, 0, 0, 0, 0), 735.49875, "W"),
    ],
)
def test_get_atom_canonico(
    table: DimensionalTable,
    symbol: str,
    dim: tuple,
    factor: float,
    si: str,
) -> None:
    atom = table.get_atom(symbol)
    assert atom is not None
    assert atom.dim == dim
    assert atom.factor == pytest.approx(factor)
    assert atom.si_canonical == si


def test_get_atom_desconhecido(table: DimensionalTable) -> None:
    assert table.get_atom("foobar") is None


# ---------------------------------------------------------------------------
# Derived_si lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dim", "esperado"),
    [
        ((1, 1, -2, 0, 0, 0, 0), "N"),
        ((1, -1, -2, 0, 0, 0, 0), "Pa"),
        ((1, 2, -2, 0, 0, 0, 0), "J"),
        ((1, 2, -3, 0, 0, 0, 0), "W"),
        ((0, 0, -1, 0, 0, 0, 0), "Hz"),
        ((1, -1, -1, 0, 0, 0, 0), "Pa·s"),
        ((0, 1, -1, 0, 0, 0, 0), "m/s"),
        ((0, 1, -2, 0, 0, 0, 0), "m/s²"),
    ],
)
def test_lookup_derived(
    table: DimensionalTable, dim: tuple, esperado: str
) -> None:
    assert table.lookup_derived(dim) == esperado


def test_lookup_derived_nao_existe(table: DimensionalTable) -> None:
    assert table.lookup_derived((9, 9, 9, 9, 9, 9, 9)) is None


# ---------------------------------------------------------------------------
# combine_terms — casos canônicos
# ---------------------------------------------------------------------------


def test_combine_terms_unidade_simples(table: DimensionalTable) -> None:
    """350 MPa = 1 termo, factor=1e6, dim Pa."""
    result = combine_terms([_term("MPa", 1)], table)
    assert result is not None
    assert result.dim == (1, -1, -2, 0, 0, 0, 0)
    assert result.factor == pytest.approx(1.0e6)
    assert result.si_unit == "Pa"


def test_combine_terms_kgf_cm2_pressao(table: DimensionalTable) -> None:
    """kgf/cm² é pressão técnica: factor=9.80665*1e4=98066.5."""
    result = combine_terms(
        [_term("kgf", 1), _term("cm", -2)], table
    )
    assert result is not None
    assert result.dim == (1, -1, -2, 0, 0, 0, 0)
    assert result.factor == pytest.approx(98066.5)
    assert result.si_unit == "Pa"


def test_combine_terms_tf_m_momento(table: DimensionalTable) -> None:
    """tf·m tem dim de J (N·m = J): factor=9806.65."""
    result = combine_terms([_term("tf", 1), _term("m", 1)], table)
    assert result is not None
    assert result.dim == (1, 2, -2, 0, 0, 0, 0)
    assert result.factor == pytest.approx(9806.65)
    assert result.si_unit == "J"


def test_combine_terms_w_por_m_k_long_tail(
    table: DimensionalTable,
) -> None:
    """W/(m·K) — átomos todos SI base, factor=1.0; dim sem nome curto."""
    result = combine_terms(
        [_term("W", 1), _term("m", -1), _term("K", -1)], table
    )
    assert result is not None
    assert result.dim == (1, 1, -3, 0, -1, 0, 0)
    assert result.factor == pytest.approx(1.0)
    # Sem entrada em derived_si — fallback para base SI composition
    assert "kg" in result.si_unit
    assert "K^-1" in result.si_unit


def test_combine_terms_tf_por_m2_c_long_tail(
    table: DimensionalTable,
) -> None:
    """tf/(m²·°C) — composição BR long tail, todos átomos conhecidos."""
    result = combine_terms(
        [_term("tf", 1), _term("m", -2), _term("°C", -1)], table
    )
    assert result is not None
    assert result.dim == (1, -1, -2, 0, -1, 0, 0)
    # factor: tf=9806.65, m^-2 factor=1, °C^-1 factor=1
    assert result.factor == pytest.approx(9806.65)


def test_combine_terms_atomo_desconhecido_devolve_none(
    table: DimensionalTable,
) -> None:
    """Se algum átomo não está na tabela, devolve None (fallback Modo B)."""
    result = combine_terms([_term("foobar", 1), _term("m", -1)], table)
    assert result is None


def test_combine_terms_terms_vazio(table: DimensionalTable) -> None:
    assert combine_terms([], table) is None


# ---------------------------------------------------------------------------
# render_si_unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dim", "esperado"),
    [
        ((1, 1, -2, 0, 0, 0, 0), "N"),
        ((1, -1, -2, 0, 0, 0, 0), "Pa"),
        ((1, 2, -2, 0, 0, 0, 0), "J"),
        ((0, 0, 0, 0, 0, 0, 0), ""),
        ((1, 0, 0, 0, 0, 0, 0), "kg"),
        ((0, 1, 0, 0, 0, 0, 0), "m"),
    ],
)
def test_render_si_unit_derivada(
    table: DimensionalTable, dim: tuple, esperado: str
) -> None:
    assert render_si_unit(dim, table) == esperado


def test_render_si_unit_composicao_sem_nome_curto(
    table: DimensionalTable,
) -> None:
    """Dim sem derived_si entry vira composição base SI."""
    # condutividade térmica = W/(m·K) = kg·m·s⁻³·K⁻¹
    dim = (1, 1, -3, 0, -1, 0, 0)
    assert render_si_unit(dim, table) == "kg·m·s^-3·K^-1"


# ---------------------------------------------------------------------------
# verify_homogeneity
# ---------------------------------------------------------------------------


def test_homogeneity_iguais() -> None:
    assert verify_homogeneity(
        (1, -1, -2, 0, 0, 0, 0), (1, -1, -2, 0, 0, 0, 0)
    )


def test_homogeneity_diferentes() -> None:
    assert not verify_homogeneity(
        (1, -1, -2, 0, 0, 0, 0), (1, 1, -2, 0, 0, 0, 0)
    )


# ---------------------------------------------------------------------------
# Cenários integradores
# ---------------------------------------------------------------------------


def test_combinacao_real_kgf_cm2_350(table: DimensionalTable) -> None:
    """350 kgf/cm² = 350 * 98066.5 Pa = 3.43e+7 Pa."""
    result = combine_terms([_term("kgf", 1), _term("cm", -2)], table)
    assert result is not None
    si_value = 350 * result.factor
    assert si_value == pytest.approx(3.43233e7, rel=1e-4)


def test_combinacao_real_tf_m_250(table: DimensionalTable) -> None:
    """250 tf·m = 250 * 9806.65 J = 2.45e+6 J."""
    result = combine_terms([_term("tf", 1), _term("m", 1)], table)
    assert result is not None
    si_value = 250 * result.factor
    assert si_value == pytest.approx(2.45166e6, rel=1e-4)
