"""Álgebra dimensional sobre vetores ℤ⁷.

Operações fundamentais (spec §3.5):
- `combine_terms(terms, table)`: dado lista de UnitTerm(symbol, power),
  devolve (dim_vector, total_factor) somando os dims ponderados por
  power e multiplicando os fatores elevados à power.
- `render_si_unit(dim, table)`: render simbólico do dim. Prefere
  derived_si (Pa, N, J, ...); fallback é composição de base SI
  (`kg·m^-1·s^-2`).
- `verify_homogeneity(dim_a, dim_b)`: igualdade exata.

Quando algum átomo não está na tabela, `combine_terms` devolve None
sinalizando graceful degradation — o instanciador então cai no
fallback (Modo B verbatim sem dim).
"""

from __future__ import annotations

from dataclasses import dataclass

from toten.dimensional.table import DimensionalTable

DimVector = tuple[int, int, int, int, int, int, int]
_ZERO_DIM: DimVector = (0, 0, 0, 0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class DimensionalAnalysis:
    """Resultado da combinação dimensional de uma lista de termos."""

    dim: DimVector
    factor: float
    si_unit: str


def combine_terms(
    terms,
    table: DimensionalTable,
) -> DimensionalAnalysis | None:
    """Combina UnitTerms em (dim_vector, factor, si_unit).

    Devolve `None` se qualquer átomo da composição não estiver na
    tabela — sinaliza que o instanciador deve cair no fallback.
    """
    if not terms:
        return None

    dim = list(_ZERO_DIM)
    factor = 1.0

    for term in terms:
        atom = table.get_atom(term.symbol)
        if atom is None:
            return None
        for i in range(7):
            dim[i] += atom.dim[i] * term.power
        factor *= atom.factor ** term.power

    dim_tuple: DimVector = tuple(dim)  # type: ignore[assignment]
    si_unit = render_si_unit(dim_tuple, table)
    return DimensionalAnalysis(dim=dim_tuple, factor=factor, si_unit=si_unit)


def render_si_unit(dim: DimVector, table: DimensionalTable) -> str:
    """Render simbólico do dim_vector.

    Estratégia:
    1. Lookup em `derived_si` — devolve nome curto (Pa, N, J, ...).
    2. Fallback: composição de base SI com powers explícitos
       (`kg·m^-1·s^-2`).
    3. Adimensional → string vazia.
    """
    derived = table.lookup_derived(dim)
    if derived is not None:
        return derived
    parts: list[str] = []
    for i, power in enumerate(dim):
        if power == 0:
            continue
        base = table.si_base_order[i]
        parts.append(base if power == 1 else f"{base}^{power}")
    return "·".join(parts)


def verify_homogeneity(dim_a: DimVector, dim_b: DimVector) -> bool:
    """Verifica igualdade dimensional estrita (necessária para soma)."""
    return tuple(dim_a) == tuple(dim_b)


@dataclass(frozen=True, slots=True)
class UnitTerm:
    """Termo de unidade composta: símbolo + expoente. Ex.: UnitTerm('kN', 1)."""

    symbol: str
    power: int = 1


def _coerce_terms(unit):
    """Aceita str ('MPa' ou 'kgf/cm²'), UnitTerm, ou lista de UnitTerm.

    Strings compostas (`'kgf/cm²'`, `'m/s²'`, `'mPa·s'`) são parseadas
    via `parse_unit_composition` (instantiators.quantity). Import é
    LAZY para evitar ciclo dimensional ↔ instanciador.
    """
    if isinstance(unit, str):
        from toten.instantiators.quantity import parse_unit_composition
        return list(parse_unit_composition(unit))
    if isinstance(unit, UnitTerm):
        return [unit]
    return list(unit)


def convert(
    value: float,
    from_unit,
    to_unit,
    table: DimensionalTable,
) -> float:
    """Converte `value` de `from_unit` para `to_unit`.

    Usa álgebra ℤ⁷: dim(from) deve igualar dim(to). Caso contrário, raise.
    Factor de conversão é razão dos fatores SI das duas composições.

    Aceita unidades como string atômica (`'MPa'`), `UnitTerm`, ou lista
    de UnitTerm para composições (`[UnitTerm('kN',1), UnitTerm('m',-2)]`).

    Exemplos:
        convert(350, 'MPa', 'kgf/cm²', table)  → 3569.93
        convert(1, 'kWh', 'MJ', table)         → 3.6
        convert(100, 'km/h', 'm/s', table)     → 27.78
    """
    a = combine_terms(_coerce_terms(from_unit), table)
    b = combine_terms(_coerce_terms(to_unit), table)
    if a is None or b is None:
        msg = f"Unidade desconhecida: from={from_unit}, to={to_unit}"
        raise ValueError(msg)
    if not verify_homogeneity(a.dim, b.dim):
        msg = (
            f"Dimensões incompatíveis: from {from_unit} (dim={a.dim}) "
            f"vs to {to_unit} (dim={b.dim})"
        )
        raise ValueError(msg)
    return value * a.factor / b.factor
