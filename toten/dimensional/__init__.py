"""Módulo dimensional — vetores ℤ⁷ SI + álgebra dimensional.

Spec §3.5. Cada átomo de unidade tem dim vector e factor de conversão
ao SI base. Composições de unidades (kgf/cm², W/(m·K), tf·m, ...) são
computadas via `algebra.combine_terms()`, NÃO enumeradas.

Esta camada habilita:
- Normalização canônica para SI no Modo B (350 MPa → 3.5e+8 Pa).
- Verificação de homogeneidade dimensional (`verify_homogeneity`).
- Render simbólico do SI base (Pa, N, J, W para os dims comuns;
  composição `kg·m^-1·s^-2` para combinações sem nome curto).
"""

from toten.dimensional.algebra import (
    DimensionalAnalysis,
    UnitTerm,
    combine_terms,
    convert,
    render_si_unit,
    verify_homogeneity,
)
from toten.dimensional.table import (
    DEFAULT_DIM_TABLE_PATH,
    SI_PREFIXES,
    AtomEntry,
    DimensionalTable,
    default_dim_table,
    load_dim_table,
)

__all__ = [
    "DEFAULT_DIM_TABLE_PATH",
    "SI_PREFIXES",
    "AtomEntry",
    "DimensionalTable",
    "default_dim_table",
    "load_dim_table",
    "DimensionalAnalysis",
    "UnitTerm",
    "combine_terms",
    "convert",
    "render_si_unit",
    "verify_homogeneity",
]
