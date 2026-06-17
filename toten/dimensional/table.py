"""Carregamento e validação da tabela dimensional.

`data/dim_table.json` é a única fonte de verdade dimensional do
framework. Carregada na inicialização, exposta via API tipada com
pydantic.

PREFIXOS SI: bases canônicas marcadas com `prefixable: true` são
expandidas algoritmicamente no load com a tabela `SI_PREFIXES` (Y, Z,
E, P, T, G, M, k, h, da, d, c, m, µ/μ, n, p, f, a, z, y). Entradas
explícitas no JSON têm precedência sobre as geradas.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIM_TABLE_PATH = PACKAGE_ROOT / "data" / "dim_table.json"
DEFAULT_DIMENSIONLESS_PATH = PACKAGE_ROOT / "data" / "dimensionless_units.json"
DEFAULT_INFO_UNITS_PATH = PACKAGE_ROOT / "data" / "info_units.json"


# Prefixos SI oficiais (BIPM). Inclui µ e μ como aliases (U+00B5 e U+03BC).
SI_PREFIXES: dict[str, float] = {
    "Y":  1.0e24,
    "Z":  1.0e21,
    "E":  1.0e18,
    "P":  1.0e15,
    "T":  1.0e12,
    "G":  1.0e9,
    "M":  1.0e6,
    "k":  1.0e3,
    "h":  1.0e2,
    "da": 1.0e1,
    "d":  1.0e-1,
    "c":  1.0e-2,
    "m":  1.0e-3,
    "µ":  1.0e-6,
    "μ":  1.0e-6,
    "n":  1.0e-9,
    "p":  1.0e-12,
    "f":  1.0e-15,
    "a":  1.0e-18,
    "z":  1.0e-21,
    "y":  1.0e-24,
}


class AtomEntry(BaseModel):
    """Entrada atômica da tabela dimensional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dim: tuple[int, int, int, int, int, int, int]
    factor: float = Field(gt=0)
    si_canonical: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prefixable: bool = False


class DimensionalTable(BaseModel):
    """Tabela dimensional completa (átomos + derived_si)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    comment: str
    si_base_order: tuple[str, ...] = Field(min_length=7, max_length=7)
    atoms: dict[str, AtomEntry] = Field(min_length=1)
    derived_si: dict[str, str]

    @model_validator(mode="after")
    def _si_base_canonica(self) -> DimensionalTable:
        expected = ("kg", "m", "s", "A", "K", "mol", "cd")
        if self.si_base_order != expected:
            msg = (
                f"si_base_order deve ser {list(expected)}, "
                f"recebido {list(self.si_base_order)}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _derived_si_chaves_validas(self) -> DimensionalTable:
        for key in self.derived_si:
            parts = key.split(",")
            if len(parts) != 7:
                msg = (
                    f"chave derived_si '{key}' deve ter 7 componentes; "
                    f"tem {len(parts)}"
                )
                raise ValueError(msg)
            for p in parts:
                int(p)  # raises se não inteiro
        return self

    def get_atom(self, symbol: str) -> AtomEntry | None:
        return self.atoms.get(symbol)

    def lookup_derived(
        self, dim: tuple[int, int, int, int, int, int, int]
    ) -> str | None:
        """Devolve o nome SI canônico para um dim_vector, se conhecido."""
        key = ",".join(str(d) for d in dim)
        return self.derived_si.get(key)


def _expand_si_prefixes(
    explicit_atoms: dict[str, dict],
) -> dict[str, dict]:
    """Para cada átomo com `prefixable: true`, gera todas formas prefixadas.

    Entradas explícitas no JSON têm PRECEDÊNCIA — só preenche o que falta.
    Isso evita conflitos com bases SI já presentes (`kg` é base com factor=1,
    não sobrescrever pela versão gerada `k+g`).

    Retorna dict atoms expandido (mantém referencias dos explícitos).
    """
    expanded = dict(explicit_atoms)
    for base_symbol, base_entry in explicit_atoms.items():
        if not base_entry.get("prefixable", False):
            continue
        base_factor = base_entry["factor"]
        for prefix, prefix_factor in SI_PREFIXES.items():
            prefixed_symbol = f"{prefix}{base_symbol}"
            if prefixed_symbol in expanded:
                continue  # precedência: explícito vence
            expanded[prefixed_symbol] = {
                "dim": base_entry["dim"],
                "factor": base_factor * prefix_factor,
                "si_canonical": base_entry["si_canonical"],
                "category": base_entry["category"],
                "prefixable": False,
            }
    return expanded


# Canônico SI por categoria semântica para átomos auxiliares
# (dimensionless / info). Permite que entradas dos arquivos
# `dimensionless_units.json` e `info_units.json` (sem vetor SI próprio)
# se adaptem ao schema `AtomEntry` que exige `si_canonical`.
_AUX_SI_CANONICAL_BY_CATEGORY: dict[str, str] = {
    # dimensionless
    "ratio": "1",
    "logarithmic_ratio": "1",
    "angle": "rad",
    "solid_angle": "sr",
    # info
    "information": "bit",
    "visualization": "pixel",
    "visual_density": "pixel/m",
    "color_depth": "bit/pixel",
    "info_other": "1",
}


def _load_auxiliary_atoms(
    path: Path, default_si_canonical: str = "1"
) -> dict[str, dict]:
    """Carrega átomos de arquivos supplementares (dimensionless ou info).

    Adapta cada entrada ao schema `AtomEntry`:
    - dim = (0, 0, 0, 0, 0, 0, 0) — não-SI por construção
    - factor = preservado
    - si_canonical = derivado da categoria via _AUX_SI_CANONICAL_BY_CATEGORY
    - category = preservada do JSON

    Arquivo ausente → dict vazio (graceful degradation).
    """
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    atoms: dict[str, dict] = {}
    for name, entry in raw.get("atoms", {}).items():
        category = entry.get("category", "other")
        si_canonical = _AUX_SI_CANONICAL_BY_CATEGORY.get(
            category, default_si_canonical
        )
        atoms[name] = {
            "dim": [0, 0, 0, 0, 0, 0, 0],
            "factor": entry["factor"],
            "si_canonical": si_canonical,
            "category": category,
        }
    return atoms


def load_dim_table(
    path: Path | str | None = None,
    *,
    include_auxiliary: bool = True,
) -> DimensionalTable:
    """Carrega, expande prefixos SI e valida a tabela dimensional completa.

    Mescla três fontes (responsabilidade única por domínio ontológico):
    - `dim_table.json` — unidades SI dimensionais (ℤ⁷)
    - `dimensionless_units.json` — razões/ângulos/log (dim=0)
    - `info_units.json` — informação/visualização (dim=0)

    Em caso de conflito de chave entre arquivos, dim_table.json vence
    (autoridade primária). `include_auxiliary=False` carrega apenas SI
    puro (útil para testes/debug).
    """
    if path is None:
        path = DEFAULT_DIM_TABLE_PATH
    json_path = Path(path)
    if not json_path.is_file():
        msg = f"dim_table.json não encontrado em {json_path}"
        raise FileNotFoundError(msg)
    with json_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    si_atoms = raw["atoms"]
    if include_auxiliary:
        aux_dimensionless = _load_auxiliary_atoms(DEFAULT_DIMENSIONLESS_PATH)
        aux_info = _load_auxiliary_atoms(DEFAULT_INFO_UNITS_PATH)
        # Merge: SI tem precedência (não sobrescreve)
        merged = {}
        merged.update(aux_dimensionless)
        merged.update(aux_info)
        merged.update(si_atoms)  # SI vence
        raw["atoms"] = merged
    else:
        raw["atoms"] = si_atoms

    raw["atoms"] = _expand_si_prefixes(raw["atoms"])
    return DimensionalTable.model_validate(raw)


@lru_cache(maxsize=1)
def default_dim_table() -> DimensionalTable:
    """Tabela default cacheada (uma instância para o pacote inteiro)."""
    return load_dim_table()
