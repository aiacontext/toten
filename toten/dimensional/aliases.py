"""Aliases de strings de unidade — pré-normalização antes da composição.

Resolve formas históricas brasileiras (`mt` → `tf·m`) e variantes ASCII
(`kgf/cm2` → `kgf/cm²`) que não cabem como átomos do dim_table.json
(que preserva atomicidade SI estrita).

Aplicado por `parse_unit_composition` em `instantiators/quantity.py` antes
do parser de composição decomposor a string em UnitTerms.

Notação ambígua entre SI (massa) e técnico (força) — `t/m`, `t·m` — é
deliberadamente NÃO mapeada. Ver `ambiguous_unmapped` no JSON.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ALIASES_PATH = PACKAGE_ROOT / "data" / "unit_aliases_v0.json"


class UnitAliases(BaseModel):
    """Mapeamento de strings de unidade para suas formas canônicas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    comment: str
    string_aliases: dict[str, str] = Field(default_factory=dict)

    def normalize(self, unit_text: str) -> str:
        """Substitui aliases conhecidos por suas formas canônicas.

        Aplica substituição EXATA da string completa. Não faz substituição
        parcial — `kgf·m/cm2` não vira `kgf·m/cm²` automaticamente, porque
        a chave `cm2` só vale se for a string INTEIRA.

        Casos não mapeados retornam intactos (graceful passthrough).
        """
        return self.string_aliases.get(unit_text, unit_text)


def load_unit_aliases(path: Path | str | None = None) -> UnitAliases:
    """Carrega e valida o arquivo de aliases."""
    if path is None:
        path = DEFAULT_ALIASES_PATH
    p = Path(path)
    if not p.is_file():
        msg = f"unit_aliases_v0.json não encontrado em {p}"
        raise FileNotFoundError(msg)
    with p.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return UnitAliases.model_validate(raw)


@lru_cache(maxsize=1)
def default_unit_aliases() -> UnitAliases:
    """Instância default cacheada para o pacote inteiro."""
    return load_unit_aliases()
