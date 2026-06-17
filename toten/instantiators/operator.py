"""Instanciador de OperadorFormal.

Operadores formais (=, ≤, ∑, ∫, ∂, →, ⇒, ...) já são universalmente
reconhecidos por LLMs frozen — `≤` é menor-ou-igual em qualquer LLM
recente. Tagging em Modo B agregaria ruído sem ganho representacional.

Modo B: passthrough. Mantém o símbolo intacto na saída canônica.

Modo A (futuro): vocabulário fechado pequeno (~30 tokens), cada operador
um ID atômico do vocab — invariante "atômico, não-fragmentável" da
OEE §2.2 honrado por construção.

Metadados (aridade, categoria) carregados do lexicon ficam disponíveis
no Token retornado para uso opcional pela Camada 4 (encoder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from toten.classifier.region import Region
from toten.ontology.types import TipoNome

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OPERATOR_LEXICON_PATH = PACKAGE_ROOT / "data" / "operator_lexicon_v0.json"


@dataclass(frozen=True, slots=True)
class OperatorToken:
    """Operador formal com metadata ontológica."""

    symbol: str
    mode: Literal["A", "B"]
    aridade: int | None = None
    categoria: str | None = None

    @property
    def text(self) -> str:
        return self.symbol


class OperadorFormalInstantiator:
    """Camada 3 — OperadorFormal. Modo B é passthrough."""

    def __init__(self, lexicon_path: Path | None = None) -> None:
        path = lexicon_path or DEFAULT_OPERATOR_LEXICON_PATH
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self._by_symbol: dict[str, dict] = {
            entry["simbolo"]: entry
            for entry in payload.get("operadores", [])
            if "simbolo" in entry
        }

    def instantiate(
        self, region: Region, mode: Literal["A", "B"] = "B"
    ) -> OperatorToken:
        if region.tipo is not TipoNome.OPERADOR_FORMAL:
            msg = (
                "OperadorFormalInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava OperadorFormal"
            )
            raise TypeError(msg)
        return self._make_token(region.content, mode)

    def instantiate_text(self, content: str, mode: Literal["A", "B"] = "B") -> str:
        return self._make_token(content, mode).text

    def _make_token(self, symbol: str, mode: str) -> OperatorToken:
        entry = self._by_symbol.get(symbol)
        return OperatorToken(
            symbol=symbol,
            mode=mode,  # type: ignore[arg-type]
            aridade=entry.get("aridade") if entry else None,
            categoria=entry.get("categoria") if entry else None,
        )
