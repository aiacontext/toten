"""Instanciador de ConstanteUniversal.

ConstanteUniversal é composta com GrandezaFisica conforme a OEE §2.2:
cada constante é um símbolo + grandeza_associada (valor + unidade SI).
Esta composição é refletida no instanciador — ele emite tag com o
símbolo canônico E os componentes da grandeza associada.

Modo B (alto valor): `[CONST:π value=3.141592653589793]` para
adimensionais, `[CONST:k_B value=1.380649e-23 unit=J/K]` para
constantes com dimensão. O LLM consumidor enxerga símbolo + valor
ancorado, eliminando alucinação de magnitude para constantes
universais (e.g., confundir `k_B = 1.380649e-23 J/K` com
`k_B = 1.380649e+23 J/K`).

Modo A: token atômico do símbolo + composição Quantity associada
(materialização completa quando vocab existir).

Constantes desconhecidas (fora do lexicon) caem em fallback que
emite `[CONST:simbolo]` sem value/unit — preserva atomicidade
ontológica mesmo sem grounding numérico.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from toten.classifier.region import Region
from toten.ontology.types import TipoNome

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONSTANT_LEXICON_PATH = PACKAGE_ROOT / "data" / "constant_lexicon_v0.json"


@dataclass(frozen=True, slots=True)
class ConstantToken:
    """Constante universal canonizada."""

    symbol: str
    value: float | None
    unit: str | None
    mode: Literal["A", "B"]
    nome: str | None = None

    @property
    def text(self) -> str:
        if self.mode == "B":
            return self._render_b()
        return self._render_a()

    def _render_b(self) -> str:
        """Modo B atômico: `[CONST:simbolo]` sem value/unit.

        Decisão de design: constantes universais consagradas (π, ℏ, k_B,
        N_A, σ_SB, R_g) têm valor canônico que LLMs de fronteira já conhecem.
        Inserir `value=N unit=U` é redundante e — pior — induz erro físico
        quando o símbolo é homônimo de variável de engenharia (`c` como
        velocidade da luz vs amortecimento). Mantém apenas o slug do
        símbolo como identificador atômico.
        """
        return f"[CONST:{self.symbol}]"

    def _render_a(self) -> str:
        # Modo A trinitário: símbolo + grandeza associada composta
        if self.value is None:
            return f"<CONST><SYM>{self.symbol}</SYM></CONST>"
        unit_part = self.unit if self.unit is not None else ""
        return (
            f"<CONST><SYM>{self.symbol}</SYM>"
            f"<QTY><VAL>{_format_value(self.value)}</VAL>"
            f"<DIM>{unit_part}</DIM><UNC/></QTY></CONST>"
        )


def _format_value(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


class ConstanteUniversalInstantiator:
    """Camada 3 — ConstanteUniversal com composição GrandezaFisica."""

    def __init__(self, lexicon_path: Path | None = None) -> None:
        path = lexicon_path or DEFAULT_CONSTANT_LEXICON_PATH
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self._by_symbol: dict[str, dict] = {
            entry["simbolo"]: entry
            for entry in payload.get("constantes", [])
            if "simbolo" in entry
        }

    def instantiate(
        self, region: Region, mode: Literal["A", "B"] = "B"
    ) -> ConstantToken:
        if region.tipo is not TipoNome.CONSTANTE_UNIVERSAL:
            msg = (
                "ConstanteUniversalInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava ConstanteUniversal"
            )
            raise TypeError(msg)
        return self._make_token(region.content, mode)

    def instantiate_text(self, content: str, mode: Literal["A", "B"] = "B") -> str:
        return self._make_token(content, mode).text

    def _make_token(self, symbol: str, mode: str) -> ConstantToken:
        entry = self._by_symbol.get(symbol)
        if entry is None:
            return ConstantToken(
                symbol=symbol, value=None, unit=None, mode=mode  # type: ignore[arg-type]
            )
        grandeza = entry.get("grandeza_associada")
        # Convenção do lexicon: "adimensional" não emite unit em Modo B
        unit = None if grandeza == "adimensional" else grandeza
        return ConstantToken(
            symbol=symbol,
            value=entry.get("valor_si"),
            unit=unit,
            mode=mode,  # type: ignore[arg-type]
            nome=entry.get("nome"),
        )
