"""ReferenciaInstantiator — Camada 3 para o 9º tipo da OEE (v1.2).

Referencia é entidade primária do mundo estrutural-normativo:
referência hierárquica a parágrafos, artigos, incisos, seções de
leis/normas/regulamentos. Tipado em OEE v1.2 com 1 propriedade
intrínseca:

    - hierarquia: string preservando a forma original como escrita
      (`8.2.1`, `14.2`, `5.3.1.2`)

Modo B emite tag atômica:
    [REF:8.2.1]

Decisão de design (preservação literal):
- A `hierarquia` preserva exatamente como o engenheiro escreveu —
  inclui pontuação interna, dígitos, e zeros à esquerda se presentes.
- Sem `value=` (não é número escalar interpretável)
- Sem `dim=` (não é grandeza física)
- Sem `category=` (referência é categoria semântica única já)

Análogo a `[IDX:slug]` mas semanticamente distinto: IDX identifica
materiais/normas como objetos primários (AISI 1045, NBR 6118);
REF identifica SUBDIVISÃO ESTRUTURAL dentro de uma norma/lei.

Reconhecido pelo classifier APENAS quando precedido por enumerador
normativo (parágrafo, §, art., inciso, alínea, lei, decreto, norma,
regulamento, etc.). Sem antecedente normativo, o mesmo padrão
`\\d+(\\.\\d+)+` poderia ser número decimal ou versão de software —
contexto determina interpretação inequívoca.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from toten.classifier.region import Region
from toten.ontology.types import TipoNome


@dataclass(frozen=True, slots=True)
class Referencia:
    """Referencia decomposta em propriedade intrínseca da OEE v1.2."""

    hierarquia: str

    def __post_init__(self) -> None:
        if not self.hierarquia.strip():
            msg = "Referencia exige hierarquia não-vazia"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ReferenciaToken:
    """Representação tokenizada de uma Referencia (Modo A ou B)."""

    referencia: Referencia
    mode: Literal["A", "B"]

    @property
    def text(self) -> str:
        if self.mode == "B":
            return self._render_b()
        return self._render_a()

    def _render_b(self) -> str:
        """Modo B — tag atômica `[REF:<hierarquia>]`."""
        return f"[REF:{self.referencia.hierarquia}]"

    def _render_a(self) -> str:
        """Modo A — formato compacto consistente com Modo B."""
        return f"<REF>{self.referencia.hierarquia}</REF>"


def parse_referencia(content: str) -> Referencia:
    """Decompõe `content` em `Referencia`.

    Aceita string já reconhecida pelo classifier como referência
    hierárquica (e.g., "8.2.1", "14.2", "5.3.1.2"). Preserva
    integralmente como `hierarquia`.
    """
    if not content or not content.strip():
        msg = "parse_referencia exige string não-vazia"
        raise ValueError(msg)
    return Referencia(hierarquia=content.strip())


class ReferenciaInstantiator:
    """Camada 3 — instancia regiões do tipo Referencia (OEE v1.2)."""

    def instantiate(
        self, region: Region, mode: Literal["A", "B"] = "B"
    ) -> ReferenciaToken:
        if region.tipo is not TipoNome.REFERENCIA:
            msg = (
                "ReferenciaInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava Referencia"
            )
            raise TypeError(msg)
        referencia = parse_referencia(region.content)
        return ReferenciaToken(referencia=referencia, mode=mode)

    def instantiate_text(
        self, content: str, mode: Literal["A", "B"] = "B"
    ) -> str:
        referencia = parse_referencia(content)
        return ReferenciaToken(referencia=referencia, mode=mode).text
