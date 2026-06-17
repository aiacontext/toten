"""Saída canônica da Camada 2: regiões tipadas em texto bruto."""

from __future__ import annotations

from dataclasses import dataclass

from toten.ontology.types import TipoNome


@dataclass(frozen=True, slots=True)
class Region:
    """Trecho de texto classificado em um tipo da OEE.

    `start` é inclusivo, `end` é exclusivo — convenção idêntica a
    `re.Match.span()`. `content` é o texto literal `text[start:end]`,
    mantido para auditoria.

    `canonical_slug` (opcional) é override declarado pelo lexicon quando
    uma classe normativa tem canônico fixo (ex.: material `CP 190 RB`
    com variantes notacionais `CP190 RB`, `CP-190-RB`). Análogo a Pint:
    o lexicon É o oráculo do canônico de cada classe.

    `ambig_kind` (opcional) é o nome canônico da dimensão de
    ambiguidade detectada estruturalmente pelo classifier (ver
    SPEC_07 §2). Quando presente, o instanciador propaga ao Modo B
    como `ambig=<tipo>` + `alternatives=<a1>|<a2>|...`. Os candidatos
    estruturalmente possíveis ficam em `ambig_alternatives` (lista
    ordenada por probabilidade).
    """

    tipo: TipoNome
    start: int
    end: int
    content: str
    canonical_slug: str | None = None
    ambig_kind: str | None = None
    ambig_alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0:
            msg = f"start negativo: {self.start}"
            raise ValueError(msg)
        if self.end <= self.start:
            msg = f"end ({self.end}) deve ser estritamente maior que start ({self.start})"
            raise ValueError(msg)

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end

    @property
    def length(self) -> int:
        return self.end - self.start
