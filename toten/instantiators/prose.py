"""Instanciador de ProsaTecnica.

Princípio operacional:

A prosa técnica é a *cola* entre as entidades tipadas. Em Modo B (texto
canônico para LLMs frozen), o framework NÃO transforma a prosa — emite
verbatim. O consumidor LLM já tem seu próprio tokenizer para prosa
natural. A *contribuição* do framework está nos tipos onde LLMs frozen
erram (grandezas, identificadores, constantes); a prosa ao redor é
deixada intacta.

Em Modo A (token IDs para um SLM treinado nativamente), a prosa precisa
de BPE para virar IDs do vocab. Esta classe define a interface
`bpe_backend` para esse caso, mas a materialização do vocab é decisão
do momento de treino do SLM — não desta fase. O default `bpe_backend`
levanta `NotImplementedError` documentando o ponto de decisão.

Quando chegarmos a treinar o SLM (Enedina v3 / exp004 em
`enedina_agentic_v0.1`), as opções são:
- Reaproveitar `enedina_agentic_v0.1/tokenizer/output/enedina_spm.model`
  (SentencePiece BPE 16k, byte fallback, mas com 25% do vocab queimado
  em multi-dígitos — pouco eficiente já que números aqui passam pelo
  `GrandezaFisicaInstantiator`, não pelo BPE).
- Treinar BPE fresh dedicado a prosa (vocab limpo, sem desperdício de
  slots em dígitos ou markup de entities).

Recomendação atual: opção 2 (fresh BPE), decisão a confirmar no momento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from toten.classifier.region import Region
from toten.ontology.types import TipoNome


class BPEBackend(Protocol):
    """Interface para um tokenizer BPE plugável (Modo A)."""

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


class _UnboundBPEBackend:
    """Default backend que documenta a decisão deferida."""

    def encode(self, text: str) -> list[int]:
        msg = (
            "ProsaTecnicaInstantiator.Modo A exige um BPEBackend; "
            "vocab será materializado no treino do SLM (Enedina v3). "
            "Use Modo B para LLMs frozen, ou injete um backend custom."
        )
        raise NotImplementedError(msg)

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("decode requer BPEBackend materializado")


@dataclass(frozen=True, slots=True)
class ProseToken:
    """Token de prosa: em Modo B é texto verbatim; em Modo A são IDs."""

    text: str
    mode: Literal["A", "B"]
    ids: tuple[int, ...] | None = None


class ProsaTecnicaInstantiator:
    """Camada 3 — ProsaTecnica.

    Modo B: passthrough (a tese deste tipo segundo a OEE §2.3 é que
    prosa carrega significado por coocorrência estatística; a
    coocorrência é capturada pelo próprio LLM consumidor).

    Modo A: delega a `BPEBackend` injetável.
    """

    def __init__(self, bpe_backend: BPEBackend | None = None) -> None:
        self._bpe = bpe_backend if bpe_backend is not None else _UnboundBPEBackend()

    def instantiate(
        self, region: Region, mode: Literal["A", "B"] = "B"
    ) -> ProseToken:
        if region.tipo is not TipoNome.PROSA_TECNICA:
            msg = (
                "ProsaTecnicaInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava ProsaTecnica"
            )
            raise TypeError(msg)
        return self._make_token(region.content, mode)

    def instantiate_text(self, content: str, mode: Literal["A", "B"] = "B") -> str:
        return self._make_token(content, mode).text

    def _make_token(self, content: str, mode: str) -> ProseToken:
        if mode == "B":
            return ProseToken(text=content, mode="B")
        ids = tuple(self._bpe.encode(content))
        return ProseToken(text=content, mode="A", ids=ids)
