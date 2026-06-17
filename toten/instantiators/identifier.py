"""Instanciador de IdentificadorTecnico.

Converte uma região classificada como `IdentificadorTecnico` em forma
canônica que preserva a identidade do referente — invariante do tipo
declarado em `oee-v1.yaml` ("identidade preservada — fragmentar
destrói referência").

Forma canônica: `IDX:<slug>` em Modo B, onde `<slug>` é derivado
determinísticamente da string:

1. Normalização NFKD (decomposição canônica de compatibilidade).
2. Remoção de marcas combinantes (acentos: `Niobrás` → `Niobras`).
3. Lower-case.
4. Substituição de sequências não-alfanuméricas por `-`.
5. Strip de `-` nas pontas.

Letras gregas, cirílicas e demais alfabetos Unicode são preservados
(via `\\p{L}` no regex). Resultado: `IDX:niobras-320`, `IDX:σ-y`,
`IDX:abnt-nbr-12655`, `IDX:hardox-500`.

Esta é a virada conceitual do framework para identificadores: mesmo
sem dicionário global, qualquer identificador detectado pela Camada
2 sai daqui como token tipado e atômico. O LLM consumidor vê
`[IDX:hardox-500]` como uma única entidade, não como sequência de
caracteres fragmentada. Self-contained, rico, robusto.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

import regex as re

from toten.classifier.region import Region
from toten.ontology.types import TipoNome

_SLUG_SEPARATOR_RE = re.compile(r"[^\p{L}\p{N}]+")


def canonicalize_identifier(content: str, canonical_slug: str | None = None) -> str:
    """Devolve o slug canônico de um identificador.

    Determinístico: mesma entrada produz mesmo slug. Idempotente:
    `canonicalize_identifier(canonicalize_identifier(x)) == canonicalize_identifier(x)`.

    **Princípio ontológico:** o underscore ASCII (`_`) é marcador VISUAL de
    subscript em notação técnica (NBR usa `f_{ck}` formal vs `fck` inline) —
    NÃO é separador semântico. Removido antes da slugificação para colapsar
    variantes notacionais ao MESMO slug canônico:
    - `f_ptk` ↔ `fptk` → ambos viram `fptk`
    - `σ_y` ↔ `σy` → ambos viram `σy`

    **Override por classe normativa:** quando o lexicon declara `canonical_slug`
    explícito para a entry (ex.: material `CP 190 RB` com variantes
    `CP190 RB`, `CP-190-RB`), esse slug é usado diretamente — o lexicon É o
    oráculo do canônico, análogo a Pint para unidades.

    Args:
        content: string capturada.
        canonical_slug: override do lexicon (entry-level) — vence sobre derivação.

    Examples
    --------
    >>> canonicalize_identifier("Hardox 500")
    'hardox-500'
    >>> canonicalize_identifier("ABNT NBR 12655")
    'abnt-nbr-12655'
    >>> canonicalize_identifier("σ_y")
    'σy'
    >>> canonicalize_identifier("Ti-6Al-4V")
    'ti-6al-4v'
    >>> canonicalize_identifier("f_ptk")  # underscore ASCII removido
    'fptk'
    >>> canonicalize_identifier("CP190 RB", canonical_slug="cp-190-rb")
    'cp-190-rb'
    """
    if canonical_slug:
        return canonical_slug
    if not content:
        msg = "canonicalize_identifier exige string não-vazia"
        raise ValueError(msg)
    decomposed = unicodedata.normalize("NFKD", content)
    no_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Underscore ASCII: marcador VISUAL de subscript em notação técnica.
    # Removido para colapsar f_ptk/fptk, σ_y/σy, etc. ao mesmo canônico.
    no_underscore = no_marks.replace("_", "")
    lowered = no_underscore.lower()
    slug = _SLUG_SEPARATOR_RE.sub("-", lowered).strip("-")
    if not slug:
        msg = f"canonicalize_identifier produziu slug vazio para '{content}'"
        raise ValueError(msg)
    return slug


@dataclass(frozen=True, slots=True)
class IdentifierToken:
    """Representação tokenizada de um IdentificadorTecnico."""

    raw: str
    slug: str
    mode: Literal["A", "B"]

    @property
    def text(self) -> str:
        """Forma textual canônica conforme o modo."""
        if self.mode == "B":
            return f"[IDX:{self.slug}]"
        return f"<ID>{self.slug}</ID>"


class IdentificadorTecnicoInstantiator:
    """Camada 3 — instancia regiões de IdentificadorTecnico.

    A interface segue o Protocol declarado na spec §3.4: dado o
    conteúdo de uma região tipada, produz a representação canônica
    do tipo. Aqui o modo B (texto canônico) é o caso central; o
    modo A (token IDs do vocab) é definido como estrutura mas a
    materialização em IDs depende do vocab construído nos Dias
    posteriores.
    """

    def instantiate(self, region: Region, mode: Literal["A", "B"] = "B") -> IdentifierToken:
        if region.tipo is not TipoNome.IDENTIFICADOR_TECNICO:
            msg = (
                "IdentificadorTecnicoInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava IdentificadorTecnico"
            )
            raise TypeError(msg)
        slug = canonicalize_identifier(
            region.content, canonical_slug=region.canonical_slug
        )
        return IdentifierToken(raw=region.content, slug=slug, mode=mode)

    def instantiate_text(self, content: str, mode: Literal["A", "B"] = "B") -> str:
        """Atalho para canonização direta a partir de string."""
        slug = canonicalize_identifier(content)
        return f"[IDX:{slug}]" if mode == "B" else f"<ID>{slug}</ID>"
