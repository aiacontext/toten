"""Instanciador de RelacaoEstrutural.

Conectores lógico-estruturais (`portanto`, `dado que`, `if and only if`)
são reconhecidos verbalmente por qualquer LLM frozen — tagging em Modo
B agregaria ruído sem ganho. Modo B: passthrough.

Modo A: delega a `BPEBackend` da ProsaTecnica (mesma natureza
linguística, mesma instanciação per OEE §2.3).

Metadata `funcao_logica` (condicional / conjuntiva / disjuntiva /
causal / conclusiva) carregada do lexicon fica disponível no Token
para uso opcional pela Camada 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from toten.classifier.region import Region
from toten.ontology.types import TipoNome

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RELATION_LEXICON_PATH = PACKAGE_ROOT / "data" / "relation_lexicon_v0.json"


@dataclass(frozen=True, slots=True)
class RelationToken:
    """Conector lógico-estrutural com metadata."""

    text: str
    mode: Literal["A", "B"]
    funcao_logica: str | None = None
    idioma: str | None = None


class RelacaoEstruturalInstantiator:
    """Camada 3 — RelacaoEstrutural. Modo B é passthrough."""

    def __init__(self, lexicon_path: Path | None = None) -> None:
        path = lexicon_path or DEFAULT_RELATION_LEXICON_PATH
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self._by_form: dict[str, dict] = {}
        for idioma in ("pt-br", "en"):
            for entry in payload.get(idioma, []):
                forma = entry.get("forma")
                if not forma:
                    continue
                self._by_form[forma.lower()] = {**entry, "idioma": idioma}

    def instantiate(
        self, region: Region, mode: Literal["A", "B"] = "B"
    ) -> RelationToken:
        if region.tipo is not TipoNome.RELACAO_ESTRUTURAL:
            msg = (
                "RelacaoEstruturalInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava RelacaoEstrutural"
            )
            raise TypeError(msg)
        return self._make_token(region.content, mode)

    def instantiate_text(self, content: str, mode: Literal["A", "B"] = "B") -> str:
        # passthrough — texto canônico mantém o conector verbatim
        return content

    def _make_token(self, content: str, mode: str) -> RelationToken:
        entry = self._by_form.get(content.lower())
        return RelationToken(
            text=content,
            mode=mode,  # type: ignore[arg-type]
            funcao_logica=entry.get("funcao") if entry else None,
            idioma=entry.get("idioma") if entry else None,
        )
