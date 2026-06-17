"""Tokenizer pipeline — fachada que integra Camada 2 + Camada 3.

API pública conforme spec §6.2:

    from toten import Tokenizer

    tok = Tokenizer.from_ontology("oee-v1")
    canonical = tok.preprocess(
        "A tensão de 350 MPa em AISI 1045 supera σ_y = 215 MPa"
    )
    # → "A tensão de [QTY value=350 unit=MPa] em [IDX:aisi-1045] "
    #   "supera [IDX:σ-y] = [QTY value=215 unit=MPa]"

O pipeline:

1. Roda o `OntologicalClassifier` (Camada 2) para identificar regiões
   tipadas — `Region(tipo, start, end, content)`.
2. Para cada região, delega ao instanciador apropriado (Camada 3) que
   produz a representação canônica conforme o modo (A ou B).
3. Reassembla o texto preservando whitespace nos gaps que a Camada 2
   não emitiu como regiões.

Modo B é o caso completo (model-agnostic). Modo A é definido pela
interface mas requer materialização de vocab no treino do modelo nativo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from toten.activation import (
    ActivatedPrompt,
    default_activation_prompt,
)
from toten.classifier import OntologicalClassifier
from toten.instantiators import (
    BPEBackend,
    ConstanteUniversalInstantiator,
    ExpressaoSimbolicaInstantiator,
    GrandezaFisicaInstantiator,
    IdentificadorTecnicoInstantiator,
    NumeroInstantiator,
    OperadorFormalInstantiator,
    ProsaTecnicaInstantiator,
    RelacaoEstruturalInstantiator,
    ReferenciaInstantiator,
)
from toten.ontology.loader import load_oee
from toten.ontology.schema import OEE
from toten.ontology.types import TipoNome


class Tokenizer:
    """Fachada Modo B do TOTEN com ativação integrada."""

    def __init__(
        self,
        oee: OEE | None = None,
        classifier: OntologicalClassifier | None = None,
        bpe_backend: BPEBackend | None = None,
        activation_lang: str = "pt-br",
        activation_prompts: dict[str, str] | None = None,
    ) -> None:
        self._oee = oee if oee is not None else load_oee()
        self._classifier = classifier or OntologicalClassifier(oee=self._oee)
        self._instantiators = {
            TipoNome.GRANDEZA_FISICA: GrandezaFisicaInstantiator(),
            TipoNome.IDENTIFICADOR_TECNICO: IdentificadorTecnicoInstantiator(),
            TipoNome.OPERADOR_FORMAL: OperadorFormalInstantiator(),
            TipoNome.CONSTANTE_UNIVERSAL: ConstanteUniversalInstantiator(),
            TipoNome.RELACAO_ESTRUTURAL: RelacaoEstruturalInstantiator(),
            TipoNome.EXPRESSAO_SIMBOLICA: ExpressaoSimbolicaInstantiator(),
            TipoNome.NUMERO: NumeroInstantiator(),  # v1.1 — 8º tipo
            TipoNome.REFERENCIA: ReferenciaInstantiator(),  # v1.2 — 9º tipo
            TipoNome.PROSA_TECNICA: ProsaTecnicaInstantiator(bpe_backend=bpe_backend),
        }
        self._activation_lang = activation_lang
        self._activation_overrides: dict[str, str] = dict(activation_prompts or {})

    @classmethod
    def from_ontology(cls, ontology_name: str = "oee-v1") -> Tokenizer:
        """Carrega ontologia versionada e constrói tokenizer.

        Hoje apenas `oee-v1` é distribuída — corresponde ao
        `data/oee-v1.yaml`. Versões futuras (v1.1+, v2) entrariam aqui
        via lookup explícito.
        """
        if ontology_name != "oee-v1":
            msg = f"Ontologia desconhecida: {ontology_name!r}. Disponíveis: oee-v1."
            raise ValueError(msg)
        return cls(oee=load_oee())

    @classmethod
    def from_oee_path(cls, path: Path | str) -> Tokenizer:
        """Constrói tokenizer carregando OEE de caminho customizado."""
        return cls(oee=load_oee(Path(path)))

    @property
    def oee(self) -> OEE:
        return self._oee

    def preprocess(self, text: str, mode: Literal["A", "B"] = "B") -> str:
        """Aplica Camada 2 + Camada 3 e devolve texto canônico.

        Whitespace nos gaps entre regiões é preservado verbatim — o
        resultado é uma reescrita do texto original onde regiões
        tipadas viram tokens ontologicamente tagados.

        Em Modo B esta é a interface universal: consumível por qualquer
        LLM frozen. Em Modo A exige BPE backend materializado em
        `ProsaTecnicaInstantiator`.
        """
        if not text:
            return ""
        regions = self._classifier.classify(text)
        chunks: list[str] = []
        cursor = 0
        for region in regions:
            if region.start > cursor:
                chunks.append(text[cursor : region.start])
            inst = self._instantiators[region.tipo]
            # AUTORIDADE ÚNICA — a Camada 2 (classifier) já decidiu o tipo de
            # cada região. O instanciador APENAS formata (`.text`), nunca
            # re-decide. Nenhuma re-validação aqui ou nos instanciadores.
            token = inst.instantiate(region, mode=mode)
            chunks.append(token.text)
            cursor = region.end
        if cursor < len(text):
            chunks.append(text[cursor:])
        return "".join(chunks)

    def activation_prompt(self, lang: str | None = None) -> str:
        """Devolve o prompt de ativação canônico para o idioma.

        Hierarquia de resolução:
        1. Override explícito passado no construtor (`activation_prompts`).
        2. Default cacheado para o idioma (lido de `data/activation_prompt_*.md`).
        """
        lang = lang or self._activation_lang
        if lang in self._activation_overrides:
            return self._activation_overrides[lang]
        return default_activation_prompt(lang)

    def preprocess_for_llm(
        self,
        text: str,
        lang: str | None = None,
        mode: Literal["A", "B"] = "B",
    ) -> ActivatedPrompt:
        """Empacota Modo B + ativação canônica em `ActivatedPrompt`.

        Esta é a interface recomendada para LLMs frozen — o piloto
        adversarial mostrou que o preprocessing sozinho não move
        consistentemente o ponteiro em frontier LLMs; a ativação
        explícita é o que materializa o ganho.

        Use `.ready_for_llm` para colar em chat ou `.as_messages()`
        para enviar via API.
        """
        lang = lang or self._activation_lang
        preprocessed = self.preprocess(text, mode=mode)
        activation = self.activation_prompt(lang)
        return ActivatedPrompt(
            activation=activation,
            preprocessed=preprocessed,
            language=lang,
        )
