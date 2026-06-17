"""Schema pydantic do EngQuant benchmark — v1.0.

Cada caso vive em um YAML; este módulo define o contrato.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# references.yaml
# ---------------------------------------------------------------------------


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9\-]*$")
    type: Literal["book", "article", "standard", "thesis", "report", "notes"]
    full_citation: str = Field(min_length=10)
    short_citation: str = Field(min_length=1)
    language: Literal["pt-br", "en", "es", "fr", "de", "it"] = "pt-br"
    isbn: str | None = None
    doi: str | None = None
    url: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)


class ReferencesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    sources: list[Reference] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unicos(self) -> ReferencesFile:
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            duplicados = sorted({x for x in ids if ids.count(x) > 1})
            msg = f"IDs duplicados em references.sources: {duplicados}"
            raise ValueError(msg)
        return self

    def get(self, source_id: str) -> Reference | None:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None


# ---------------------------------------------------------------------------
# tags_vocabulary.yaml
# ---------------------------------------------------------------------------


class TagsVocabulary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    high_risk_tags: dict[str, list[str]] = Field(min_length=1)
    contextual_tags: list[str] = Field(default_factory=list)

    @property
    def all_high_risk_tags(self) -> frozenset[str]:
        """Achata high_risk_tags (por categoria) em conjunto único."""
        all_tags: set[str] = set()
        for cat_tags in self.high_risk_tags.values():
            all_tags.update(cat_tags)
        return frozenset(all_tags)

    @property
    def all_contextual_tags(self) -> frozenset[str]:
        return frozenset(self.contextual_tags)

    @property
    def all_known_tags(self) -> frozenset[str]:
        return self.all_high_risk_tags | self.all_contextual_tags


# ---------------------------------------------------------------------------
# Caso (case.yaml)
# ---------------------------------------------------------------------------


class Asset(BaseModel):
    """Asset associado ao caso (figura SVG, PDF-fonte, tabela)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    type: Literal["svg", "pdf", "png", "table_csv", "table_md"]
    path: str = Field(min_length=1)  # relativo ao YAML
    description: str = Field(min_length=1)
    embed_inline: bool = False


class CaseMetadata(BaseModel):
    """Metadados taxonômicos e contextuais do caso."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discipline_folder: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    language: Literal["pt-br", "en"] = "pt-br"
    has_figure: bool = False
    has_table: bool = False
    n_subgrandezas: int | None = Field(default=None, ge=1)


class CaseReference(BaseModel):
    """Procedência bibliográfica do caso."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    chapter: str | int | None = None
    section: str | None = None
    page_range: str | None = None
    notes: str | None = None


class ExpectedNorm(BaseModel):
    """Cláusulas normativas que se espera o LLM citar (ou referenciar)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    clauses: list[str] = Field(default_factory=list)


class GabaritoEntry(BaseModel):
    """Uma sub-grandeza auditável no gabarito.

    Suporta dois modos:

    1. **Numérico puro:** problema com valores físicos concretos.
       `value` é o valor numérico (ex.: 30672); `unit` é a unidade física
       (ex.: "MPa"); `symbolic_form` é `None`.

    2. **Simbólico/paramétrico:** problema com variáveis livres (`p`, `l`,
       `E`, etc.) onde a forma fechada é a resposta canônica. `value` é
       o **coeficiente adimensional** da forma fechada (ex.: 0.03937 para
       `(pl²/32)·∛2`); `unit` é a **base simbólica** (ex.: "pl²", "pl",
       "l", "adim"); `symbolic_form` carrega a **expressão fechada
       literal** (ex.: "(pl²/32)·∛2"). O grader compara via SymPy ou
       avaliação numérica em pontos.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    tolerance_relative: float = Field(ge=0.0, le=0.5)
    formula: str | None = None
    inputs: dict[str, float | str] | None = None
    notes: str | None = None
    symbolic_form: str | None = Field(default=None, min_length=1)


class Case(BaseModel):
    """Um caso do EngQuant benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9]*(-[a-z0-9]+)+$")
    title: str = Field(min_length=1)
    metadata: CaseMetadata
    reference: CaseReference
    tags: list[str] = Field(min_length=1)
    assets: list[Asset] = Field(default_factory=list)
    prompt_baseline: str = Field(min_length=20)
    expected_signals: list[str] = Field(default_factory=list)
    expected_norms: list[ExpectedNorm] = Field(default_factory=list)
    gabarito: list[GabaritoEntry] = Field(min_length=1)
    memoria_calculo: dict[str, str] | None = None
    """Materiais que pertencem à RESOLUÇÃO, não ao enunciado: diagrama
    de momento (pontos notáveis), função M(x) por trechos, passos-chave.

    NÃO deve ser exibido ao usuário/LLM no prompt baseline — esses
    artefatos são parte da resposta esperada, não dica de solução.
    Decisão SPEC_06 §3.P2 (honestidade epistemológica): enunciado
    declara apenas problema; respondente produz diagrama e função
    como parte da solução.

    Chaves convencionais (todas opcionais):
        - diagrama_momento: pontos notáveis (renderizado como string)
        - funcao_por_trechos: M(x) por trechos com domínios explícitos
        - passos_chave: roteiro abreviado da solução (opcional)
    """
    review_status: Literal["draft", "reviewed", "validated"] = "draft"
    review_notes: str | None = None
    created: str | None = None  # ISO date "YYYY-MM-DD"
    author: str | None = None

    @model_validator(mode="after")
    def _id_consistente_com_discipline(self) -> Case:
        """ID começa com o prefixo da discipline_folder."""
        prefix = self.metadata.discipline_folder
        if not self.id.startswith(f"{prefix}-"):
            msg = (
                f"case.id '{self.id}' deve começar com prefixo da "
                f"discipline_folder '{prefix}-'"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _tags_sem_duplicatas(self) -> Case:
        if len(set(self.tags)) != len(self.tags):
            duplicadas = sorted({t for t in self.tags if self.tags.count(t) > 1})
            msg = f"tags duplicadas no caso {self.id}: {duplicadas}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _n_subgrandezas_consistente(self) -> Case:
        if self.metadata.n_subgrandezas is not None:
            n = self.metadata.n_subgrandezas
            if n != len(self.gabarito):
                msg = (
                    f"metadata.n_subgrandezas={n} divergente de "
                    f"len(gabarito)={len(self.gabarito)} no caso {self.id}"
                )
                raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# manifest.yaml
# ---------------------------------------------------------------------------


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    primary_language: Literal["pt-br", "en"] = "pt-br"
    created: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Bundle compilado (saída do compiler)
# ---------------------------------------------------------------------------


class BenchmarkBundle(BaseModel):
    """Conjunto compilado pronto para automação experimental."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: Manifest
    references: ReferencesFile
    tags_vocabulary: TagsVocabulary
    cases: list[Case] = Field(min_length=1)
