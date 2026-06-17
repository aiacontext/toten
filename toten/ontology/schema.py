"""Esquema pydantic da OEE.

Modela `data/oee-v1.yaml` em estruturas tipadas com validação automática.
As invariantes da OEE v1.0 (P1–P5) são verificadas no carregamento via
validators de modelo.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from toten.ontology.types import (
    TIPOS_V1,
    TIPOS_V1_1,
    TIPOS_V1_2,
    TipoNome,
)


def _expected_types_for_version(version: str) -> frozenset[TipoNome]:
    """Conjunto canônico de tipos esperados conforme a versão da OEE.

    v1.0 → 7 tipos (sem Numero, sem Referencia)
    v1.1 → 8 tipos (com Numero — P5 extensibilidade)
    v1.2+ → 9 tipos (com Referencia — P5 extensibilidade)
    """
    if version.startswith("1.0"):
        return TIPOS_V1
    if version.startswith("1.1"):
        return TIPOS_V1_1
    if version.startswith("1."):
        return TIPOS_V1_2
    msg = f"versão OEE não suportada: {version!r}"
    raise ValueError(msg)


class Principio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    nome: str = Field(min_length=1)
    declaracao: str = Field(min_length=1)


class TipoOntologico(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    natureza: str = Field(min_length=1)
    propriedades_intrinsecas: dict[str, str] = Field(min_length=1)
    invariantes: list[str] = Field(min_length=1)
    exemplos: list[str] = Field(min_length=1)
    composicoes: list[TipoNome] | None = None
    # Campos documentais opcionais (v1.1+ — documentam extensões e
    # justificativas ontológicas/empíricas/SOTA). Validação literária
    # apenas; não usados pelo runtime do classifier/instantiator.
    extensao_v1_1: str | None = None
    racional_ontologico: str | None = None

    @model_validator(mode="after")
    def _propriedades_nao_vazias(self) -> TipoOntologico:
        for nome, descricao in self.propriedades_intrinsecas.items():
            if not nome.strip():
                msg = "propriedade com nome vazio"
                raise ValueError(msg)
            if not descricao.strip():
                msg = f"propriedade '{nome}' sem descrição"
                raise ValueError(msg)
        return self


class ResolucaoAmbiguidade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    estrategia: str
    ordem: list[TipoNome] = Field(min_length=1)

    @model_validator(mode="after")
    def _sem_duplicatas(self) -> ResolucaoAmbiguidade:
        # Verificação simples — completude vs versão é feita no nível OEE
        # (onde temos acesso a oee_version).
        if len(set(self.ordem)) != len(self.ordem):
            msg = "resolucao_ambiguidade.ordem contém duplicatas"
            raise ValueError(msg)
        return self


class OEE(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    oee_version: str = Field(min_length=1)
    nome: str = Field(min_length=1)
    descricao: str = Field(min_length=1)
    referencias: list[str] = Field(min_length=1)
    principios: list[Principio] = Field(min_length=1)
    tipos: dict[TipoNome, TipoOntologico]
    resolucao_ambiguidade: ResolucaoAmbiguidade

    @model_validator(mode="after")
    def _versao_completa(self) -> OEE:
        """Tipos declarados devem bater EXATAMENTE com o conjunto canônico
        da versão (TIPOS_V1 para v1.0, TIPOS_V1_1 para v1.1+)."""
        expected = _expected_types_for_version(self.oee_version)
        declarados = set(self.tipos.keys())
        if declarados != expected:
            faltando = expected - declarados
            sobrando = declarados - expected
            msg = (
                f"OEE {self.oee_version} deve declarar exatamente "
                f"{len(expected)} tipos canônicos "
                f"(faltando={sorted(faltando)}, sobrando={sorted(sobrando)})"
            )
            raise ValueError(msg)

        # E a ordem de resolução deve cobrir todos os tipos declarados
        ordem_set = set(self.resolucao_ambiguidade.ordem)
        if ordem_set != expected:
            faltando = expected - ordem_set
            sobrando = ordem_set - expected
            msg = (
                f"resolucao_ambiguidade.ordem (OEE {self.oee_version}) deve "
                f"cobrir exatamente os {len(expected)} tipos canônicos "
                f"(faltando={sorted(faltando)}, sobrando={sorted(sobrando)})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _composicoes_referenciam_tipos_declarados(self) -> OEE:
        for nome, tipo in self.tipos.items():
            if tipo.composicoes is None:
                continue
            for referido in tipo.composicoes:
                if referido not in self.tipos:
                    msg = (
                        f"tipo '{nome}' declara composição com '{referido}', "
                        "que não está declarado na ontologia"
                    )
                    raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _constante_universal_compoe_grandeza(self) -> OEE:
        cu = self.tipos.get(TipoNome.CONSTANTE_UNIVERSAL)
        if cu is None:
            return self
        if cu.composicoes is None or TipoNome.GRANDEZA_FISICA not in cu.composicoes:
            msg = (
                "ConstanteUniversal deve declarar composição com GrandezaFisica "
                "(invariante de §2.2 da spec)"
            )
            raise ValueError(msg)
        return self
