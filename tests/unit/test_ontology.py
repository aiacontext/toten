"""Testes da Camada 1 — carregamento e validação da OEE v1.0."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from toten.ontology import (
    OEE,
    TipoNome,
    default_oee_path,
    load_oee,
)
from toten.ontology.types import TIPOS_V1, TIPOS_V1_1, TIPOS_V1_2


def test_default_oee_path_existe() -> None:
    assert default_oee_path().is_file(), (
        f"OEE v1.0 default não encontrada em {default_oee_path()}"
    )


def test_load_oee_default_nao_levanta() -> None:
    oee = load_oee()
    assert isinstance(oee, OEE)
    assert oee.oee_version.startswith("1.")


def test_oee_default_declara_tipos_da_versao() -> None:
    """OEE default (v1.2) declara 9 tipos. v1.1 declara 8. v1.0 declara 7.
    Cada versão deve ser self-consistent."""
    oee = load_oee()
    if oee.oee_version.startswith("1.0"):
        assert set(oee.tipos.keys()) == TIPOS_V1
    elif oee.oee_version.startswith("1.1"):
        assert set(oee.tipos.keys()) == TIPOS_V1_1
    elif oee.oee_version.startswith("1."):
        assert set(oee.tipos.keys()) == TIPOS_V1_2
    else:
        raise AssertionError(f"versão OEE desconhecida: {oee.oee_version}")


@pytest.mark.parametrize("tipo", list(TipoNome))
def test_cada_tipo_tem_propriedades_invariantes_e_exemplos(tipo: TipoNome) -> None:
    oee = load_oee()
    t = oee.tipos[tipo]
    assert t.natureza.strip()
    assert len(t.propriedades_intrinsecas) >= 1
    assert len(t.invariantes) >= 1
    assert len(t.exemplos) >= 1


def test_constante_universal_compoe_grandeza_fisica() -> None:
    oee = load_oee()
    cu = oee.tipos[TipoNome.CONSTANTE_UNIVERSAL]
    assert cu.composicoes is not None
    assert TipoNome.GRANDEZA_FISICA in cu.composicoes


def test_grandeza_fisica_tem_invariante_dimensional() -> None:
    oee = load_oee()
    gf = oee.tipos[TipoNome.GRANDEZA_FISICA]
    assert any("dimensão" in inv.lower() or "dimensao" in inv.lower() for inv in gf.invariantes)


def test_resolucao_ambiguidade_cobre_todos_os_tipos() -> None:
    oee = load_oee()
    if oee.oee_version.startswith("1.0"):
        expected = TIPOS_V1
    elif oee.oee_version.startswith("1.1"):
        expected = TIPOS_V1_1
    else:
        expected = TIPOS_V1_2
    assert set(oee.resolucao_ambiguidade.ordem) == expected
    assert oee.resolucao_ambiguidade.ordem[0] == TipoNome.GRANDEZA_FISICA


def test_oee_eh_imutavel() -> None:
    oee = load_oee()
    with pytest.raises(ValidationError):
        oee.oee_version = "2.0"  # type: ignore[misc]


def _yaml_minimo_valido() -> dict:
    return {
        "oee_version": "1.0",
        "nome": "OEE Mínima Test",
        "descricao": "fixture",
        "referencias": ["test"],
        "principios": [
            {"id": "P1", "nome": "P1", "declaracao": "d"},
        ],
        "tipos": {
            "GrandezaFisica": {
                "natureza": "n",
                "propriedades_intrinsecas": {"valor": "escalar"},
                "invariantes": ["dimensão sempre presente"],
                "exemplos": ["1 m"],
            },
            "ProsaTecnica": {
                "natureza": "n",
                "propriedades_intrinsecas": {"idioma": "iso"},
                "invariantes": ["não-numérico"],
                "exemplos": ["texto"],
            },
            "IdentificadorTecnico": {
                "natureza": "n",
                "propriedades_intrinsecas": {"classe": "x"},
                "invariantes": ["identidade preservada"],
                "exemplos": ["ER70S-6"],
            },
            "OperadorFormal": {
                "natureza": "n",
                "propriedades_intrinsecas": {"aridade": "1"},
                "invariantes": ["atômico"],
                "exemplos": ["="],
            },
            "ConstanteUniversal": {
                "natureza": "n",
                "propriedades_intrinsecas": {
                    "simbolo": "s",
                    "grandeza_associada": "GrandezaFisica",
                },
                "invariantes": ["símbolo identifica unicamente"],
                "composicoes": ["GrandezaFisica"],
                "exemplos": ["π"],
            },
            "RelacaoEstrutural": {
                "natureza": "n",
                "propriedades_intrinsecas": {"funcao_logica": "condicional"},
                "invariantes": ["estrutura lógica"],
                "exemplos": ["então"],
            },
            "ExpressaoSimbolica": {
                "natureza": "n",
                "propriedades_intrinsecas": {"forma_canonica": "sympy.Expr"},
                "invariantes": ["parseável por sympy"],
                "composicoes": ["IdentificadorTecnico", "OperadorFormal"],
                "exemplos": ["pl/12"],
            },
        },
        "resolucao_ambiguidade": {
            "estrategia": "maior_especificidade",
            "ordem": [
                "GrandezaFisica",
                "ConstanteUniversal",
                "IdentificadorTecnico",
                "OperadorFormal",
                "ExpressaoSimbolica",
                "RelacaoEstrutural",
                "ProsaTecnica",
            ],
        },
    }


def test_yaml_minimo_valido_passa() -> None:
    OEE.model_validate(_yaml_minimo_valido())


def test_v1_rejeita_tipo_faltando() -> None:
    raw = _yaml_minimo_valido()
    del raw["tipos"]["OperadorFormal"]
    with pytest.raises(ValidationError, match=r"deve declarar exatamente \d+ tipos canônicos"):
        OEE.model_validate(raw)


def test_v1_rejeita_constante_sem_composicao_de_grandeza() -> None:
    raw = _yaml_minimo_valido()
    raw["tipos"]["ConstanteUniversal"]["composicoes"] = []
    with pytest.raises(ValidationError, match="ConstanteUniversal deve declarar"):
        OEE.model_validate(raw)


def test_composicao_para_tipo_nao_declarado_eh_invalida() -> None:
    raw = _yaml_minimo_valido()
    raw["tipos"]["ConstanteUniversal"]["composicoes"] = ["TipoInexistente"]
    with pytest.raises(ValidationError):
        OEE.model_validate(raw)


def test_oee_path_inexistente_levanta_filenotfound(tmp_path: Path) -> None:
    bad = tmp_path / "no_such.yaml"
    with pytest.raises(FileNotFoundError):
        load_oee(bad)


def test_oee_yaml_invalido_levanta_validation_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            oee_version: "1.0"
            nome: ""
            descricao: "x"
            referencias: []
            principios: []
            tipos: {}
            resolucao_ambiguidade:
              estrategia: x
              ordem: []
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_oee(bad)
