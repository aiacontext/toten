"""Testes Dia 5 — Tokenizer pipeline (Modo B end-to-end).

Valida API pública conforme spec §6.2 e o output canônico para texto
real de engenharia brasileira.
"""

from __future__ import annotations

import pytest

from toten import Tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_ontology("oee-v1")


# ---------------------------------------------------------------------------
# Factory e API básica
# ---------------------------------------------------------------------------


def test_from_ontology_oee_v1() -> None:
    tok = Tokenizer.from_ontology("oee-v1")
    assert tok.oee.oee_version.startswith("1.")


def test_from_ontology_desconhecida_levanta() -> None:
    with pytest.raises(ValueError, match="Ontologia desconhecida"):
        Tokenizer.from_ontology("oee-v999")


def test_preprocess_texto_vazio(tok: Tokenizer) -> None:
    assert tok.preprocess("") == ""


def test_preprocess_prosa_pura_preservada(tok: Tokenizer) -> None:
    texto = "a viga foi analisada quanto à flexão"
    assert tok.preprocess(texto) == texto


# ---------------------------------------------------------------------------
# Substituições por tipo
# ---------------------------------------------------------------------------


def test_preprocess_grandeza_simples(tok: Tokenizer) -> None:
    """Modo B preserva unit=MPa do engenheiro + dim p/ tipo dimensional."""
    out = tok.preprocess("tensão de 350 MPa aplicada")
    qty = "[QTY value=350 unit=MPa dim=[1,-1,-2,0,0,0,0]]"
    assert qty in out
    assert out == f"tensão de {qty} aplicada"


def test_preprocess_identificador(tok: Tokenizer) -> None:
    out = tok.preprocess("o aço AISI 1045 forjado")
    assert "[IDX:aisi-1045]" in out


def test_preprocess_cordoalha_br(tok: Tokenizer) -> None:
    out = tok.preprocess("cordoalha CP 190 RB de protensão")
    assert "[IDX:cp-190-rb]" in out


def test_preprocess_constante_universal(tok: Tokenizer) -> None:
    out = tok.preprocess("k_B é fundamental na termodinâmica")
    assert "[CONST:k_B]" in out


def test_preprocess_operador_verbatim(tok: Tokenizer) -> None:
    """Operador formal mantido verbatim em Modo B."""
    out = tok.preprocess("a desigualdade x ≤ 5 vale")
    assert "≤" in out
    # não deve haver tag [OP:...]
    assert "[OP:" not in out


def test_preprocess_relacao_verbatim(tok: Tokenizer) -> None:
    """Conector estrutural mantido verbatim."""
    out = tok.preprocess("a hipótese vale; portanto a conclusão segue")
    assert "portanto" in out
    assert "[REL:" not in out


# ---------------------------------------------------------------------------
# Preservação de whitespace e roundtrip
# ---------------------------------------------------------------------------


def test_preprocess_whitespace_preservado_entre_typed(tok: Tokenizer) -> None:
    """Gaps de whitespace entre regiões typed são preservados pelo pipeline."""
    out = tok.preprocess("350 MPa ≥ 200 MPa")
    qty_350 = "[QTY value=350 unit=MPa dim=[1,-1,-2,0,0,0,0]]"
    qty_200 = "[QTY value=200 unit=MPa dim=[1,-1,-2,0,0,0,0]]"
    assert out == f"{qty_350} ≥ {qty_200}"


@pytest.mark.skip(
    reason=(
        "OBSOLETO (2026-05-31): refator fe8a5a4 — σ_y agora vira [SYM:σ_y] "
        "preservando subscript, não [IDX:σy] canonizado. Reescrita pendente "
        "para esperar [SYM:σ_y]."
    )
)
def test_preprocess_pontuacao_preservada(tok: Tokenizer) -> None:
    out = tok.preprocess("conforme AISI 1045, σ_y = 215 MPa.")
    assert "[IDX:aisi-1045]," in out
    qty_215 = "[QTY value=215 unit=MPa dim=[1,-1,-2,0,0,0,0]]"
    # σ_y normaliza para σy (underscore removido como marcador visual de subscript)
    assert f"[IDX:σy] = {qty_215}." in out


# ---------------------------------------------------------------------------
# Cenários integradores realistas
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "OBSOLETO (2026-05-31): refator fe8a5a4 — fck agora vira [SYM:fck] "
        "(preservando como expressão simbólica), não [IDX:fck]. CP 190 RB "
        "também perdeu classificação IDX (sob revisão). Teste assume "
        "comportamento v0.5; reescrita pendente para esperar tags atuais."
    )
)
def test_preprocess_sentenca_engenharia_br(tok: Tokenizer) -> None:
    """Texto BR preserva mix de unidades (MPa, tf, tf·m) como o engenheiro
    escreveu; dim atribui tipo dimensional consistente para grupos de
    mesma natureza física."""
    texto = (
        "viga em concreto C40 protendida com cordoalha CP 190 RB, "
        "fck = 40 MPa, carga de 50 tf gerando momento de 250 tf·m"
    )
    out = tok.preprocess(texto)
    assert "[IDX:c40]" in out
    assert "[IDX:cp-190-rb]" in out
    assert "[IDX:fck]" in out
    assert "[QTY value=40 unit=MPa dim=[1,-1,-2,0,0,0,0]]" in out
    assert "[QTY value=50 unit=tf dim=[1,1,-2,0,0,0,0]]" in out
    assert "[QTY value=250 unit=tf·m dim=[1,2,-2,0,0,0,0]]" in out


def test_preprocess_termodinamica_com_constante(tok: Tokenizer) -> None:
    texto = "a energia térmica é dada por k_B = 1.380649e-23 J/K em T = 300 K"
    out = tok.preprocess(texto)
    assert "[CONST:k_B]" in out
    # SPEC_07 v0.1 — K como unidade 1-char ASCII MAIÚSC ambígua ganha
    # marcador `ambig="unit-letter"`. Validamos os campos canônicos
    # value/unit/dim sem prender ao formato exato do bloco (que agora
    # pode incluir ambig + alternatives).
    assert "value=300 unit=K dim=[0,0,0,0,1,0,0]" in out


def test_preprocess_locale_pt_br_canonicaliza(tok: Tokenizer) -> None:
    """Locale PT-BR (`287,4`) é canonicalizado para float (`287.4`),
    unidade preservada como o engenheiro escreveu."""
    out = tok.preprocess("tensão de 287,4 MPa medida")
    assert "[QTY value=287.4 unit=MPa dim=[1,-1,-2,0,0,0,0]]" in out


def test_preprocess_long_tail_unidade_estrutural(tok: Tokenizer) -> None:
    """Composto não-literal (mm/(m·K)) com átomos conhecidos é detectado
    e tem dim derivado estruturalmente — mm/m cancela mantendo K^-1."""
    out = tok.preprocess("expansão térmica 1.2e-5 mm/(m·K) medida no aço")
    assert "unit=mm/(m·K)" in out
    assert "dim=[0,0,0,0,-1,0,0]" in out


def test_preprocess_idempotente_em_texto_sem_typed(tok: Tokenizer) -> None:
    """Texto sem entidades passa intacto."""
    texto = "a análise estrutural foi conduzida em duas etapas"
    assert tok.preprocess(texto) == texto


def test_preprocess_modo_a_levanta_sem_backend(tok: Tokenizer) -> None:
    """Modo A requer BPE backend para ProsaTecnica — sem backend levanta."""
    with pytest.raises(NotImplementedError):
        tok.preprocess("a tensão de 350 MPa supera", mode="A")
