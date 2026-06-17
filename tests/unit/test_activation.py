"""Tests do módulo activation — promove o contrato de ativação a
artefato de primeira classe.

Justificativa do design (spec §13.11): o piloto adversarial revelou
que Modo B passivo não move o ponteiro em frontier LLMs. A ativação
via system prompt é o que materializa o ganho. Portanto, o framework
ENTREGA ambos: texto preprocessado E prompt de ativação canônico,
empacotados em `ActivatedPrompt`.
"""

from __future__ import annotations

import pytest

from toten import (
    ActivatedPrompt,
    Tokenizer,
    default_activation_prompt,
    load_activation_prompt,
)
from toten.activation import (
    SUPPORTED_LANGUAGES,
    _activation_prompt_path,
)

# ---------------------------------------------------------------------------
# load_activation_prompt — fontes canônicas
# ---------------------------------------------------------------------------


def test_load_activation_prompt_pt_br() -> None:
    prompt = load_activation_prompt("pt-br")
    assert prompt.strip()
    # Verifica que o prompt tem as anotações documentadas
    assert "[QTY" in prompt
    assert "[IDX:" in prompt
    assert "[CONST:" in prompt
    # Vocabulário PT-BR
    assert "INSTRUÇÃO" in prompt
    assert "Tarefa:" in prompt


def test_load_activation_prompt_en() -> None:
    prompt = load_activation_prompt("en")
    assert prompt.strip()
    assert "[QTY" in prompt
    assert "[IDX:" in prompt
    assert "[CONST:" in prompt
    assert "INSTRUCTION" in prompt
    assert "Task:" in prompt


def test_load_activation_prompt_lang_invalida() -> None:
    with pytest.raises(ValueError, match="não suportado"):
        load_activation_prompt("xx")


def test_load_activation_prompt_file_missing(tmp_path, monkeypatch) -> None:
    """Se o arquivo da fonte canônica não existir, levanta FileNotFoundError."""
    from toten import activation

    fake_dir = tmp_path / "noexist"
    monkeypatch.setattr(activation, "DATA_DIR", fake_dir)
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        load_activation_prompt("pt-br")


def test_default_activation_prompt_cacheado() -> None:
    """default_activation_prompt cacheia — chamadas repetidas devolvem
    a mesma string sem re-ler do disco."""
    a = default_activation_prompt("pt-br")
    b = default_activation_prompt("pt-br")
    assert a is b  # mesmo objeto (cache hit)


def test_activation_path_helper() -> None:
    """_activation_prompt_path produz nomes consistentes (- → _)."""
    p = _activation_prompt_path("pt-br")
    assert p.name == "activation_prompt_pt_br.md"
    p2 = _activation_prompt_path("en")
    assert p2.name == "activation_prompt_en.md"


def test_supported_languages() -> None:
    assert frozenset({"pt-br", "en"}) == SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# ActivatedPrompt — dataclass + invariantes
# ---------------------------------------------------------------------------


def test_activated_prompt_basico() -> None:
    p = ActivatedPrompt(
        activation="INSTRUÇÃO: contrato.\n\nTarefa:",
        preprocessed="350 MPa",
        language="pt-br",
    )
    assert p.activation
    assert p.preprocessed == "350 MPa"
    assert p.language == "pt-br"


def test_activated_prompt_imutavel() -> None:
    p = ActivatedPrompt(activation="x", preprocessed="y", language="pt-br")
    with pytest.raises(AttributeError):
        p.activation = "z"  # type: ignore[misc]


def test_activated_prompt_rejeita_activation_vazia() -> None:
    with pytest.raises(ValueError, match="activation"):
        ActivatedPrompt(activation="  ", preprocessed="y", language="pt-br")


def test_activated_prompt_rejeita_preprocessed_vazio() -> None:
    with pytest.raises(ValueError, match="preprocessed"):
        ActivatedPrompt(activation="x", preprocessed="", language="pt-br")


def test_activated_prompt_rejeita_lang_invalida() -> None:
    with pytest.raises(ValueError, match="language"):
        ActivatedPrompt(activation="x", preprocessed="y", language="xx")


def test_activated_prompt_ready_for_llm() -> None:
    """ready_for_llm concatena ativação + texto com separação clara."""
    p = ActivatedPrompt(
        activation="INSTRUÇÃO: usa as tags.\n\nTarefa:",
        preprocessed="qual a [QTY value=350 unit=MPa]?",
        language="pt-br",
    )
    out = p.ready_for_llm
    assert "INSTRUÇÃO: usa as tags." in out
    assert "Tarefa:" in out
    assert "[QTY value=350 unit=MPa]" in out
    # ativação precede texto
    assert out.index("Tarefa:") < out.index("[QTY")


def test_activated_prompt_as_messages_formato_api() -> None:
    """as_messages() devolve lista no formato Anthropic/OpenAI."""
    p = ActivatedPrompt(
        activation="INSTRUÇÃO: ...\nTarefa:",
        preprocessed="[QTY value=10 unit=tf]",
        language="pt-br",
    )
    msgs = p.as_messages()
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "INSTRUÇÃO: ...\nTarefa:"}
    assert msgs[1] == {"role": "user", "content": "[QTY value=10 unit=tf]"}


# ---------------------------------------------------------------------------
# Tokenizer integration — activation_prompt + preprocess_for_llm
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_ontology("oee-v1")


def test_tokenizer_activation_prompt_default_pt_br(tok: Tokenizer) -> None:
    prompt = tok.activation_prompt()
    assert "INSTRUÇÃO" in prompt
    assert "[QTY" in prompt


def test_tokenizer_activation_prompt_en(tok: Tokenizer) -> None:
    prompt = tok.activation_prompt("en")
    assert "INSTRUCTION" in prompt
    assert "Task:" in prompt


def test_tokenizer_preprocess_for_llm_full_bundle(tok: Tokenizer) -> None:
    texto = "Uma viga SAE 1045 tem σ_y = 350 MPa."
    result = tok.preprocess_for_llm(texto)
    assert isinstance(result, ActivatedPrompt)
    assert result.language == "pt-br"
    # Texto preprocessado contém as tags do framework
    assert "[IDX:sae-1045]" in result.preprocessed
    assert "[QTY value=350 unit=MPa" in result.preprocessed
    # Activation contém o contrato canônico
    assert "INSTRUÇÃO" in result.activation


def test_tokenizer_preprocess_for_llm_lang_override(tok: Tokenizer) -> None:
    texto = "350 MPa applied"
    result = tok.preprocess_for_llm(texto, lang="en")
    assert result.language == "en"
    assert "INSTRUCTION" in result.activation
    assert "[QTY value=350 unit=MPa" in result.preprocessed


def test_tokenizer_preprocess_for_llm_messages_format(tok: Tokenizer) -> None:
    texto = "350 MPa em SAE 1045"
    result = tok.preprocess_for_llm(texto)
    msgs = result.as_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "[QTY value=350" in msgs[1]["content"]


def test_tokenizer_activation_override_no_constructor() -> None:
    """Constructor aceita activation_prompts custom (override do default)."""
    custom = "CUSTOM CONTRACT INSTRUCTION\n\nTarefa:"
    tok = Tokenizer(activation_prompts={"pt-br": custom})
    prompt = tok.activation_prompt()
    assert prompt == custom

    # preprocess_for_llm usa o override
    result = tok.preprocess_for_llm("350 MPa")
    assert result.activation == custom


def test_tokenizer_activation_lang_padrao_no_constructor() -> None:
    """Constructor aceita activation_lang para fixar idioma default."""
    tok = Tokenizer(activation_lang="en")
    assert "INSTRUCTION" in tok.activation_prompt()
    result = tok.preprocess_for_llm("steel sample")
    assert result.language == "en"


# ---------------------------------------------------------------------------
# Pilot replication — confirma que ativação canônica está acessível
# ---------------------------------------------------------------------------


def test_pilot_replication_ready_for_llm() -> None:
    """O Caso 1 do piloto v2 (Condição C) pode ser regenerado integralmente
    a partir de Tokenizer.preprocess_for_llm — garantia de sincronia entre
    fonte canônica e material experimental."""
    tok = Tokenizer.from_ontology("oee-v1")
    texto = (
        "Estime o limite de fadiga sem correções Se' de um aço SAE 4140 H "
        "temperado e revenido com σ_ut = 1080 MPa."
    )
    result = tok.preprocess_for_llm(texto)
    bloco = result.ready_for_llm
    # Estrutura esperada
    assert bloco.startswith("INSTRUÇÃO")
    assert "[IDX:sae-4140-h]" in bloco
    # Evolução 4 (R2L grouping, Singh-Strouse 2024): valores ≥1000 ganham
    # campo `r2l=` entre value e unit — endereça lacuna SOTA Number
    # Cookbook (Yang ICLR 2025).
    assert 'value=1080 r2l="1 080" unit=MPa' in bloco
    assert "Tarefa:" in bloco
