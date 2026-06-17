"""Testes Dia 5 — instanciadores restantes da Camada 3.

Cobre ProsaTecnica (passthrough Modo B, NotImplementedError Modo A
sem backend), OperadorFormal (passthrough + metadata),
ConstanteUniversal (composição Quantity, fallback para constantes
fora do lexicon), RelacaoEstrutural (passthrough + metadata).
"""

from __future__ import annotations

import pytest

from toten.classifier import Region
from toten.instantiators import (
    ConstanteUniversalInstantiator,
    OperadorFormalInstantiator,
    ProsaTecnicaInstantiator,
    RelacaoEstruturalInstantiator,
)
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# ProsaTecnica
# ---------------------------------------------------------------------------


def test_prosa_modo_b_passthrough() -> None:
    inst = ProsaTecnicaInstantiator()
    region = Region(TipoNome.PROSA_TECNICA, 0, 18, "a tensão máxima foi")
    token = inst.instantiate(region, mode="B")
    assert token.text == "a tensão máxima foi"
    assert token.mode == "B"
    assert token.ids is None


def test_prosa_modo_b_atalho_string() -> None:
    inst = ProsaTecnicaInstantiator()
    assert inst.instantiate_text("considerando o regime elástico") == (
        "considerando o regime elástico"
    )


def test_prosa_modo_a_sem_backend_levanta() -> None:
    inst = ProsaTecnicaInstantiator()
    region = Region(TipoNome.PROSA_TECNICA, 0, 4, "test")
    with pytest.raises(NotImplementedError, match="BPEBackend"):
        inst.instantiate(region, mode="A")


def test_prosa_modo_a_com_backend_custom() -> None:
    class FakeBPE:
        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

    inst = ProsaTecnicaInstantiator(bpe_backend=FakeBPE())
    region = Region(TipoNome.PROSA_TECNICA, 0, 4, "test")
    token = inst.instantiate(region, mode="A")
    assert token.ids == (116, 101, 115, 116)


def test_prosa_rejeita_tipo_errado() -> None:
    inst = ProsaTecnicaInstantiator()
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
    with pytest.raises(TypeError, match="ProsaTecnica"):
        inst.instantiate(region)


# ---------------------------------------------------------------------------
# OperadorFormal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simbolo", ["=", "≤", "≥", "≠", "→", "⇒", "∑", "∫", "∂"])
def test_operador_modo_b_verbatim(simbolo: str) -> None:
    inst = OperadorFormalInstantiator()
    region = Region(TipoNome.OPERADOR_FORMAL, 0, len(simbolo), simbolo)
    token = inst.instantiate(region, mode="B")
    assert token.text == simbolo
    assert token.symbol == simbolo


def test_operador_carrega_metadata() -> None:
    inst = OperadorFormalInstantiator()
    region = Region(TipoNome.OPERADOR_FORMAL, 0, 1, "≤")
    token = inst.instantiate(region)
    assert token.categoria == "relacional"
    assert token.aridade == 2


def test_operador_calculo_metadata() -> None:
    inst = OperadorFormalInstantiator()
    region = Region(TipoNome.OPERADOR_FORMAL, 0, 1, "∫")
    token = inst.instantiate(region)
    assert token.categoria == "calculo"
    assert token.aridade == 1


def test_operador_rejeita_tipo_errado() -> None:
    inst = OperadorFormalInstantiator()
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
    with pytest.raises(TypeError, match="OperadorFormal"):
        inst.instantiate(region)


# ---------------------------------------------------------------------------
# ConstanteUniversal
# ---------------------------------------------------------------------------


@pytest.fixture
def constante() -> ConstanteUniversalInstantiator:
    return ConstanteUniversalInstantiator()


def test_constante_pi_modo_b(constante: ConstanteUniversalInstantiator) -> None:
    region = Region(TipoNome.CONSTANTE_UNIVERSAL, 0, 1, "π")
    token = constante.instantiate(region, mode="B")
    # Modo B atômico: [CONST:simbolo] sem value/unit (decisão ontológica:
    # LLM já conhece valor de constantes consagradas; verbose induz erro
    # físico em letras homônimas).
    assert token.text == "[CONST:π]"


def test_constante_k_b_modo_b(constante: ConstanteUniversalInstantiator) -> None:
    region = Region(TipoNome.CONSTANTE_UNIVERSAL, 0, 3, "k_B")
    token = constante.instantiate(region, mode="B")
    assert token.text == "[CONST:k_B]"


def test_constante_simbolo_multi_char(constante: ConstanteUniversalInstantiator) -> None:
    """Constantes auto-capturadas são multi-char não-ambíguas (k_B, N_A, R_g, σ_SB, ℏ, π)."""
    region = Region(TipoNome.CONSTANTE_UNIVERSAL, 0, 3, "R_g")
    token = constante.instantiate(region, mode="B")
    assert token.text == "[CONST:R_g]"


def test_constante_modo_a_trinitario(
    constante: ConstanteUniversalInstantiator,
) -> None:
    region = Region(TipoNome.CONSTANTE_UNIVERSAL, 0, 3, "k_B")
    token = constante.instantiate(region, mode="A")
    expected = (
        "<CONST><SYM>k_B</SYM>"
        "<QTY><VAL>1.380649e-23</VAL><DIM>J/K</DIM><UNC/></QTY>"
        "</CONST>"
    )
    assert token.text == expected


def test_constante_desconhecida_fallback(
    constante: ConstanteUniversalInstantiator,
) -> None:
    region = Region(TipoNome.CONSTANTE_UNIVERSAL, 0, 1, "ψ")
    token = constante.instantiate(region, mode="B")
    assert token.text == "[CONST:ψ]"
    assert token.value is None
    assert token.unit is None


def test_constante_rejeita_tipo_errado(
    constante: ConstanteUniversalInstantiator,
) -> None:
    region = Region(TipoNome.OPERADOR_FORMAL, 0, 1, "=")
    with pytest.raises(TypeError, match="ConstanteUniversal"):
        constante.instantiate(region)


# ---------------------------------------------------------------------------
# RelacaoEstrutural
# ---------------------------------------------------------------------------


@pytest.fixture
def relacao() -> RelacaoEstruturalInstantiator:
    return RelacaoEstruturalInstantiator()


@pytest.mark.parametrize("conector", ["portanto", "logo", "então", "therefore", "thus"])
def test_relacao_modo_b_passthrough(
    relacao: RelacaoEstruturalInstantiator, conector: str
) -> None:
    region = Region(
        TipoNome.RELACAO_ESTRUTURAL, 0, len(conector), conector
    )
    assert relacao.instantiate_text(region.content) == conector


def test_relacao_carrega_funcao_logica(
    relacao: RelacaoEstruturalInstantiator,
) -> None:
    region = Region(TipoNome.RELACAO_ESTRUTURAL, 0, 8, "portanto")
    token = relacao.instantiate(region)
    assert token.funcao_logica == "conclusiva"
    assert token.idioma == "pt-br"


def test_relacao_idioma_en(relacao: RelacaoEstruturalInstantiator) -> None:
    region = Region(TipoNome.RELACAO_ESTRUTURAL, 0, 9, "therefore")
    token = relacao.instantiate(region)
    assert token.idioma == "en"
    assert token.funcao_logica == "conclusiva"


def test_relacao_multi_token(relacao: RelacaoEstruturalInstantiator) -> None:
    region = Region(
        TipoNome.RELACAO_ESTRUTURAL, 0, 15, "se e somente se"
    )
    token = relacao.instantiate(region)
    assert token.funcao_logica == "condicional"


def test_relacao_rejeita_tipo_errado(
    relacao: RelacaoEstruturalInstantiator,
) -> None:
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
    with pytest.raises(TypeError, match="RelacaoEstrutural"):
        relacao.instantiate(region)
