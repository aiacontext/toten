"""Testes do NumeroInstantiator (Evolução 3 v0.7 — 8º tipo OEE).

Cobre `toten.instantiators.numero` — parse locale-aware
(pt-br / en / none), detecção de representação (decimal / científica /
inteira / fracionária / percentual), invariante sinal-vs-valor, render
Modo B com r2l opcional, render Modo A canônico.
"""

from __future__ import annotations

import pytest

from toten.classifier.region import Region
from toten.instantiators.numero import (
    Numero,
    NumeroInstantiator,
    NumeroToken,
    parse_numero,
)
from toten.ontology.types import TipoNome


class TestParseNumeroInteiro:
    """Inteiros puros — locale=none, repr=inteira."""

    def test_inteiro_positivo(self) -> None:
        n = parse_numero("100")
        assert n.valor == 100.0
        assert n.locale == "none"
        assert n.representacao == "inteira"
        assert n.sinal == "positivo"
        assert n.original == "100"

    def test_inteiro_negativo_ascii(self) -> None:
        n = parse_numero("-42")
        assert n.valor == -42.0
        assert n.sinal == "negativo"

    def test_inteiro_negativo_unicode(self) -> None:
        n = parse_numero("−273")
        assert n.valor == -273.0
        assert n.sinal == "negativo"
        # Original preserva o unicode minus
        assert n.original == "−273"

    def test_inteiro_zero(self) -> None:
        n = parse_numero("0")
        assert n.valor == 0.0
        assert n.sinal == "zero"


class TestParseNumeroDecimal:
    """Decimais com locale detection."""

    def test_decimal_pt_br(self) -> None:
        n = parse_numero("0,5")
        assert n.valor == 0.5
        assert n.locale == "pt-br"
        assert n.representacao == "decimal"

    def test_decimal_en_explicito(self) -> None:
        n = parse_numero("0.5")
        assert n.valor == 0.5
        assert n.locale == "en"

    def test_milhar_pt_br_simples(self) -> None:
        # PT-BR: 1.234 = 1234 (milhar) — primeiro dígito ≠ 0
        n = parse_numero("1.234")
        assert n.valor == 1234.0
        assert n.locale == "pt-br"

    def test_milhar_pt_br_multiplos(self) -> None:
        n = parse_numero("1.234.567")
        assert n.valor == 1234567.0
        assert n.locale == "pt-br"

    def test_zero_dot_nao_e_milhar_pt_br(self) -> None:
        # ESTRUTURAL: notação PT-BR de milhar não começa com 0.
        # "0.500" → decimal EN (0.5), não milhar PT-BR (500).
        n = parse_numero("0.500")
        assert n.valor == 0.5
        assert n.locale == "en"

    def test_misto_pt_br_milhar_decimal(self) -> None:
        # 1.234,56 → PT-BR (milhar . + decimal ,)
        n = parse_numero("1.234,56")
        assert n.valor == 1234.56
        assert n.locale == "pt-br"

    def test_misto_en_milhar_decimal(self) -> None:
        # 1,234.56 → EN (milhar , + decimal .)
        n = parse_numero("1,234.56")
        assert n.valor == 1234.56
        assert n.locale == "en"

    def test_negativo_decimal_pt_br(self) -> None:
        n = parse_numero("−273,15")
        assert n.valor == -273.15
        assert n.sinal == "negativo"


class TestParseNumeroFracao:
    """Frações simples — divisão direta, sem álgebra simbólica."""

    def test_fracao_basica(self) -> None:
        n = parse_numero("3/4")
        assert n.valor == 0.75
        assert n.representacao == "fracionaria"

    def test_fracao_um_meio(self) -> None:
        n = parse_numero("1/2")
        assert n.valor == 0.5

    def test_fracao_impropria(self) -> None:
        n = parse_numero("22/7")
        assert n.valor == pytest.approx(22 / 7)

    def test_fracao_divisao_por_zero(self) -> None:
        with pytest.raises(ValueError, match="divisão por zero"):
            parse_numero("3/0")


class TestParseNumeroOrdinal:
    """Ordinais PT-BR — marcadores UCD `ª`/`º` (ORDINAL INDICATOR).

    `°` (DEGREE SIGN) explicitamente NÃO é ordinal — pertence a domínio
    distinto (grau angular = QTY).
    """

    def test_ordinal_feminino_simples(self) -> None:
        n = parse_numero("1ª")
        assert n.valor == 1.0
        assert n.representacao == "ordinal"
        assert n.original == "1ª"

    def test_ordinal_masculino_simples(self) -> None:
        n = parse_numero("2º")
        assert n.valor == 2.0
        assert n.representacao == "ordinal"

    def test_ordinal_multi_digito(self) -> None:
        n = parse_numero("100ª")
        assert n.valor == 100.0
        assert n.representacao == "ordinal"

    def test_ordinal_grande(self) -> None:
        n = parse_numero("21º")
        assert n.valor == 21.0
        assert n.representacao == "ordinal"

    def test_ordinal_preserva_marcador_original(self) -> None:
        # `original` preserva o ª/º exatamente como escrito; `value`
        # extrai apenas o componente numérico
        n = parse_numero("3ª")
        assert n.original == "3ª"
        assert n.valor == 3.0


class TestParseNumeroPercentual:
    """Percentuais — convenção X% = X/100."""

    def test_percentual_inteiro(self) -> None:
        n = parse_numero("50%")
        assert n.valor == 0.5
        assert n.representacao == "percentual"

    def test_percentual_decimal_pt_br(self) -> None:
        n = parse_numero("0,5%")
        assert n.valor == 0.005

    def test_percentual_zero(self) -> None:
        n = parse_numero("0%")
        assert n.valor == 0.0
        assert n.sinal == "zero"


class TestParseNumeroCientifica:
    """Notação científica Unicode (× 10⁻²³) e ASCII (1.5e-3)."""

    def test_cientifica_unicode(self) -> None:
        n = parse_numero("6,022 × 10²³")
        assert n.valor == pytest.approx(6.022e23, rel=1e-9)
        assert n.representacao == "cientifica"

    def test_cientifica_unicode_negativa(self) -> None:
        n = parse_numero("1,38 × 10⁻²³")
        assert n.valor == pytest.approx(1.38e-23, rel=1e-9)

    def test_cientifica_ascii(self) -> None:
        n = parse_numero("1.5e-3")
        assert n.valor == pytest.approx(1.5e-3)


class TestNumeroInvariantes:
    """Invariantes da OEE v1.1."""

    def test_original_vazio_rejeitado(self) -> None:
        with pytest.raises(ValueError):
            Numero(
                valor=0.0,
                locale="none",
                representacao="inteira",
                sinal="zero",
                original="",
            )

    def test_sinal_inferido_consistente(self) -> None:
        n = parse_numero("100")
        assert n.sinal_inferido == n.sinal


class TestNumeroTokenRenderB:
    """Modo B — tag rica `[NUM ...]`."""

    def test_render_b_inteiro_pequeno_sem_r2l(self) -> None:
        n = parse_numero("100")
        tok = NumeroToken(numero=n, mode="B")
        assert tok.text == '[NUM value=100 locale=none repr=inteira original="100"]'

    def test_render_b_inteiro_grande_com_r2l(self) -> None:
        n = parse_numero("1234567")
        tok = NumeroToken(numero=n, mode="B")
        assert (
            tok.text
            == '[NUM value=1234567 r2l="1 234 567" locale=none repr=inteira original="1234567"]'
        )

    def test_render_b_decimal_pt_br(self) -> None:
        n = parse_numero("0,5")
        tok = NumeroToken(numero=n, mode="B")
        assert tok.text == '[NUM value=0.5 locale=pt-br repr=decimal original="0,5"]'

    def test_render_b_percentual(self) -> None:
        n = parse_numero("50%")
        tok = NumeroToken(numero=n, mode="B")
        assert (
            tok.text
            == '[NUM value=0.5 locale=none repr=percentual original="50%"]'
        )

    def test_render_b_fracionaria(self) -> None:
        n = parse_numero("3/4")
        tok = NumeroToken(numero=n, mode="B")
        assert (
            tok.text
            == '[NUM value=0.75 locale=none repr=fracionaria original="3/4"]'
        )

    def test_render_b_milhar_pt_br_com_r2l(self) -> None:
        n = parse_numero("1.234.567")
        tok = NumeroToken(numero=n, mode="B")
        assert 'r2l="1 234 567"' in tok.text
        assert 'value=1234567' in tok.text
        assert 'locale=pt-br' in tok.text


class TestNumeroTokenRenderA:
    """Modo A — formato canônico compacto, idêntico em formatação ao Modo B."""

    def test_render_a_inteiro(self) -> None:
        n = parse_numero("100")
        tok = NumeroToken(numero=n, mode="A")
        assert tok.text == "<NUM>100</NUM>"

    def test_render_a_decimal(self) -> None:
        n = parse_numero("0,5")
        tok = NumeroToken(numero=n, mode="A")
        assert tok.text == "<NUM>0.5</NUM>"

    def test_render_a_grande_consistente_modo_b(self) -> None:
        # Modo A usa o MESMO `_format_value` que Modo B — inteiros até
        # 1e15 ficam compactos, sem científica.
        n = parse_numero("1234567")
        tok = NumeroToken(numero=n, mode="A")
        assert tok.text == "<NUM>1234567</NUM>"


class TestNumeroInstantiator:
    """Camada 3 — instanciador roteado pelo pipeline."""

    def test_instantiator_aceita_regiao_numero(self) -> None:
        region = Region(TipoNome.NUMERO, 0, 3, "100")
        token = NumeroInstantiator().instantiate(region, mode="B")
        assert token.text.startswith("[NUM ")

    def test_instantiator_rejeita_tipo_errado(self) -> None:
        region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
        with pytest.raises(TypeError, match="Numero"):
            NumeroInstantiator().instantiate(region, mode="B")

    def test_instantiate_text_atalho(self) -> None:
        result = NumeroInstantiator().instantiate_text("100", mode="B")
        assert result == '[NUM value=100 locale=none repr=inteira original="100"]'
