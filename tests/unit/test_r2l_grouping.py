"""Testes do helper R2L grouping (Evolução 4 v0.7).

Cobre `toten.instantiators._r2l.r2l_group` — endereça
SOTA Number Cookbook (Yang ICLR 2025) + Singh-Strouse 2024. Testes
verificam contrato: agrupamento right-to-left por 3 dígitos com
separador espaço, omissão quando agrupamento é visualmente vazio.
"""

from __future__ import annotations

import pytest

from toten.instantiators._r2l import r2l_group


class TestR2LBasicGrouping:
    """Agrupamento básico para inteiros e decimais."""

    def test_inteiro_4_digitos(self) -> None:
        assert r2l_group(1234) == "1 234"

    def test_inteiro_7_digitos(self) -> None:
        assert r2l_group(1234567) == "1 234 567"

    def test_inteiro_10_digitos(self) -> None:
        assert r2l_group(1234567890) == "1 234 567 890"

    def test_decimal_inteiro_grande(self) -> None:
        assert r2l_group(1234.56) == "1 234.56"

    def test_decimal_alta_precisao(self) -> None:
        assert r2l_group(12345.67890123) == "12 345.67890123"

    def test_decimal_inteiro_muito_grande(self) -> None:
        assert r2l_group(1234567890.123) == "1 234 567 890.123"


class TestR2LSinal:
    """Preservação de sinal."""

    def test_negativo_inteiro(self) -> None:
        assert r2l_group(-1234567) == "-1 234 567"

    def test_negativo_decimal(self) -> None:
        assert r2l_group(-12345.6) == "-12 345.6"

    def test_zero_sem_agrupamento(self) -> None:
        assert r2l_group(0) is None

    def test_negativo_pequeno_sem_agrupamento(self) -> None:
        assert r2l_group(-999) is None


class TestR2LSemAgrupamento:
    """Casos onde agrupamento seria visualmente idêntico — retorna None."""

    def test_inteiro_3_digitos(self) -> None:
        assert r2l_group(100) is None

    def test_inteiro_borda_999(self) -> None:
        assert r2l_group(999) is None

    def test_decimal_subunitario(self) -> None:
        assert r2l_group(0.5) is None

    def test_decimal_milesimal(self) -> None:
        # parte inteira = 0, sem agrupamento
        assert r2l_group(0.0015) is None

    def test_inteiro_1_digito(self) -> None:
        assert r2l_group(7) is None


class TestR2LCientificaRejeitada:
    """Notação científica não se beneficia de R2L na mantissa."""

    def test_cientifica_pequena(self) -> None:
        assert r2l_group(1.5e-3) is None

    def test_cientifica_muito_pequena(self) -> None:
        assert r2l_group(1.38e-23) is None

    def test_cientifica_muito_grande(self) -> None:
        assert r2l_group(1e15) is None

    def test_cientifica_avogadro(self) -> None:
        assert r2l_group(6.022e23) is None


class TestR2LBordaInteiroDecimal:
    """Boundary: |value| = 1000 entra; |value| = 999 não."""

    def test_borda_inferior_1000(self) -> None:
        assert r2l_group(1000) == "1 000"

    def test_borda_inferior_negativa(self) -> None:
        assert r2l_group(-1000) == "-1 000"

    def test_borda_999_99(self) -> None:
        # parte inteira = 999, sem agrupamento (só 3 dígitos)
        assert r2l_group(999.99) is None

    def test_borda_1000_99(self) -> None:
        assert r2l_group(1000.99) == "1 000.99"


class TestR2LInputs:
    """Tipos de entrada aceitos."""

    def test_input_none(self) -> None:
        assert r2l_group(None) is None  # type: ignore[arg-type]

    def test_input_int_python(self) -> None:
        assert r2l_group(1234) == "1 234"

    def test_input_float_inteiro(self) -> None:
        assert r2l_group(1234.0) == "1 234"

    @pytest.mark.parametrize(
        "v,expected",
        [
            (10000, "10 000"),
            (100000, "100 000"),
            (1000000, "1 000 000"),
            (-100000, "-100 000"),
        ],
    )
    def test_diversos_inteiros(self, v: int, expected: str) -> None:
        assert r2l_group(v) == expected
