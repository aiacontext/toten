"""Testes do tipo Referencia (Evolução 8 — OEE v1.2 §9).

Cobre:
- Captura quando antecedente é enumerador normativo (parágrafo, §, art., ...)
- Captura quando antecedente é enumerador hierárquico-capable (item, capítulo, ...)
- Rejeição quando sem antecedente (decimal, versão de software)
- Preservação literal da hierarquia
"""

from __future__ import annotations

import pytest

from toten import Tokenizer
from toten.instantiators.reference import (
    Referencia,
    parse_referencia,
)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_ontology("oee-v1")


class TestParseReferencia:
    """Unit do parser."""

    def test_hierarquia_simples(self) -> None:
        ref = parse_referencia("8.2.1")
        assert ref.hierarquia == "8.2.1"

    def test_hierarquia_profunda(self) -> None:
        ref = parse_referencia("1.2.3.4.5")
        assert ref.hierarquia == "1.2.3.4.5"

    def test_dois_niveis(self) -> None:
        ref = parse_referencia("14.2")
        assert ref.hierarquia == "14.2"

    def test_vazio_rejeitado(self) -> None:
        with pytest.raises(ValueError):
            parse_referencia("")

    def test_invariante_construcao(self) -> None:
        with pytest.raises(ValueError):
            Referencia(hierarquia="   ")


class TestClassifierComEnumeradorNormativo:
    """Captura REF quando antecedente é normativo."""

    def test_paragrafo_simbolo(self, tok: Tokenizer) -> None:
        out = tok.preprocess("§ 8.2.1 da norma")
        assert "[REF:8.2.1]" in out

    def test_paragrafo_palavra(self, tok: Tokenizer) -> None:
        out = tok.preprocess("parágrafo 14.2.3 do regulamento")
        assert "[REF:14.2.3]" in out

    def test_artigo(self, tok: Tokenizer) -> None:
        out = tok.preprocess("art. 5.3.1.2 da CF")
        assert "[REF:5.3.1.2]" in out

    def test_inciso(self, tok: Tokenizer) -> None:
        out = tok.preprocess("inciso 1.2 da lei")
        assert "[REF:1.2]" in out

    def test_referencia_apos_idx(self, tok: Tokenizer) -> None:
        # `NBR 6118` (IDX) + `§ 14.2.3` (REF) coexistem
        out = tok.preprocess("NBR 6118 § 14.2.3")
        assert "[IDX:nbr-6118]" in out
        assert "[REF:14.2.3]" in out


class TestClassifierComEnumeradorHierarquico:
    """Captura REF para `item`, `capítulo`, etc. quando padrão é hierárquico."""

    def test_item_hierarquico(self, tok: Tokenizer) -> None:
        out = tok.preprocess("item 4.3.1 do manual")
        assert "[REF:4.3.1]" in out

    def test_capitulo_hierarquico(self, tok: Tokenizer) -> None:
        out = tok.preprocess("capítulo 2.1 da apostila")
        assert "[REF:2.1]" in out

    def test_anexo_hierarquico(self, tok: Tokenizer) -> None:
        out = tok.preprocess("anexo 5.2 do edital")
        assert "[REF:5.2]" in out


class TestRejeitaSemEnumerador:
    """SEM antecedente normativo/hierárquico — padrão fica como NUM."""

    def test_decimal_puro_nao_e_ref(self, tok: Tokenizer) -> None:
        out = tok.preprocess("3.14 não é referência")
        assert "[REF:" not in out
        assert "[NUM value=3.14" in out

    def test_versao_software_nao_e_ref(self, tok: Tokenizer) -> None:
        # `versão 1.2.3` — `versão` não está nos enumeradores de
        # referência (semanticamente é descrição de produto, não
        # subdivisão estrutural)
        out = tok.preprocess("versão 1.2.3 do software")
        assert "[REF:" not in out


class TestPreservacaoLiteral:
    """Hierarquia preservada exatamente como escrita."""

    def test_zeros_a_esquerda_preservados(self, tok: Tokenizer) -> None:
        out = tok.preprocess("§ 08.02.01 da norma")
        assert "[REF:08.02.01]" in out

    def test_dois_digitos(self, tok: Tokenizer) -> None:
        out = tok.preprocess("art. 12.45.67 da CF")
        assert "[REF:12.45.67]" in out
