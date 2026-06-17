"""Testes das âncoras técnicas que suprimem ambig (Evolução 8).

Cobre:
- Palavras-âncora da dimensão da unidade na janela próxima suprimem
  `ambig="unit-letter"`
- Sem âncora, ambig permanece
- Lista por letra (A, V, K, W, N, J, T, H, F, C, S)
"""

from __future__ import annotations

import pytest

from toten import Tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_ontology("oee-v1")


class TestAncoraTemperaturaK:
    """K (kelvin) — âncoras: temperatura, frio, quente, térmico."""

    def test_temperatura_antes(self, tok: Tokenizer) -> None:
        out = tok.preprocess("A temperatura é 300 K")
        assert "[QTY value=300 unit=K" in out
        assert "ambig=" not in out

    def test_frio_distante_mesmo_ancora(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Em Plutão é muito frio, com temperatura igual a 40 K")
        assert "ambig=" not in out

    def test_temperatura_depois(self, tok: Tokenizer) -> None:
        out = tok.preprocess("mede 40 K de temperatura")
        assert "ambig=" not in out

    def test_sem_ancora_temperatura_mantem_ambig(
        self, tok: Tokenizer
    ) -> None:
        out = tok.preprocess("a janela tem 5 K de altura")
        # K aqui não combina semanticamente; sem âncora → ambig
        assert 'ambig="unit-letter"' in out


class TestAncoraCorrenteA:
    def test_corrente(self, tok: Tokenizer) -> None:
        out = tok.preprocess("corrente de 5 A no fio")
        assert "[QTY value=5 unit=A" in out
        assert "ambig=" not in out

    def test_intensidade(self, tok: Tokenizer) -> None:
        out = tok.preprocess("intensidade 10 A na bobina")
        assert "ambig=" not in out


class TestAncoraTensaoV:
    def test_tensao(self, tok: Tokenizer) -> None:
        out = tok.preprocess("tensão de 220 V na tomada")
        assert "ambig=" not in out

    def test_voltagem(self, tok: Tokenizer) -> None:
        out = tok.preprocess("voltagem 12 V da bateria")
        assert "ambig=" not in out


class TestAncoraPotenciaW:
    def test_potencia(self, tok: Tokenizer) -> None:
        out = tok.preprocess("potência de 100 W consumida")
        assert "ambig=" not in out

    def test_dissipacao(self, tok: Tokenizer) -> None:
        out = tok.preprocess("dissipação 5 W no resistor")
        assert "ambig=" not in out


class TestAncoraForcaN:
    def test_forca(self, tok: Tokenizer) -> None:
        out = tok.preprocess("força de 50 N aplicada")
        assert "ambig=" not in out


class TestAncoraEnergiaJ:
    def test_energia(self, tok: Tokenizer) -> None:
        out = tok.preprocess("energia de 500 J liberada")
        assert "ambig=" not in out


class TestStemmerCobrirFlexoes:
    """Stemmer RSLP cobre flexões PT-BR automaticamente."""

    def test_fria_flexao_feminino(self, tok: Tokenizer) -> None:
        # `fria` (feminino de frio) — RSLP reduz ambos a `fri`
        out = tok.preprocess(
            "superfície extremamente fria de Plutão (~40 K) é composta de gases"
        )
        assert "[QTY value=40 unit=K" in out
        assert "ambig=" not in out

    def test_geladas_flexao_plural_feminino(self, tok: Tokenizer) -> None:
        out = tok.preprocess("regiões geladas atingem 200 K na noite")
        assert "ambig=" not in out

    def test_termicas_flexao_plural_feminino(self, tok: Tokenizer) -> None:
        out = tok.preprocess("propriedades térmicas a 300 K medidas")
        assert "ambig=" not in out

    def test_quentes_flexao_plural(self, tok: Tokenizer) -> None:
        out = tok.preprocess("zonas quentes do reator atingem 800 K")
        assert "ambig=" not in out

    def test_potencias_flexao_plural(self, tok: Tokenizer) -> None:
        out = tok.preprocess("as potências dos motores são 100 W cada")
        assert "ambig=" not in out

    def test_forcas_flexao_plural(self, tok: Tokenizer) -> None:
        out = tok.preprocess("as forças aplicadas valem 50 N")
        assert "ambig=" not in out


class TestStemmerNaoColisaoFalsosHomofonos:
    """Stemmer distingue palavras homofônicas — `frita`, `frigorífico`
    NÃO devem casar como âncora de K (mesmo começando com `fri`)."""

    def test_frita_nao_e_ancora_de_k(self, tok: Tokenizer) -> None:
        # `frita` → stem `frit`, distinto de `fri` (frio)
        out = tok.preprocess("batata frita ao lado de 5 K de algo")
        assert 'ambig="unit-letter"' in out

    def test_frigorifico_nao_e_ancora(self, tok: Tokenizer) -> None:
        # `frigorífico` → stem `frigoríf`, distinto de `fri`
        out = tok.preprocess("frigorífico armazena alimentos a 5 K")
        assert 'ambig="unit-letter"' in out


class TestSemAncoraMantemAmbig:
    """Sem âncora técnica próxima, ambig permanece (default)."""

    def test_consome_5_w_sem_ancora_mantem_ambig(
        self, tok: Tokenizer
    ) -> None:
        # `consome` não é âncora da dimensão potência (não está na lista)
        out = tok.preprocess("consome 5 W")
        assert 'ambig="unit-letter"' in out

    def test_resposta_2_a_isolada_mantem_ambig(
        self, tok: Tokenizer
    ) -> None:
        out = tok.preprocess("Resposta: 2 A.")
        assert 'ambig="unit-letter"' in out

    def test_palavra_aleatoria_nao_e_ancora(
        self, tok: Tokenizer
    ) -> None:
        out = tok.preprocess("relatório com 5 W escrito")
        assert 'ambig="unit-letter"' in out
