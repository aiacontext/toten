"""Testes da dimensão `unit-letter` (SPEC_07 §2.1) — Evolução 7.

Cobre:
- Rejeição direta quando antecedente é enumerador PT-BR
- Marcação `ambig="unit-letter"` + `alternatives` em casos default
- Não-afetação de unidades Unicode/compostas/multi-char
"""

from __future__ import annotations

import pytest

from toten import Tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_ontology("oee-v1")


class TestEnumeratorRejeicao:
    """SPEC_07 §2.1 — antecedente enumerador rejeita QTY direto."""

    def test_topico_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Tópico 2 A menina caminhou")
        assert "[QTY" not in out
        assert "[NUM value=2" in out
        assert "A menina" in out

    def test_capitulo_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Capítulo 4 C exercícios resolvidos")
        assert "[QTY" not in out
        assert "[NUM value=4" in out

    def test_item_rejeita(self, tok: Tokenizer) -> None:
        # B não é unidade SI, então não tem QTY mesmo; verifica que
        # número fica como NUM (não fragmenta com falso QTY)
        out = tok.preprocess("Item 3 B representa a opção")
        assert "[NUM value=3" in out

    def test_questao_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Questão 5 N é difícil")
        assert "[QTY value=5 unit=N" not in out
        assert "[NUM value=5" in out

    def test_secao_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Seção 2 V analisa")
        assert "[QTY value=2 unit=V" not in out

    def test_figura_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Figura 1 K mostra o ciclo")
        assert "[QTY value=1 unit=K" not in out

    def test_enumerator_lowercase_tambem_rejeita(
        self, tok: Tokenizer
    ) -> None:
        # `topico` (sem acento) também é reconhecido como enumerador
        out = tok.preprocess("topico 2 A menina")
        assert "[QTY value=2 unit=A" not in out


class TestAmbigMarcador:
    """SPEC_07 §2.1 — caso default: emite QTY com `ambig="unit-letter"`."""

    def test_5_w_emite_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("O motor de 5 W liga rápido")
        assert "[QTY value=5 unit=W" in out
        assert 'ambig="unit-letter"' in out
        assert "watts" in out
        assert "alternatives=" in out

    def test_2_a_de_corrente_sem_ambig(
        self, tok: Tokenizer
    ) -> None:
        # Evolução 8: âncora técnica "corrente" (associada a A=ampere)
        # confirma uso técnico inequívoco e suprime ambig.
        out = tok.preprocess("O circuito consome 2 A de corrente")
        assert "[QTY value=2 unit=A" in out
        assert "ambig=" not in out

    def test_alternatives_no_formato_chave_descricao(
        self, tok: Tokenizer
    ) -> None:
        out = tok.preprocess("a fonte fornece 5 V")
        assert 'alternatives="unit:volts' in out
        assert "|reference:letra_V" in out

    def test_5_n_emite_ambig(self, tok: Tokenizer) -> None:
        # N pode ser newton (física) ou normal (química, futuramente
        # via domain-semantic). Marcador unit-letter cobre por enquanto.
        out = tok.preprocess("Solução 5 N produz reação")
        assert "[QTY value=5 unit=N" in out
        assert 'ambig="unit-letter"' in out

    def test_isolada_em_fim_de_frase_emite_ambig(
        self, tok: Tokenizer
    ) -> None:
        out = tok.preprocess("Resposta: 2 A.")
        assert "[QTY value=2 unit=A" in out
        # Mesmo em fim, marca ambig — agente decide se pergunta
        assert 'ambig="unit-letter"' in out


class TestNaoAfetadasPorAmbig:
    """Unidades fora do conjunto ambíguo NÃO recebem marcador."""

    def test_unicode_omega_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Resistor de 10 Ω limita")
        assert "[QTY value=10 unit=Ω" in out
        assert "ambig=" not in out

    def test_unicode_celsius_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("20 °C de temperatura")
        assert "[QTY value=20 unit=°C" in out
        assert "ambig=" not in out

    def test_multi_char_mpa_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("tensão 350 MPa")
        assert "[QTY value=350 unit=MPa" in out
        assert "ambig=" not in out

    def test_multi_char_kn_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("carga 12500 kN")
        assert "[QTY value=12500" in out
        assert "ambig=" not in out

    def test_composta_kg_m3_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("densidade 7850 kg/m³")
        assert "[QTY value=7850" in out
        assert "ambig=" not in out

    def test_lowercase_unit_sem_ambig(self, tok: Tokenizer) -> None:
        # `m` (metro) é 1-char ASCII mas LOWERCASE — não está no
        # conjunto ambíguo {A,V,K,W,N,J,T,H,F,C,S} (que são MAIÚSC).
        out = tok.preprocess("comprimento de 5 m")
        assert "[QTY value=5 unit=m" in out
        assert "ambig=" not in out


class TestEnumeradorNormativo:
    """SPEC_07 §2.1 — enumeradores normativos (parágrafo, §, art., …)
    rejeitam QTY de QUALQUER unidade. Textos legais não medem; apenas
    referenciam estrutura."""

    def test_paragrafo_rejeita_grau_angular(self, tok: Tokenizer) -> None:
        # `§ 5°` é ordinal mal-formado (deveria ser `5º`), NÃO grau angular
        out = tok.preprocess("Conforme § 5°")
        assert "[QTY" not in out
        assert "[NUM value=5" in out

    def test_artigo_rejeita_grau(self, tok: Tokenizer) -> None:
        out = tok.preprocess("artigo 5° da CF")
        assert "[QTY" not in out

    def test_paragrafo_simbolo_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("§ 3 do regulamento")
        assert "[QTY" not in out
        assert "[NUM value=3" in out

    def test_inciso_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("inciso 3 da lei")
        assert "[QTY" not in out
        assert "[NUM value=3" in out

    def test_paragrafo_palavra_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("O parágrafo 5 da norma")
        assert "[QTY" not in out
        assert "[NUM value=5" in out

    def test_alinea_rejeita(self, tok: Tokenizer) -> None:
        out = tok.preprocess("alínea 2 do art.")
        assert "[QTY" not in out
        assert "[NUM value=2" in out


class TestSymEquationSuprimeAmbig:
    """SPEC_07 §2.1 — equação `<sym> = <num> <unit>` antes suprime
    `ambig` (contexto matemático confirma unit inequivocamente)."""

    def test_p_igual_5_w_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("P = 5 W")
        assert "[QTY value=5 unit=W" in out
        assert "ambig=" not in out
        assert "[SYM:P]" in out

    def test_sigma_y_igual_350_n_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("σ_y = 350 N")
        assert "[QTY value=350 unit=N" in out
        assert "ambig=" not in out

    def test_t_igual_5_k_sem_ambig(self, tok: Tokenizer) -> None:
        out = tok.preprocess("T = 5 K")
        assert "[QTY value=5 unit=K" in out
        assert "ambig=" not in out

    def test_consome_5_w_mantém_ambig(self, tok: Tokenizer) -> None:
        # Sem SYM antes: ambig continua emitido (sinal único negativo
        # não basta para confirmar unit)
        out = tok.preprocess("consome 5 W")
        assert 'ambig="unit-letter"' in out

    def test_relacao_desigualdade_tambem_suprime(self, tok: Tokenizer) -> None:
        out = tok.preprocess("P ≤ 100 W")
        assert "[QTY value=100 unit=W" in out
        assert "ambig=" not in out


class TestRegressaoEnumeradorNaoAfetaUnidadeForaDoConjunto:
    """Enumerador antes NÃO afeta unidades fora do conjunto ambíguo."""

    def test_topico_kg_mantem_qty(self, tok: Tokenizer) -> None:
        # `kg` é 2-char, não está no conjunto ambíguo — preservado
        out = tok.preprocess("Tópico 5 kg de massa")
        assert "[QTY value=5 unit=kg" in out

    def test_capitulo_mpa_mantem_qty(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Capítulo 350 MPa de tensão")
        assert "[QTY value=350 unit=MPa" in out

    def test_secao_celsius_mantem_qty(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Seção 20 °C de operação")
        assert "[QTY value=20 unit=°C" in out
