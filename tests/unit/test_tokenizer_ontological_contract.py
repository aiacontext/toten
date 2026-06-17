"""Testes do contrato ontológico do tokenizer (Camada 1 + Camada 2).

Cada teste codifica uma decisão ontológica formal do framework TOTEN.
Falhas aqui = violação do contrato.

Categorias:
    - NEGATIVOS: padrões que NÃO devem ser capturados como SYM (bugs
      descobertos em 28/05/2026 + casos derivados).
    - POSITIVOS: padrões que DEVEM ser capturados como SYM (validação
      de cobertura).
    - ATOMICIDADE: tags renderizadas devem ser entidades limpas, sem
      whitespace engolido nem fronteiras grudadas.

A marca `pytest.mark.xfail(strict=False)` indica testes que falham no
tokenizer pré-refatoração (Camadas indistintas) e devem passar após.
Remover a marca à medida que cada bug é corrigido.
"""

from __future__ import annotations

import pytest

from toten import Tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer()


# ===========================================================================
# NEGATIVOS — padrões que NÃO devem virar SYM
# ===========================================================================


class TestBugsConhecidos:
    """5 bugs catalogados em 28/05/2026. Cada um deve ser eliminado."""

    def test_bug1_multiplicacao_geometrica_entre_qtys(self, tok: Tokenizer) -> None:
        """`b = 20 cm × h = 50 cm` — `×` é operador isolado, não parte de SYM."""
        out = tok.preprocess("Seção retangular b = 20 cm × h = 50 cm")
        # `× h` NÃO deve virar SYM. Deve ficar `× h` cru ou `[OP:×] h` se
        # operator emitir; nunca `[SYM:×h]`.
        assert "[SYM:×h]" not in out, (
            f"Operador × engolido como SYM: {out!r}"
        )
        assert "[SYM:×" not in out, (
            f"Qualquer SYM começando com × é erro: {out!r}"
        )

    def test_bug2_bullet_lista_nao_e_sinal_unario(self, tok: Tokenizer) -> None:
        """`\\n  - I_z = ...` — bullet de lista, não negação matemática."""
        texto = "Dados:\n  - L = 8 m\n  - I_z = 208300 cm⁴"
        out = tok.preprocess(texto)
        # `-I_z` NÃO deve virar SYM com hífen colado
        assert "[SYM:-" not in out, (
            f"Bullet capturado como sinal unário: {out!r}"
        )

    def test_bug2b_atomicidade_de_tags(self, tok: Tokenizer) -> None:
        """Bullet engole whitespace, fazendo tags adjacentes grudarem."""
        texto = "Dados:\n  - L = 8 m\n  - I_z = 208300 cm⁴"
        out = tok.preprocess(texto)
        # `][SYM` (sem espaço) é violação de atomicidade — tags devem ter
        # whitespace ou pontuação entre elas, preservando a estrutura
        # textual original.
        assert "]][SYM" not in out and "]][QTY" not in out, (
            f"Tags grudadas sem separador: {out!r}"
        )

    def test_bug3_aposicao_parentetica(self, tok: Tokenizer) -> None:
        """`§8.2.8 (E_cs)` — parens de aposição prosaica, não SYM."""
        out = tok.preprocess("conforme NBR 6118 §8.2.8 (E_cs)")
        # `(E_cs)` em contexto de aposição NÃO deve virar SYM.
        # E_cs sozinho PODE virar IDX/SYM atômico se reconhecido,
        # mas o cluster inteiro com parens é prosaico.
        assert "[SYM:(E_cs)]" not in out, (
            f"Aposição engolida como SYM: {out!r}"
        )

    def test_bug4_consistencia_funcao_aplicada(self, tok: Tokenizer) -> None:
        """`M(B)` e `M(C)` são equivalentes — tratamento deve ser idêntico."""
        out = tok.preprocess("M(B) = -202,3 kN·m em M(C) = -134,9 kN·m")
        # Ambos devem ter o mesmo tratamento. Conta ocorrências:
        n_sym_mb = out.count("[SYM:M(B)]")
        n_sym_mc = out.count("[SYM:M(C)]")
        assert n_sym_mb == n_sym_mc, (
            f"Tratamento inconsistente M(B)={n_sym_mb} vs M(C)={n_sym_mc}: {out!r}"
        )

    def test_bug5_secao_numerada_com_parens(self, tok: Tokenizer) -> None:
        """`§8.2.8)` — número de seção com parêntese de fechamento de aposição."""
        out = tok.preprocess("conforme (NBR 6118 §8.2.8)")
        # `§8.2.8)` NÃO deve virar SYM. Parêntese pertence à aposição,
        # não à expressão.
        assert "[SYM:8.2)" not in out and "[SYM:8.2.8)" not in out, (
            f"Número de seção com paren engolido como SYM: {out!r}"
        )


class TestNegativosDerivados:
    """Casos derivados dos princípios ontológicos. Verificam que a
    refatoração não introduz novos falsos positivos."""

    def test_multiplos_bullets_seguidos_em_lista(self, tok: Tokenizer) -> None:
        texto = "Lista:\n  - X = 1\n  - Y = 2\n  - Z = 3"
        out = tok.preprocess(texto)
        assert "[SYM:-" not in out, f"Bullet vira sinal: {out!r}"

    def test_aposicao_em_meio_de_prosa(self, tok: Tokenizer) -> None:
        out = tok.preprocess("o módulo (E_cs) determina a deflexão")
        assert "[SYM:(E_cs)]" not in out, f"Aposição engolida: {out!r}"

    def test_dimensao_geometrica_QTY_x_QTY(self, tok: Tokenizer) -> None:
        """`30 cm × 40 cm` — REGRA DE PUREZA QTY: cluster composto apenas
        de grandezas físicas com unidade NÃO emerge como SYM. Mantém-se
        `[QTY 30 cm] × [QTY 40 cm]` para o LLM extrair valores
        individuais. SYM é para composição GENUINAMENTE SIMBÓLICA (≥1
        IDX/CONST presente).
        """
        out = tok.preprocess("dimensões 30 cm × 40 cm")
        assert "[SYM:" not in out, (
            f"QTYs puras não devem virar cluster: {out!r}"
        )
        # Ambas QTYs devem estar capturadas como átomos
        assert "[QTY value=30" in out and "[QTY value=40" in out, (
            f"QTYs não capturadas individualmente: {out!r}"
        )

    def test_palavras_naturais_nao_viram_sym(self, tok: Tokenizer) -> None:
        texto = "apoios verticais e bullets dos itens"
        out = tok.preprocess(texto)
        assert "[SYM:" not in out, f"Prosa virou SYM: {out!r}"


# ===========================================================================
# POSITIVOS — padrões que DEVEM virar SYM (validação de cobertura)
# ===========================================================================


class TestPositivosFundamentais:
    """Padrões legítimos de SYM que devem continuar funcionando."""

    def test_soma_de_variaveis(self, tok: Tokenizer) -> None:
        """`σ_x + σ_y` — composição mediada por +."""
        out = tok.preprocess("a tensão σ_x + σ_y é constante")
        assert "[SYM:" in out, f"Composição não capturada: {out!r}"

    @pytest.mark.skip(
        reason=(
            "OBSOLETO: refator anterior — 'pl/12' em prosa "
            "(sem âncora estrutural antes) não vira SYM atômico. P7 da "
            "OEE exige âncora explícita (=, ≤, etc.). Reescrita pendente."
        )
    )
    def test_quociente_simbolico(self, tok: Tokenizer) -> None:
        """`pl/12` — composição mediada por /."""
        out = tok.preprocess("o momento é pl/12 na extremidade")
        assert "[SYM:" in out, f"Quociente não capturado: {out!r}"

    def test_derivada(self, tok: Tokenizer) -> None:
        """`dM/dx` — padrão estrutural inerentemente atômico."""
        out = tok.preprocess("a derivada dM/dx representa o cortante")
        assert "[SYM:" in out, f"Derivada não capturada: {out!r}"

    def test_funcao_log(self, tok: Tokenizer) -> None:
        """`log(x)` — função matemática nomeada."""
        out = tok.preprocess("aplique log(x) ao argumento")
        assert "[SYM:" in out, f"Função nomeada não capturada: {out!r}"

    def test_funcao_aplicada_generica_M_de_x(self, tok: Tokenizer) -> None:
        """`M(x)` — função aplicada com variável."""
        out = tok.preprocess("o momento M(x) varia com a posição x")
        assert "[SYM:M(x)]" in out or "[SYM:" in out, (
            f"Função aplicada não capturada: {out!r}"
        )

    def test_valor_absoluto(self, tok: Tokenizer) -> None:
        out = tok.preprocess("a magnitude |σ_cp| é positiva")
        assert "[SYM:" in out, f"Valor absoluto não capturado: {out!r}"

    def test_monomio_coeficiente_x_variavel(self, tok: Tokenizer) -> None:
        out = tok.preprocess("o coeficiente vale 3l e depois 5p")
        assert "[SYM:" in out, f"Monômio não capturado: {out!r}"


# ===========================================================================
# ATOMICIDADE — tags renderizadas devem ser entidades limpas
# ===========================================================================


class TestAtomicidade:
    """Tags emitidas têm fronteiras limpas — nem whitespace engolido,
    nem grudadas em vizinhas."""

    def test_whitespace_entre_tags_preservado(self, tok: Tokenizer) -> None:
        out = tok.preprocess("o valor de 20 cm e depois 50 mm são distintos")
        # Não deve haver `]][` (tags grudadas).
        assert "]][" not in out, f"Tags grudadas: {out!r}"

    def test_tag_nao_engole_quebra_de_linha(self, tok: Tokenizer) -> None:
        texto = "L = 8 m\nI_z = 208300 cm⁴"
        out = tok.preprocess(texto)
        # A quebra de linha original deve estar preservada em algum lugar
        # do output (não pode ter sido comida por uma tag SYM).
        assert "\n" in out, f"Quebra de linha perdida: {out!r}"

    def test_tag_nao_engole_parens_de_aposicao(self, tok: Tokenizer) -> None:
        out = tok.preprocess("define o módulo (E_cs) em MPa")
        # Os parens devem ficar fora da tag SYM (são aposição, não
        # parens matemáticos).
        if "[SYM:" in out:
            # Se virou SYM, não pode ter `(` no conteúdo
            sym_content = out.split("[SYM:")[1].split("]")[0]
            assert "(" not in sym_content, (
                f"Parens de aposição dentro de SYM: {out!r}"
            )
