"""Testes E2E do pipeline v0.7 — coexistência das 8 entidades OEE v1.1.

Validação ontológica de que as 4 evoluções (OEE 8º tipo, unit-only
QTY, NumeroInstantiator, R2L grouping) interagem corretamente com as
entidades pré-existentes (IDX, SYM, CONST, OP, REL, PROSA).

Foco: priority resolution entre tipos sobrepostos, robustez de bordas
ontológicas (prosa-vs-unidade, fração-vs-referência, locale ambíguo),
ausência de regressões em casos canônicos.
"""

from __future__ import annotations

import re

import pytest

from toten import Tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_ontology("oee-v1")


class TestPriorityResolution:
    """QTY > NUM, IDX > SYM, CONST detectado antes de SYM."""

    def test_qty_vence_num_quando_unit_segue(self, tok: Tokenizer) -> None:
        out = tok.preprocess("350 MPa de tensão")
        assert "[QTY value=350" in out
        assert "[NUM" not in out  # 350 fica dentro do QTY, não vira NUM solto

    def test_num_emerge_quando_sem_unit(self, tok: Tokenizer) -> None:
        out = tok.preprocess("100 alunos no exame")
        assert "[NUM value=100" in out
        assert "[QTY" not in out

    def test_const_vence_sym(self, tok: Tokenizer) -> None:
        # k_B é constante universal canônica — não deve virar SYM
        out = tok.preprocess("k_B é a constante de Boltzmann")
        assert "[CONST:k_B]" in out
        assert "[SYM:k_B]" not in out


class TestMixOcho_Entidades:
    """Texto técnico realista misturando IDX + SYM + QTY + NUM + CONST."""

    def test_caso_engenharia_mecanica(self, tok: Tokenizer) -> None:
        texto = "σ_y do AISI 1045 é 215 MPa para 100 ciclos"
        out = tok.preprocess(texto)
        assert "[SYM:σ_y]" in out
        assert "[IDX:aisi-1045]" in out
        assert "[QTY value=215 unit=MPa" in out
        assert '[NUM value=100 locale=none repr=inteira original="100"]' in out

    def test_caso_termodinamica(self, tok: Tokenizer) -> None:
        texto = "k_B = 1,38 × 10⁻²³ J/K relaciona T = −273,15 °C"
        out = tok.preprocess(texto)
        assert "[CONST:k_B]" in out
        assert "[QTY value=1.38e-23 unit=J/K" in out
        assert "[QTY value=-273.15 unit=°C" in out


class TestUnitOnlyOntologico:
    """Evolução 2 — unit-only QTY exige marcador estrutural inequívoco."""

    def test_kgf_cm2_emite_unit_only(self, tok: Tokenizer) -> None:
        # Tem operador `/` e superscript `²` — marcador estrutural
        out = tok.preprocess("kgf/cm² é a unidade clássica")
        assert "[QTY unit=kgf/cm²" in out

    def test_celsius_emite_unit_only(self, tok: Tokenizer) -> None:
        # Char não-ASCII `°` — marcador estrutural
        out = tok.preprocess("°C de calor sensível")
        assert "[QTY unit=°C" in out

    def test_em_preposicao_nao_emite_unit_only(self, tok: Tokenizer) -> None:
        # "Em" é alfa-ASCII puro sem marcador → fica em PROSA.
        # Antes da correção: "Em" virava [QTY unit=Em] (false positive).
        out = tok.preprocess("Em 2023 houve 1.234 casos")
        assert "[QTY unit=Em" not in out
        assert out.startswith("Em ")

    def test_pa_puro_nao_emite_unit_only(self, tok: Tokenizer) -> None:
        # "Pa" sozinho em prosa, sem marcador estrutural
        out = tok.preprocess("Pa de pressão é uma referência")
        assert "[QTY unit=Pa" not in out
        assert out.startswith("Pa ")

    def test_tf_alfa_nao_emite_unit_only(self, tok: Tokenizer) -> None:
        out = tok.preprocess("A unidade Tf é tonelada-força")
        assert "[QTY unit=Tf" not in out

    def test_350_mpa_mantem_qty_com_value(self, tok: Tokenizer) -> None:
        # Caso canônico: número + unit vira QTY completa, não unit-only
        out = tok.preprocess("σ_y = 350 MPa")
        assert "[QTY value=350 unit=MPa" in out


class TestFracaoVsReferencia:
    """Evolução 3 — fração ontológica vs referência temporal/normativa."""

    def test_fracao_genuina_vira_num(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Aplicar 3/4 da força máxima")
        assert '[NUM value=0.75' in out
        assert 'repr=fracionaria' in out

    def test_mes_ano_nao_vira_fracao(self, tok: Tokenizer) -> None:
        # 12/2024 NÃO é fração — denominador ≥ 4 dígitos é referência
        out = tok.preprocess("Norma 12/2024 vigente")
        assert "[NUM value=0.0594" not in out
        # 12 e 2024 viram NUMs separados, `/` permanece literal
        assert '[NUM value=12' in out
        assert '[NUM value=2024' in out

    def test_edital_e_fracao_no_mesmo_texto(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Edital 5/2026 sobre item 1/2 do regulamento")
        # 1/2 = fração
        assert '[NUM value=0.5 locale=none repr=fracionaria original="1/2"]' in out
        # 5/2026 = referência, não fração
        assert "[NUM value=0.00246" not in out


class TestLocaleOntologico:
    """Evolução 3 — locale detection sem heurística ad-hoc."""

    def test_zero_dot_nao_e_milhar_pt_br(self, tok: Tokenizer) -> None:
        # "0.500" — milhar PT-BR exige primeiro dígito ≠ 0
        out = tok.preprocess("Valor 0.500 em texto")
        assert "[NUM value=0.5 locale=en" in out

    def test_um_dot_500_e_milhar_pt_br(self, tok: Tokenizer) -> None:
        # "1.500" — primeiro dígito 1 → milhar PT-BR válido
        out = tok.preprocess("Carga de 1.500 toneladas")
        assert "[NUM value=1500" in out
        assert "locale=pt-br" in out

    def test_decimal_pt_br_vs_en_mesmo_texto(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Verifique 1.234,56 e 1,234.56 separadamente")
        # Ambos parseiam para 1234.56 mas com locales distintos preservados
        assert 'value=1234.56' in out
        assert 'locale=pt-br' in out
        assert 'locale=en' in out


class TestR2LIntegracao:
    """Evolução 4 — r2l= aparece em values ≥1000 (QTY e NUM)."""

    def test_qty_grande_tem_r2l(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Capacidade de 12500 kN")
        assert 'r2l="12 500"' in out

    def test_num_grande_tem_r2l(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Total de 1234567 itens")
        assert 'r2l="1 234 567"' in out

    def test_qty_pequeno_sem_r2l(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Apenas 350 MPa")
        assert "r2l=" not in out

    def test_num_pequeno_sem_r2l(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Total 100 itens")
        assert "r2l=" not in out

    def test_qty_negativo_grande_tem_r2l(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Pretensão −1500 kN aplicada")
        assert 'r2l="-1 500"' in out


class TestFormatacaoConsistente:
    """Modo A e Modo B usam o mesmo `_format_value` para o mesmo valor."""

    def test_inteiro_grande_qty_vs_num_modo_b(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Carga 1234567 kN e 1234567 pessoas")
        # Mesmo número, mesma formatação em QTY e NUM
        assert "value=1234567" in out
        # Aparece duas vezes (QTY + NUM)
        assert out.count("value=1234567") == 2

    def test_consistencia_int_vs_float(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Exatamente 1000000 ciclos")
        # 1e6 inteiro → "1000000" (não "1e+06")
        assert "value=1000000" in out


class TestNumeroNaoQuebraExistente:
    """Regressão — Evoluções v0.7 não devem afetar capacidades pré-existentes."""

    def test_constante_universal(self, tok: Tokenizer) -> None:
        out = tok.preprocess("π aparece em muitas fórmulas")
        assert "[CONST:π]" in out

    def test_relacao_estrutural(self, tok: Tokenizer) -> None:
        out = tok.preprocess("σ ≤ σ_adm")
        assert "≤" in out

    def test_identificador_tecnico(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Liga AISI 4340")
        assert "[IDX:aisi-4340]" in out

    def test_expressao_simbolica_atomica(self, tok: Tokenizer) -> None:
        out = tok.preprocess("σ_y é o limite de escoamento")
        assert "[SYM:σ_y]" in out


class TestOrdinaisPTBR:
    """Q4 — Ordinais PT-BR via UCD `ª`/`º` (ORDINAL INDICATOR).

    Princípio: ordinais são números com convenção tipográfica posicional
    derivada do UCD (`FEMININE/MASCULINE ORDINAL INDICATOR`). Forma atômica
    única `[NUM value=N repr=ordinal original="Nª"]`, sem fragmentação.
    """

    def test_ordinal_feminino_atomico(self, tok: Tokenizer) -> None:
        out = tok.preprocess("A 1ª questão é difícil")
        assert '[NUM value=1 locale=none repr=ordinal original="1ª"]' in out
        # Não deve fragmentar em [NUM 1]ª
        assert ']ª' not in out

    def test_ordinal_masculino_atomico(self, tok: Tokenizer) -> None:
        out = tok.preprocess("O 2º passo é calcular")
        assert '[NUM value=2 locale=none repr=ordinal original="2º"]' in out

    def test_ordinal_multi_digito(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Acerca da 100ª revisão")
        assert '[NUM value=100 r2l="100" locale=none repr=ordinal original="100ª"]' in out or \
               '[NUM value=100 locale=none repr=ordinal original="100ª"]' in out

    def test_grau_angular_nao_e_ordinal(self, tok: Tokenizer) -> None:
        # `°` (DEGREE SIGN U+00B0) ≠ `º` (MASCULINE ORDINAL INDICATOR U+00BA)
        # `45° de inclinação` é grau angular, não ordinal
        out = tok.preprocess("ângulo de 45°")
        assert "[QTY value=45 unit=°" in out
        assert "ordinal" not in out

    def test_ordinal_e_decimal_no_mesmo_texto(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Compare 9,11 vs 9,9 e a 1ª opção")
        assert "value=9.11" in out
        assert "value=9.9" in out
        assert 'repr=ordinal original="1ª"' in out


class TestSymContextSensitivoPalavras:
    """Q2 — A.6.2 estendido: palavras inteiras antes de operador relacional.

    Princípio: sintaxe `<token> = <grandeza>` é declarativa — nomeia a
    grandeza independente do comprimento do token. Vale para `L = 1,80 m`
    (1 char) e `largura = 1,80 m` (palavra). Sem `=`/`≤`/`≈`, palavra
    fica em PROSA (decisão consciente, alta confiança).
    """

    def test_palavra_curta_def(self, tok: Tokenizer) -> None:
        out = tok.preprocess("largura = 5 m")
        assert "[SYM:largura]" in out
        assert "[QTY value=5 unit=m" in out

    def test_palavra_longa_def(self, tok: Tokenizer) -> None:
        out = tok.preprocess("comprimento = 1,80 m")
        assert "[SYM:comprimento]" in out

    def test_palavra_com_operador_desigualdade(self, tok: Tokenizer) -> None:
        out = tok.preprocess("profundidade ≤ 10 cm")
        assert "[SYM:profundidade]" in out

    def test_palavra_com_aproximacao(self, tok: Tokenizer) -> None:
        out = tok.preprocess("densidade ≈ 7,85 kg/m³")
        assert "[SYM:densidade]" in out

    def test_letra_unica_continua_funcionando(self, tok: Tokenizer) -> None:
        # Regressão: A.6.2 estendido não deve quebrar caso `L = 1,80 m`
        out = tok.preprocess("L = 1,80 m")
        assert "[SYM:L]" in out

    def test_cadeia_de_definicoes(self, tok: Tokenizer) -> None:
        # Múltiplas palavras numa mesma cadeia `=`
        out = tok.preprocess("Onde, largura = comprimento = 5 m")
        assert "[SYM:largura]" in out
        assert "[SYM:comprimento]" in out

    def test_palavra_em_prosa_sem_operador_nao_vira_sym(
        self, tok: Tokenizer
    ) -> None:
        out = tok.preprocess("A largura é 5 m de extensão")
        assert "[SYM:largura]" not in out
        # QTY ainda capturada
        assert "[QTY value=5 unit=m" in out

    def test_substantivos_em_prosa_natural(self, tok: Tokenizer) -> None:
        out = tok.preprocess("Considere a largura e a altura da viga")
        assert "[SYM:largura]" not in out
        assert "[SYM:altura]" not in out


class TestEspacoComposicionalSI:
    """Espaço entre átomos do oráculo é operador multiplicativo (BIPM §5.2).

    Caso real: ENEM 2019 (alvorada-enem_2019-2_q_97) publica `10 m s−2`
    e variantes — convenção SI permite `m s⁻²` equivalente a `m·s⁻²`.
    """

    def test_aceleracao_com_espaco(self, tok: Tokenizer) -> None:
        out = tok.preprocess("g = 10 m s⁻²")
        assert "[QTY value=10 unit=m s⁻² dim=[0,1,-2,0,0,0,0]]" in out

    def test_velocidade_com_espaco(self, tok: Tokenizer) -> None:
        out = tok.preprocess("v = 5 m s⁻¹")
        assert "[QTY value=5 unit=m s⁻¹ dim=[0,1,-1,0,0,0,0]]" in out

    def test_tres_atomos_com_espaco(self, tok: Tokenizer) -> None:
        out = tok.preprocess("F = 10 kg m s⁻²")
        assert "unit=kg m s⁻²" in out
        assert "dim=[1,1,-2,0,0,0,0]" in out

    def test_n_m_sem_expoente(self, tok: Tokenizer) -> None:
        # `N m` (sem expoente) = N·m = momento
        out = tok.preprocess("M = 10 N m")
        assert "unit=N m" in out
        assert "dim=[1,2,-2,0,0,0,0]" in out

    def test_entropia_molar_com_espaco(self, tok: Tokenizer) -> None:
        out = tok.preprocess("S = 5 mol K⁻¹")
        assert "unit=mol K⁻¹" in out

    def test_espaco_nao_captura_prosa(self, tok: Tokenizer) -> None:
        # `m` seguido de palavra não-átomo NÃO estende a composição
        out = tok.preprocess("uma viga de 10 m de comprimento")
        # Apenas UM QTY (10 m), prosa preservada
        assert out.count("[QTY") == 1
        assert "de comprimento" in out

    def test_espaco_e_pontocentrado_equivalentes(self, tok: Tokenizer) -> None:
        # Mesmo valor + mesma dimensão; só o `unit_text` difere
        a = tok.preprocess("10 m·s⁻²")
        b = tok.preprocess("10 m s⁻²")
        assert "dim=[0,1,-2,0,0,0,0]" in a
        assert "dim=[0,1,-2,0,0,0,0]" in b


class TestNumberCookbookCanonico:
    """SOTA Number Cookbook — caso clássico Yang ICLR 2025."""

    def test_9_11_vs_9_9(self, tok: Tokenizer) -> None:
        out = tok.preprocess("9,11 vs 9,9 — qual é maior?")
        assert 'value=9.11' in out
        assert 'value=9.9' in out

    def test_extracao_numerica_modo_b(self, tok: Tokenizer) -> None:
        # LLM downstream deve enxergar value= explícito em ambos
        out = tok.preprocess("9,11 e 9,9")
        # Extrai os values: ambos > 9.0
        values = [float(m) for m in re.findall(r"value=(\d+\.?\d*)", out)]
        assert 9.11 in values
        assert 9.9 in values
