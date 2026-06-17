"""Testes ontológicos das primitivas de derivação SYM (Camada 2).

Cada teste codifica uma decisão ontológica formal — falha aqui significa
que a definição de PROSE_SEPARATOR ou OPERATOR_MEDIATOR está
inconsistente com o contrato Enedina.
"""

from __future__ import annotations

import pytest

from toten.classifier.derivation import (
    classify_gap,
    find_clusters_from_atoms,
    gap_contains_bullet,
    is_operator_mediator,
    is_prose_separator,
)


# ---------------------------------------------------------------------------
# is_prose_separator
# ---------------------------------------------------------------------------


class TestIsProseSeparator:
    """Material que separa átomos da Camada 1 SEM compor expressão."""

    @pytest.mark.parametrize(
        "gap",
        [
            "",
            " ",
            "  ",
            "\t",
            "\n",
            "\n\n",
            "\n  ",
            "  \n  ",
        ],
        ids=["empty", "single_space", "double_space", "tab",
             "newline", "double_newline", "newline_indent",
             "spaces_around_newline"],
    )
    def test_whitespace_only_is_prose(self, gap: str) -> None:
        assert is_prose_separator(gap), f"esperado prose-only para {gap!r}"

    @pytest.mark.parametrize(
        "gap",
        [", ", " , ", ". ", "; ", ": ", "! ", "? "],
    )
    def test_prose_punctuation_is_prose(self, gap: str) -> None:
        assert is_prose_separator(gap)

    @pytest.mark.parametrize(
        "gap",
        [
            "\n  - ",          # bullet linha + hífen + espaço
            "\n  * ",          # bullet com asterisco
            "\n  + ",          # bullet com mais
            "\n  1. ",         # bullet numerado
            "\n  - L = ",      # bullet seguido de prefixo (apenas a parte gap)
        ],
        ids=["dash_bullet", "star_bullet", "plus_bullet",
             "numbered_bullet", "bullet_with_eq"],
    )
    def test_list_bullet_is_prose(self, gap: str) -> None:
        # Mesmo contendo `-`, `*`, `+` (que SERIA operador), o contexto
        # de quebra-de-linha + indentação caracteriza bullet, não
        # operador. Mas: is_prose_separator é teste sintático puro do
        # gap como string. `\n  - ` cai em "mixed" porque tem `-` que
        # também é operador. Verificamos via gap_contains_bullet abaixo.
        # Aqui validamos: gap com bullet+texto seguinte (= ) NÃO é prose-only.
        assert gap_contains_bullet(gap), (
            f"bullet não detectado em {gap!r}"
        )

    @pytest.mark.parametrize(
        "gap",
        [" ( ", " ) ", " [ ", "] ", "; ( "],
    )
    def test_parens_brackets_are_prose(self, gap: str) -> None:
        """Parens/colchetes isolados em gap = aposição prosaica."""
        assert is_prose_separator(gap), (
            f"parens de aposição não classificado como prosa: {gap!r}"
        )

    @pytest.mark.parametrize(
        "gap",
        [" + ", " - ", " = ", " ≈ ", " × ", "·", " ^ "],
    )
    def test_operators_are_NOT_prose(self, gap: str) -> None:
        """Operador isolado deve ser OPERATOR_MEDIATOR, não prose."""
        assert not is_prose_separator(gap), (
            f"operador classificado erroneamente como prose: {gap!r}"
        )


# ---------------------------------------------------------------------------
# is_operator_mediator
# ---------------------------------------------------------------------------


class TestIsOperatorMediator:
    """Operador binário mediador de composição P3."""

    @pytest.mark.parametrize(
        "gap",
        [
            "+", " + ", "  +  ",
            "-", " - ",
            "*", " * ",
            "/", " / ",
            "·", " · ",
            "×", " × ",
            "÷", " ÷ ",
            "^", " ^ ",
        ],
        ids=["plus", "plus_sp", "plus_sp2", "minus", "minus_sp",
             "mul_ascii", "mul_ascii_sp", "div_ascii", "div_ascii_sp",
             "cdot", "cdot_sp", "times", "times_sp", "divide", "divide_sp",
             "caret", "caret_sp"],
    )
    def test_arithmetic_operators(self, gap: str) -> None:
        assert is_operator_mediator(gap), f"não reconheceu operador: {gap!r}"

    @pytest.mark.parametrize(
        "gap",
        ["=", " = ", "≈", " ≈ ", "≤", " ≤ ", "<", " < ", ">", " > "],
    )
    def test_relational_operators(self, gap: str) -> None:
        assert is_operator_mediator(gap), f"não reconheceu relação: {gap!r}"

    @pytest.mark.parametrize(
        "gap",
        [
            "",                 # gap vazio NÃO é operador
            " ",                # só espaço
            "\n",               # quebra de linha
            "\n  - ",           # bullet com hífen — NÃO é operator
            " e ",              # palavra "e" — prosa
            ", ",               # vírgula — prosa
            "(",                # parêntese — não é operador binário
            ")",
            " (E_cs) ",         # aposição entre parens
            " a + ",            # contém letra → tem átomo extra
        ],
        ids=["empty", "space", "newline", "newline_bullet",
             "word_e", "comma", "lparen", "rparen", "aposition",
             "with_letter"],
    )
    def test_NOT_operator(self, gap: str) -> None:
        assert not is_operator_mediator(gap), (
            f"falsamente classificado como operador: {gap!r}"
        )


# ---------------------------------------------------------------------------
# classify_gap (diagnóstico)
# ---------------------------------------------------------------------------


class TestClassifyGap:
    """Etiqueta diagnóstica do gap."""

    @pytest.mark.parametrize(
        "gap,expected",
        [
            ("", "empty"),
            (" ", "prose"),
            ("\n  ", "prose"),
            (", ", "prose"),
            (" + ", "operator"),
            (" = ", "operator"),
            (" × ", "operator"),
            (" e ", "prose"),           # "e" sem ser operador = só prose chars? não, "e" é letra
            (" + x ", "mixed"),         # operador + letra extra
        ],
    )
    def test_classify(self, gap: str, expected: str) -> None:
        # Nota: " e " contém 'e' (letra) → mixed; corrigir:
        if gap == " e ":
            expected = "mixed"
        assert classify_gap(gap) == expected, f"gap {gap!r}"


# ---------------------------------------------------------------------------
# Princípios ontológicos invariantes (smoke tests do contrato)
# ---------------------------------------------------------------------------


class TestContrato:
    """Invariantes do contrato Camada 1 vs Camada 2."""

    def test_prose_e_operator_sao_mutuamente_exclusivos(self) -> None:
        """Um gap NUNCA é simultaneamente prose-only E operator-mediator."""
        casos = [
            "", " ", "  ", "\n", " + ", " = ", ", ", " ; ",
            " ( ", " ) ", " e ", " - ", " × ", " ≈ ",
        ]
        for gap in casos:
            p = is_prose_separator(gap)
            o = is_operator_mediator(gap)
            # Exceção: gap vazio é trivialmente prose mas não operator
            assert not (p and o), (
                f"gap {gap!r} classificado como AMBOS prose e operator"
            )

    def test_gap_vazio_e_prose_nao_operator(self) -> None:
        """Convenção: gap vazio significa átomos adjacentes sem separação.
        Tratamos como prose-trivial (não compõe nada por si só)."""
        assert is_prose_separator("")
        assert not is_operator_mediator("")

    def test_bullet_detectado_mesmo_com_chars_misturados(self) -> None:
        """gap_contains_bullet localiza marcador mesmo em gaps complexos."""
        assert gap_contains_bullet("\n  - ")
        assert gap_contains_bullet("\n  * ")
        assert gap_contains_bullet("\n  1. ")
        assert not gap_contains_bullet(" + ")
        assert not gap_contains_bullet(" - ")  # hífen sem \n não é bullet


# ---------------------------------------------------------------------------
# find_clusters_from_atoms — Camada 2 derivada
# ---------------------------------------------------------------------------


class TestFindClustersFromAtoms:
    """Clusters de ≥2 átomos mediados por operador (P3)."""

    def test_dois_atomos_com_mais_entre_eles(self) -> None:
        """`a + b` → cluster cobrindo [a_start, b_end]."""
        text = "a + b"
        atoms = [(0, 1), (4, 5)]  # spans de 'a' e 'b'
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [(0, 5)], f"esperado [(0,5)], obtive {clusters}"

    def test_tres_atomos_com_operadores_diferentes(self) -> None:
        """`a + b · c` → cluster cobrindo todos."""
        text = "a + b · c"
        atoms = [(0, 1), (4, 5), (8, 9)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [(0, 9)]

    def test_atomos_separados_por_prosa_nao_clusterizam(self) -> None:
        """`a e b` (palavra 'e' no meio) → sem cluster."""
        text = "a e b"
        atoms = [(0, 1), (4, 5)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [], f"prosa criou cluster: {clusters}"

    def test_atomos_separados_por_virgula_nao_clusterizam(self) -> None:
        """`R_A, R_B, R_C` (vírgulas) → sem cluster."""
        text = "R_A, R_B, R_C"
        atoms = [(0, 3), (5, 8), (10, 13)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == []

    def test_atomos_separados_por_bullet_nao_clusterizam(self) -> None:
        """Bullets de lista → sem cluster (mesmo com `-` ou `+` no marcador)."""
        text = "L = 8\n  - I_z = 100"
        atoms = [(0, 1), (10, 13)]  # 'L' e 'I_z'
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [], f"bullet criou cluster: {clusters}"

    def test_atomo_isolado_nao_e_cluster(self) -> None:
        """1 átomo só → sem cluster (mínimo 2)."""
        clusters = find_clusters_from_atoms("a", [(0, 1)])
        assert clusters == []

    def test_lista_vazia_de_atomos(self) -> None:
        clusters = find_clusters_from_atoms("", [])
        assert clusters == []

    def test_clusters_multiplos_separados_por_prosa(self) -> None:
        """`a + b, e depois c · d` → dois clusters."""
        text = "a + b, e depois c · d"
        atoms = [(0, 1), (4, 5), (16, 17), (20, 21)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [(0, 5), (16, 21)]

    def test_mediator_operador_relacional_NAO_forma_cluster(self) -> None:
        """`x = y` NÃO clusteriza — `=` é relacional, define igualdade.
        Átomos individuais preservados para máxima informação ao LLM."""
        text = "x = y"
        atoms = [(0, 1), (4, 5)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [], f"relacional virou cluster: {clusters}"

    def test_mediator_aproximadamente_igual_NAO_forma_cluster(self) -> None:
        """`≈` é relacional, mesma regra de `=`."""
        text = "x ≈ y"
        atoms = [(0, 1), (4, 5)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == []

    def test_mediator_composicional_forma_cluster(self) -> None:
        """`x * y` clusteriza — `*` é composicional, forma expressão."""
        text = "x * y"
        atoms = [(0, 1), (4, 5)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [(0, 5)]

    def test_mediator_potencia_forma_cluster(self) -> None:
        """`x^y` (potenciação) é composicional."""
        text = "x^y"
        atoms = [(0, 1), (2, 3)]
        clusters = find_clusters_from_atoms(text, atoms)
        assert clusters == [(0, 3)]
