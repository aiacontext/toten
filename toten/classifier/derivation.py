"""Camada 2 — derivação ontológica de ExpressaoSimbolica.

Implementa a CARACTERIZAÇÃO UNIFICADA: uma região é simbólica se
satisfaz pelo menos UMA destas 6 condições:

  A) MARCA TIPOGRÁFICA INTRÍNSECA (P8 estrito) — dentro de uma única
     palavra-token:
       1. Sub/sup Unicode ou subscript ASCII (`²`, `_y`, `_max`)
       2. Letra grega (`σ`, `α`, `π`, `Δ`)
       3. Operador matemático Unicode (`·`, `×`, `÷`, `∫`, `∑`, `√`,
          `≈`, `≤`, `≥`, `≠`, `≡`, `∂`, `∇`)
       4. Dígito justaposto a letra sem espaço (`3x`, `2π`, `pl12`)
       5. Estrutura sintática delimitada (`<nome>(<args>)`, `\\<cmd>`,
          `$...$`, `|x|`, `⟨ψ|`, `∫...d<var>`)

  B) MARCA CONTEXTUAL (propagação por adjacência):
       6. Letra isolada (1-3 chars latino ou grego) imediatamente vizinha
          a entidade matemática reconhecida — operador binário, QTY,
          IDX, CONST ou outro SYM já identificado — separada apenas por
          whitespace.

Princípio unificador
--------------------
SÍMBOLO MATEMÁTICO = entidade portadora de marca intrínseca OU letra
propagada por adjacência sintática a contexto matemático.

SEM catálogo de variáveis curtas. SEM palavras-âncora. SEM flag
`atomic_over_protected`. SEM distinção composicional/relacional.
SEM regex permissivo cobrindo casos arbitrários.

Casos genuinamente ambíguos (e.g., `a viga` — artigo ou variável?)
ficam como PROSA — decisão consciente em favor de previsibilidade.
Resolução semântica é responsabilidade downstream (LLM).
"""

from __future__ import annotations

import unicodedata

import regex as re

# ---------------------------------------------------------------------------
# PROSE_SEPARATOR e OPERATOR_MEDIATOR — primitivas Camada 1 ↔ Camada 2
# ---------------------------------------------------------------------------

# Bullet de lista: marcador de item após quebra de linha.
# P6 — convenção tipográfica. Bloqueia interpretação do marcador como
# operador unário (sinal matemático).
_BULLET_RE = re.compile(r"\n[ \t]*(?:[-*+]|\d+[.)])[ \t]+")

# Whitespace + pontuação prosa + parens vazios. NÃO inclui operadores.
_PROSE_ONLY_RE = re.compile(
    r"""^[\s.,;:!?"'`""''…—–()\[\]{}]*$""",
)


def gap_contains_bullet(gap: str) -> bool:
    """True se o gap contém um marcador de bullet de lista."""
    return bool(_BULLET_RE.search(gap))


def is_prose_separator(gap: str) -> bool:
    """True se o gap consiste APENAS de material de prosa."""
    if not gap:
        return True
    return bool(_PROSE_ONLY_RE.match(gap))


# Operadores binários — mediadores P3, divididos por classe ontológica:
#   COMPOSICIONAL — forma expressão nova: + - * / · × ÷ ^ ∝
#   RELACIONAL — define/compara entidades: = ≈ ≠ ≡ ≤ ≥ < > ⇒ ⇔ → ↔
# Distinção informa cluster derivation: clusters precisam de ≥1 mediador
# COMPOSICIONAL para emergirem (definição IDX = IDX é preservada como
# átomos, não absorvida em cluster).
_OPERATOR_COMPOSITIONAL_CHARS = r"+\-*/·×÷^∝"
# ~ ASCII incluído: usado em estatística para "X ~ N(...)" (distribuído como)
_OPERATOR_RELATIONAL_CHARS = r"=≈≠≡≤≥<>⇒⇔→↔~"
_OPERATOR_MEDIATOR_CHARS = (
    _OPERATOR_COMPOSITIONAL_CHARS + _OPERATOR_RELATIONAL_CHARS
)
_OPERATOR_MEDIATOR_RE = re.compile(
    rf"^\s*[{_OPERATOR_MEDIATOR_CHARS}]+\s*$",
)
_OPERATOR_COMPOSITIONAL_RE = re.compile(
    rf"^\s*[{_OPERATOR_COMPOSITIONAL_CHARS}]+\s*$",
)


def is_operator_mediator(gap: str) -> bool:
    """True se o gap é APENAS operador binário matemático + whitespace.

    Inclui composicionais E relacionais. Use `is_compositional_operator`
    para distinguir composição genuína de definição.

    Bullet de lista (P6) bloqueia.
    """
    if not gap:
        return False
    if gap_contains_bullet(gap):
        return False
    return bool(_OPERATOR_MEDIATOR_RE.match(gap))


def is_compositional_operator(gap: str) -> bool:
    """True se o gap é operador COMPOSICIONAL (+, -, *, /, ·, ×, ÷, ^, ∝).

    Operadores composicionais formam nova expressão matemática a partir
    de termos. Operadores relacionais (=, ≈, etc.) definem/comparam —
    átomos preservados separadamente.
    """
    if not gap:
        return False
    if gap_contains_bullet(gap):
        return False
    return bool(_OPERATOR_COMPOSITIONAL_RE.match(gap))


def classify_gap(gap: str) -> str:
    """Diagnóstico do gap: 'empty', 'operator', 'prose' ou 'mixed'."""
    if not gap:
        return "empty"
    if is_operator_mediator(gap):
        return "operator"
    if is_prose_separator(gap):
        return "prose"
    return "mixed"


# ---------------------------------------------------------------------------
# Caracterização ontológica de TOKEN-PALAVRA SIMBÓLICO
# ---------------------------------------------------------------------------

# Marca 1 — subscript/superscript Unicode ou subscript ASCII
_SUB_SUP_UNICODE = "²³⁰¹⁴⁵⁶⁷⁸⁹⁻⁺₀₁₂₃₄₅₆₇₈₉"

# Marca 2 — letra matemática Unicode (grego + Letterlike Symbols + infinity).
# Inclui:
#   - Grego α-ω, Α-Ω (blocos U+0370–U+03FF)
#   - Letterlike Symbols U+2100–U+214F (ℝ ℕ ℤ ℚ ℂ ℙ ℋ ℓ ℵ ℶ ℷ ℸ etc.)
#     — usadas como nomes canônicos de conjuntos numéricos e espaços (ℝ, ℂ).
#   - Mathematical Alphanumeric Symbols U+1D400–U+1D7FF (caligráficas etc.)
#   - ∞
_GREEK_RE = re.compile(
    r"[Ͱ-Ͽ℀-⅏\U0001D400-\U0001D7FF∞]"
)

# Marca 3 — chars matemáticos Unicode INEQUÍVOCOS via UCD (oráculo externo).
#
# Princípio ontológico [[oraculo-externo]]: usar autoridade Unicode em vez
# de listar manualmente. \p{Sm} (Symbol, math) cobre TODOS os símbolos
# matemáticos Unicode incluindo:
#   - Operadores aritméticos: + - = · × ÷ ∗ ∘ ∙
#   - Relacionais: ≈ ≠ ≡ ≤ ≥ < > ∝
#   - Lógicos: ∧ ∨ ¬ ⇒ ⇔ → ↔
#   - Conjuntos: ∈ ∉ ∋ ∋ ⊂ ⊃ ⊆ ⊇ ∪ ∩ ∅
#   - Quantificadores: ∀ ∃ ∄
#   - Cálculo: ∫ ∑ ∏ ∂ ∇ √ ∛ ∜ ∮
#
# EXCLUSÃO ascii: chars ASCII com Sm category (+, =, <, >) são AMBÍGUOS em
# prosa (`a+b` matemática vs `a + b` texto) — excluídos via lookahead
# `[^\\x00-\\x7F]`. Operadores ASCII inequívocos têm tratamento separado
# pelo lexicon OP (Camada 1).
# Compatibilidade legada: string de chars mais comuns. Mas a FONTE DE
# VERDADE é `is_math_unicode_char()` abaixo que consulta o UCD via
# `unicodedata.category(c) == "Sm"` e exclui ASCII.
_UNAMBIGUOUS_MATH_OPS = (
    "·×÷∗∘∙"           # aritméticos não-ASCII
    "≈≠≡≤≥∝"          # relacionais
    "∧∨¬⇒⇔→↔"         # lógicos
    "∈∉∋⊂⊃⊆⊇∪∩∅"     # conjuntos
    "∀∃∄"             # quantificadores
    "∫∑∏∂∇√∛∜∮"      # cálculo
)


# Chars matemáticos canônicos cuja categoria UCD NÃO é 'Sm' por razões
# históricas do Unicode, mas que são INEQUIVOCAMENTE matemáticos em uso:
#   ·  U+00B7  MIDDLE DOT       — Po (mas é multiplicação técnica)
#   ∙  U+2219  BULLET OPERATOR  — Sm (incluso no UCD)
#   °  U+00B0  DEGREE SIGN      — So (ângulo, temperatura — unidade, não op)
#   ′  U+2032  PRIME            — Po (derivada, ângulo)
#   ″  U+2033  DOUBLE PRIME     — Po
#   ‴  U+2034  TRIPLE PRIME     — Po
# Não incluímos ° por ser usado como unidade (pertence ao QTY/Pint).
_MATH_NON_SM_CHARS = frozenset("·′″‴")


def is_math_unicode_char(c: str) -> bool:
    """True se c é símbolo matemático Unicode INEQUÍVOCO (ULTIMATE ORACLE).

    Fonte primária: UCD (Unicode Character Database) via `unicodedata.category`.
    Categoria 'Sm' = Symbol, math cobre a maioria — ∀ ∃ ∈ ∧ ∨ ⇒ ⊂ ∪ ≈ ≠
    ≤ ≥ × ÷ ∫ ∑ ∂ ∇ √ etc. — sem listagem manual.

    ASCII (< 128) excluído: `=`, `+`, `<`, `>`, `*` são ambíguos com
    prosa/email/data; operadores ASCII inequívocos têm tratamento
    separado pelo lexicon OP.

    Exceções controladas (_MATH_NON_SM_CHARS): chars cuja categoria UCD
    é Po/So mas usados inequivocamente como operadores matemáticos em
    notação técnica (·, ′, ″, ‴).
    """
    if not c or ord(c) < 128:
        return False
    if unicodedata.category(c) == "Sm":
        return True
    return c in _MATH_NON_SM_CHARS

# Caractere-classe LETRA MATEMÁTICA derivada do UCD.
# Princípio ontológico: uma letra é "matemática" se tem categoria L
# E não tem decomposition_type tipográfico (<super> ou <sub>).
# Decomp super/sub identifica indicadores tipográficos universais:
#   - ª º              ordinais portugueses/espanhois
#   - ᵃ ᵇ ᶜ ʰ ʲ ʷ ...  Phonetic Extensions + Modifier Letters
#   - ⁿ ᵏ ₐ ...        Super/Subscript letters
# Fonte de verdade: Unicode UCD (não enumeração manual).
def _build_typographic_letters() -> frozenset[str]:
    chars = []
    for cp in range(0x10000):  # BMP cobre todos os superscript/subscript letters
        c = chr(cp)
        if not unicodedata.category(c).startswith("L"):
            continue
        decomp = unicodedata.decomposition(c)
        if decomp.startswith("<super>") or decomp.startswith("<sub>"):
            chars.append(c)
    return frozenset(chars)


_TYPOGRAPHIC_LETTERS = _build_typographic_letters()
_TYPOGRAPHIC_EXCL_CLASS = "[" + "".join(
    re.escape(c) for c in sorted(_TYPOGRAPHIC_LETTERS)
) + "]"
# Lookahead negativo (portável V0/V1) contra UCD-derived set.
_MATH_LETTER_CLASS = r"(?:(?!" + _TYPOGRAPHIC_EXCL_CLASS + r")[\p{L}\p{M}])"

# Marca 4 — combinação letra+dígito dentro da palavra-token, indicando
# notação matemática compacta. Duas formas:
#   4a) JUSTAPOSIÇÃO direta: `\d<letra>` ou `<letra>\d` (e.g., `3x`, `pl12`)
#   4b) MEDIAÇÃO ASCII: letra e dígito separados por operador matemático
#       ASCII (`/`, `*`, `^`, `+`, `-`) no mesmo token (e.g., `pl/12`,
#       `x*2`, `M^3`). Distingue de palavras puras como `e/ou` (sem dígito).
# CRÍTICO: "letra" aqui é _MATH_LETTER_CLASS (exclui indicadores tipográficos
# super/sub via UCD). Assim `1ª`, `2º`, `1ⁿ` não disparam A.4.
_DIGIT_LETTER_ADJACENT_RE = re.compile(
    r"\d" + _MATH_LETTER_CLASS + r"|" + _MATH_LETTER_CLASS + r"\d"
)
_LETTER_OP_DIGIT_RE = re.compile(
    _MATH_LETTER_CLASS + r"[/*^+\-]\d|\d[/*^+\-]" + _MATH_LETTER_CLASS
)

# Marca 5 — estrutura sintática delimitada
# - função aplicada: `<letra>(`
# - LaTeX: começa com `\` ou cercado por `$`
# - valor absoluto: `|<conteúdo matemático>|`
# - bra-ket: contém `⟨` ou `⟩`
# Função aplicada: letra (opcionalmente seguida de primes Lagrange) + `(`.
# Cobre: `f(x)`, `f'(x)`, `f''(x)`, `f‴(x)`, etc.
_FUNCTION_APPLY_RE = re.compile(r"[\p{L}\p{M}][\'″‴]*\(")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_DIRAC_RE = re.compile(r"[⟨⟩]")


# Diacríticos combinatórios usados em NOTAÇÃO MATEMÁTICA (lista branca).
# Acentos linguísticos comuns (´ ` ~ ^ usados em PT-BR: á à ã â etc.) NÃO
# entram aqui — viraria falso positivo em prosa natural.
_MATH_COMBINING_MARKS = frozenset({
    "̇",   # COMBINING DOT ABOVE — notação Newton: ẋ velocidade, ẍ aceleração
    "̈",   # COMBINING DIAERESIS — segunda derivada: ẍ (no Brasil usual)
    "⃗",   # COMBINING RIGHT ARROW ABOVE — vetor: x⃗
    "⃑",   # COMBINING RIGHT HARPOON ABOVE — vetor: x⃑
    "⃖",   # COMBINING LEFT ARROW ABOVE
    "⃡",   # COMBINING LEFT RIGHT ARROW ABOVE — vetor bidirecional
    "̊",   # COMBINING RING ABOVE — angstrom/átomo: Å, x̊
})


def _has_math_diacritic(token: str) -> bool:
    """True se o token contém letra com diacrítico INEQUIVOCAMENTE matemático.

    Decompõe NFD para detectar marcas combinatórias mesmo em chars
    pré-compostos (ẍ = LATIN SMALL X WITH DIAERESIS → x + ̈ em NFD).

    CIRCUMFLEX (̂) NÃO entra aqui — é AMBÍGUO em PT-BR (ô, ê, â são
    acentos linguísticos comuns; Ô em início de frase é prosa também).
    Operadores hat (Ĥ, Â, B̂) emergem via PROPAGAÇÃO A.6 quando há
    contexto matemático adjacente (`Ĥ = ...`, `Â·ψ = ...`).
    """
    decomposed = unicodedata.normalize("NFD", token)
    return any(c in _MATH_COMBINING_MARKS for c in decomposed)


def is_symbolic_token(token: str) -> bool:
    """True se a palavra-token exibe ≥1 marca matemática intrínseca (A.1–A.5).

    REQUISITO ESTRUTURAL: o token deve conter ≥1 letra ou mark combinatória
    (\\p{L} ou \\p{M}). Operadores PUROS (∪, ∧, ⇒, ∈ isolados) NÃO viram
    SYM atômico — são OperadorFormal passthrough. SYM denota
    entidade VARIÁVEL ou expressão COM variáveis.

    A propagação contextual (A.6) NÃO é avaliada aqui — esse predicado
    requer info sobre vizinhança de outros átomos, tratada em
    `find_propagated_symbols`.
    """
    if not token:
        return False
    # GATE ESTRUTURAL: precisa de letra/mark — operadores puros excluídos
    if not re.search(r"[\p{L}\p{M}]", token):
        return False
    # A.1 — sub/sup Unicode
    if any(c in _SUB_SUP_UNICODE for c in token):
        return True
    # A.1 — subscript ASCII (`_` entre chars, não no início)
    if "_" in token and not token.startswith("_") and not token.endswith("_"):
        return True
    # A.2 — letra grega ou letterlike Unicode
    if _GREEK_RE.search(token):
        return True
    # A.2.5 — letra com diacrítico MATEMÁTICO (Newton, vetor, etc.):
    # ẋ ẍ x⃗ x̊ — distingue de acentos linguísticos PT-BR (á é í ó ã etc.)
    if _has_math_diacritic(token):
        return True
    # A.2.6 — primes Lagrange (notação de derivada): <letra>+<primes> NO FIM
    # do token. Cobre y' y'' f''' mas NÃO `I'm` `don't` (primes no meio
    # com letra depois indicam contração linguística, não derivada).
    if re.search(r"[\p{L}\p{M}][\'″‴]+$", token):
        return True
    # A.3 — char matemático Unicode (consulta UCD: \p{Sm} excluindo ASCII)
    if any(is_math_unicode_char(c) for c in token):
        return True
    # A.4a — dígito justaposto a letra (sem chars no meio)
    if _DIGIT_LETTER_ADJACENT_RE.search(token):
        return True
    # A.4b — letra e dígito mediados por operador ASCII matemático
    if _LETTER_OP_DIGIT_RE.search(token):
        return True
    # A.5 — estrutura delimitada (função aplicada, LaTeX, Dirac)
    if _FUNCTION_APPLY_RE.search(token):
        return True
    if _LATEX_CMD_RE.search(token):
        return True
    if _DIRAC_RE.search(token):
        return True
    return False


# ---------------------------------------------------------------------------
# Balanceamento de delimitadores nas extremidades
# ---------------------------------------------------------------------------


_BRACKET_PAIRS = (("(", ")"), ("{", "}"), ("[", "]"))


def _balance_brackets_at_edges(token: str) -> str:
    """[LEGACY] Trunca o token destrutivamente. Mantido para compatibilidade.

    Versão atual usa `_split_at_orphan_brackets` que é NÃO-DESTRUTIVO
    e devolve todos os sub-segmentos balanceados (preservando atomicidade
    de cada parte). Use essa função preferencialmente.
    """
    segs = _split_at_orphan_brackets(token)
    return segs[0] if segs else ""


def _split_at_orphan_brackets(token: str) -> list[str]:
    """Divide o token em sub-segmentos balanceados, removendo apenas os
    delimitadores ÓRFÃOS (sem par dentro do token).

    Princípio: a palavra-token original (sem whitespace) representa uma
    unidade textual que o autor escreveu junta. Quando ela contém parens
    desbalanceados (escopo abre/fecha em token vizinho), os sub-segmentos
    SÃO REGIÕES MATEMATICAMENTE COESAS individualmente.

    Ex.: `ω²·M)·φ` (do contexto `(K - ω²·M)·φ`) tem `)` órfão na pos 4.
    Divide em:
        - "ω²·M"  (antes do `)` órfão)
        - "·φ"    (depois do `)` órfão)
    Ambos potencialmente SYM (cada um com sua marca matemática).

    Ex.: `(K` tem `(` órfão. Divide em ["K"].
    Ex.: `f(x)` é balanceado. Devolve ["f(x)"] inteiro.
    """
    if not token:
        return []
    pairs_open_to_close = {o: c for o, c in _BRACKET_PAIRS}
    pairs_close_to_open = {c: o for o, c in _BRACKET_PAIRS}

    # Fase 1: detecta posições de delimitadores ÓRFÃOS (sem par no token).
    # Closers órfãos: nenhum opener anterior matching (varredura esquerda→direita).
    closers_orphan: set[int] = set()
    depth: dict[str, int] = {o: 0 for o in pairs_open_to_close}
    stack_openers: list[tuple[str, int]] = []  # (opener_char, idx) — pos dos openers vivos
    for i, c in enumerate(token):
        if c in pairs_open_to_close:
            depth[c] += 1
            stack_openers.append((c, i))
        elif c in pairs_close_to_open:
            opener = pairs_close_to_open[c]
            if depth[opener] > 0:
                depth[opener] -= 1
                # Remove o último opener desse tipo da pilha
                for k in range(len(stack_openers) - 1, -1, -1):
                    if stack_openers[k][0] == opener:
                        stack_openers.pop(k)
                        break
            else:
                closers_orphan.add(i)
    # Openers órfãos: os que restaram na pilha
    openers_orphan: set[int] = {idx for _, idx in stack_openers}

    orphan_positions = sorted(closers_orphan | openers_orphan)
    if not orphan_positions:
        return [token]

    # Fase 2: divide o token nas posições órfãs, descartando os delimitadores órfãos.
    segments: list[str] = []
    cursor = 0
    for pos in orphan_positions:
        if pos > cursor:
            segments.append(token[cursor:pos])
        cursor = pos + 1
    if cursor < len(token):
        segments.append(token[cursor:])

    return [s for s in segments if s]


# ---------------------------------------------------------------------------
# Tokenização em palavras-token
# ---------------------------------------------------------------------------

# Palavra-token: sequência maximal de caracteres não-whitespace.
# Pontuação forte (`,;:!?"”“`) separa palavras. Parens `()`, colchetes
# `[]`, chaves `{}` NÃO separam — são parte de notação matemática
# (função aplicada `M(x)`, intervalo `[a,b]`, conjunto `{1,2,3}`).
# Marcadores de lista `(a)` ainda são corretamente classificados como
# PROSA porque is_symbolic_token rejeita: `(` `)` não são `\p{Sm}` e
# `<letra>(` requer letra ANTES do `(` (em `(a)` o `(` vem primeiro).
_PROSE_STRONG_PUNCT = r",;:!?\"”“„‚‛"
_WORD_TOKEN_RE = re.compile(rf"[^\s{_PROSE_STRONG_PUNCT}]+")


def tokenize_into_word_tokens(text: str) -> list[tuple[int, int]]:
    """Particiona texto em palavras-token (sequências sem whitespace nem
    pontuação prosa forte).

    Returns:
        Lista de (start, end) de cada palavra-token, ordenada.
    """
    return [(m.start(), m.end()) for m in _WORD_TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Padrões SYM com delimitadores estruturalmente inequívocos
# (mantém-se em lexicon; mas roda APENAS para classes delimitadas)
# ---------------------------------------------------------------------------

# Classes do expression_lexicon que são GENUINAMENTE delimitadas
# (estrutura sintática própria, não regex permissivo).
ATOMIC_SYM_CLASSES: frozenset[str] = frozenset([
    "notacao_latex",
    "notacao_dirac_bra_ket",
    "integral_definida",
    "somatorio_produtorio",
    "derivada",
    "funcao_matematica_nomeada",
    "funcao_aplicada_generica",
    "valor_absoluto",
    "raiz_radical_unicode",
])


def find_delimited_sym_patterns(
    text: str,
    protected_atom_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Reconhece padrões SYM com delimitadores sintáticos inequívocos.

    Classes mantidas: funções aplicadas, LaTeX, integrais, somatórios,
    derivadas, valor absoluto, radicais, Dirac. TODAS têm marcadores
    próprios (`(`, `\\`, `∫`, `∑`, `|`, `√`, `⟨`/`⟩`) que delimitam o
    span de forma não-ambígua.

    Classes ELIMINADAS na refatoração 2026-05-28 (Via 6-condições):
        - cluster_com_sinal_unario
        - expressao_composta_sem_espaco
        - expressao_com_operador_espacado
        - cluster_compacto_matematico
        - monomio_com_coeficiente
    Substituídas por: `is_symbolic_token` (marcas A.1-A.5 dentro de
    palavra-token) + cluster derivado.

    Args:
        text: texto cru.
        protected_atom_spans: spans de Camada 1 a respeitar (átomos
            específicos como QTY, IDX preservados; padrão delimitado
            ainda absorve para preservar atomicidade — Função M(x)
            engole IDX `x` interno).

    Returns:
        Lista (start, end) ordenada.
    """
    from toten.instantiators.expression import (
        _load_expression_lexicon,
    )
    spans: list[tuple[int, int]] = []
    for classe, pattern, _bypass_p8, _atomic_over in _load_expression_lexicon():
        if classe not in ATOMIC_SYM_CLASSES:
            continue
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            sub = text[start:end]
            sub_stripped = sub.rstrip(" \t.,;:·*/+-^=≈≤≥<>")
            sub_stripped = sub_stripped.lstrip(" \t.,;:·*/^=≈≤≥<>")
            # Parens órfãs de extremidade
            while (
                sub_stripped.count("(") < sub_stripped.count(")")
                and sub_stripped.endswith(")")
            ):
                sub_stripped = sub_stripped[:-1]
            while (
                sub_stripped.count(")") < sub_stripped.count("(")
                and sub_stripped.startswith("(")
            ):
                sub_stripped = sub_stripped[1:]
            if len(sub_stripped) < 2:
                continue
            offset = sub.find(sub_stripped)
            if offset < 0:
                continue
            spans.append((start + offset, start + offset + len(sub_stripped)))
    spans.sort()
    return spans


# ---------------------------------------------------------------------------
# Camada 2 — derivação de SYM (algoritmo unificado)
# ---------------------------------------------------------------------------


def _split_word_around_layer1(
    w_start: int,
    w_end: int,
    layer1_sorted: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Divide um word-token em segmentos não cobertos por átomos Camada 1.

    Exemplo: word=(0,5) `α_p·E`, layer1=[(0,3)] → [(3,5)] = `·E`.
    """
    overlapping = sorted(
        (ps, pe) for ps, pe in layer1_sorted
        if ps < w_end and w_start < pe
    )
    if not overlapping:
        return [(w_start, w_end)]
    segments: list[tuple[int, int]] = []
    cursor = w_start
    for ps, pe in overlapping:
        if ps > cursor:
            segments.append((cursor, ps))
        cursor = max(cursor, pe)
    if cursor < w_end:
        segments.append((cursor, w_end))
    return segments


# ---------------------------------------------------------------------------
# Sub-atomização de variáveis dentro de palavra-token contígua
# ---------------------------------------------------------------------------


# Variável atômica: SEQUÊNCIA CONTÍGUA de letras (variável simples como `x`,
# `K`, `phi`, ou nome de função como `tan`, `sin`, `exp`, `log`) + opcional
# subscript ASCII `_xyz` + opcional sup Unicode + opcional primes Lagrange.
#
# CRÍTICO: capturar letras contíguas como UMA unidade (não 1 letra de cada
# vez). Caso contrário fragmenta palavras prosa (`estacionário` → 12 átomos)
# e nomes de funções (`tan` → t, a, n). A separação em variáveis individuais
# acontece via OPERADORES no texto bruto (·, /, +, -, parens) — não dentro
# de sequência alfabética contígua.
#
# Primes só aceitos se NÃO seguidos de letra (lookahead) — distingue `y'`
# `f''` (derivadas) de `I'm` `don't` (contrações linguísticas).

# Subscript: aceita formas `_x`, `_xy` (ASCII curto) E `_{...}` (LaTeX
# com conteúdo arbitrário entre chaves). Cobre t_{1/2}, x_{k+1}, σ_{ij}.
_SUBSCRIPT_RE_FRAG = r"(?:_(?:\{[^{}]+\}|[\p{L}\p{M}\p{N}]+))?"

# Variável atômica: coeficiente numérico OPCIONAL + letras contíguas
# + subscript opc + sup Unicode opc + primes opc.
# Coeficiente: `2x`, `4y'`, `45a` — monômio matemático. NÃO cobre QTY
# (que tem unidade e é capturado pela Camada 1 antes).
_VAR_ATOMIC_RE = re.compile(
    r"\d*"                                                # coeficiente numérico opc
    + _MATH_LETTER_CLASS + r"+"                          # letras matemáticas contíguas
    + _SUBSCRIPT_RE_FRAG +                               # subscript ASCII ou LaTeX
    r"[²³⁴⁰¹⁵⁶⁷⁸⁹]*"                                    # sup Unicode opcional
    r"(?:[\'″‴]+(?![\p{L}\p{M}]))?"                       # primes opc; sem letra depois
)

# Função aplicada: <nome função>(<args balanceados nivel 1>)
# Nome de função pode ter sup Unicode (`tan²(x)`, `sin³(x)`).
_FUNC_APPL_RE = re.compile(
    r"[\p{L}\p{M}]+"                                      # nome função (≥1 letra)
    + _SUBSCRIPT_RE_FRAG +
    r"[²³⁴⁰¹⁵⁶⁷⁸⁹]*"                                    # sup opc antes de (
    r"(?:[\'″‴]+(?![\p{L}\p{M}]))?"                       # primes opc
    r"\((?:[^()]|\([^()]*\))*\)"                          # parens balanceados
)


# Lista de variáveis em delimitador: [A, B], (x, y, z), {p, q, r}.
# Padrão clássico de comutador, par ordenado, tupla matemática.
# Cada variável é 1-3 chars (latina/grega/com diacrítico) — comprimento
# que distingue de palavras prosa enumeradas como `[primeiro, segundo]`.
_LIST_VARS_DELIM_RE = re.compile(
    r"[\[\(\{]\s*"                                            # delimitador aberto
    r"(?:[\p{L}\p{M}]{1,3}(?:_[\p{L}\p{N}]+)?[²³⁴]?\s*,\s*)+"  # vars seguidas por vírgula
    r"[\p{L}\p{M}]{1,3}(?:_[\p{L}\p{N}]+)?[²³⁴]?"             # última variável
    r"\s*[\]\)\}]"                                            # delimitador fechado
)

# Variável simples para extrair de dentro de _LIST_VARS_DELIM_RE
_VAR_INSIDE_LIST_RE = re.compile(
    r"[\p{L}\p{M}]{1,3}(?:_[\p{L}\p{N}]+)?[²³⁴]?"
)


def _find_list_var_spans(text: str) -> list[tuple[int, int]]:
    """Detecta variáveis dentro de listas matemáticas `[X, Y]`, `(x, y)`, `{p, q}`.

    Cobre notações: comutador `[A, B]`, anticomutador `{A, B}`, par
    ordenado `(x, y)`, tupla matemática. Cada variável dentro vira SYM.
    """
    spans: list[tuple[int, int]] = []
    for m in _LIST_VARS_DELIM_RE.finditer(text):
        inner_start = m.start() + 1  # pula `[` `(` `{`
        inner_end = m.end() - 1      # pula `]` `)` `}`
        for v in _VAR_INSIDE_LIST_RE.finditer(text, inner_start, inner_end):
            spans.append((v.start(), v.end()))
    return spans


def _atomize_variables(token: str, base_offset: int) -> list[tuple[int, int]]:
    """Sub-atomiza uma palavra-token contígua em variáveis individuais.

    Princípio Opção A: cada VARIÁVEL é um átomo SYM separado. Operadores
    (·, ×, ÷, +, -, /, =) e parens entre variáveis ficam passthrough.

    Ex.: `η·U·I/v` → [(η), (U), (I), (v)] — 4 SYMs atômicos
    Ex.: `K_p·e(t)` → [(K_p), (e(t))] — 2 SYMs (e(t) como função aplicada)
    Ex.: `x²·y` → [(x²), (y)] — 2 SYMs (x² com sup)
    Ex.: `(K - ω²·M)·φ` → [(K), (ω²·M)] — mas (K) e ω²·M são word-tokens
        DIFERENTES; aqui processamos UMA palavra-token por vez.

    Para fragmentação correta: precedência FUNC > VAR (longest-match via
    posição). Posições retornadas em coordenadas ABSOLUTAS (base_offset).
    """
    spans: list[tuple[int, int]] = []
    # 1ª passada: FUNC aplicada (var + parens) — longest match
    func_matches = list(_FUNC_APPL_RE.finditer(token))
    func_spans = [(m.start(), m.end()) for m in func_matches]
    # 2ª passada: variáveis fora dos func_spans
    for m in _VAR_ATOMIC_RE.finditer(token):
        # Verifica se overlap com FUNC já capturado
        if any(fs <= m.start() and m.end() <= fe for fs, fe in func_spans):
            continue
        # Verifica se overlap com FUNC parcial (ex.: var dentro da arg de func)
        if any(fs <= m.start() < fe or fs < m.end() <= fe for fs, fe in func_spans):
            continue
        spans.append((m.start(), m.end()))
    spans.extend(func_spans)
    spans.sort()
    # Converte para coordenadas absolutas
    return [(base_offset + s, base_offset + e) for s, e in spans]


def find_sym_tokens(
    text: str,
    layer1_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Identifica palavras-token simbólicas pelas marcas A.1–A.5.

    Para cada palavra-token, subdivide em segmentos não cobertos por
    átomos Camada 1 (caso `α_p·E` onde `α_p` é IDX e `·E` é o
    segmento remanescente). Em cada segmento, aplica `is_symbolic_token`.

    Propagação A.6 é tratada em `find_propagated_symbols` (etapa
    seguinte do pipeline).
    """
    word_spans = tokenize_into_word_tokens(text)
    sym_spans: list[tuple[int, int]] = []
    layer1_sorted = sorted(layer1_spans)

    # Strip de extremidades:
    # rstrip — operadores órfãos no FIM (mediadores que perderam o termo
    #   direito); removemos ASCII + Unicode operators para evitar SYMs
    #   como `[SYM:143·]` (número + op órfão).
    # lstrip — preservamos operadores Unicode inequívocos (`·`, `×`, `÷`)
    #   no início pois são marca A.3 que torna o token simbólico. Sem
    #   eles, `·E` viraria `E` (não simbólico) e perderíamos sinal
    #   matemático. ASCII ambíguos removidos (sinal unário tratado por
    #   classe específica do lexicon).
    _EXTREM_PUNCT_RIGHT = " \t.,;:!?*/+-^=≈≤≥<>·×÷"
    _EXTREM_PUNCT_LEFT = " \t.,;:!?*/^=≈≤≥<>"

    for w_start, w_end in word_spans:
        # OPÇÃO A — SUB-ATOMIZAÇÃO de variáveis dentro da palavra-token.
        # Cada VARIÁVEL é um SYM atômico separado. Operadores composicionais
        # (·, +, -, /, *, ×, ÷) e parens entre variáveis ficam passthrough,
        # preservando a estrutura sintática para o LLM/modelo.
        #
        # Ex.: `η·U·I/v` → 4 SYMs: η, U, I, v
        # Ex.: `K_p·e(t)` → 2 SYMs: K_p, e(t)
        # Ex.: `ω²·M)·φ` → 3 SYMs: ω², M, φ (parens órfãos ignorados)
        #
        # Para palavra-token contígua matematicamente, a sub-atomização
        # cobre os casos onde o autor não inseriu whitespace entre vars.
        token = text[w_start:w_end]
        sub_atoms = _atomize_variables(token, base_offset=w_start)
        consumed_any = False
        for sub_start, sub_end in sub_atoms:
            sub_text = text[sub_start:sub_end]
            if is_symbolic_token(sub_text):
                sym_spans.append((sub_start, sub_end))
                consumed_any = True
        if consumed_any:
            continue
        # Se a palavra TEM sub-átomos VAR mas nenhum satisfez is_symbolic_token,
        # confiamos na PROPAGAÇÃO para cada sub-átomo individual. NÃO caímos
        # em ETAPA 2 (que emitiria o segmento inteiro como SYM, contradizendo
        # a Opção A — variáveis atômicas).
        if sub_atoms:
            continue

        # ETAPA 2 (fallback): palavra-token SEM sub-átomos VAR — ex.: operadores
        # puros, números soltos. Subdivide em torno de átomos Camada 1 e testa
        # cada segmento como SYM.
        for seg_start, seg_end in _split_word_around_layer1(
            w_start, w_end, layer1_sorted
        ):
            segment = text[seg_start:seg_end]
            stripped = segment.rstrip(_EXTREM_PUNCT_RIGHT).lstrip(_EXTREM_PUNCT_LEFT)
            stripped = _balance_brackets_at_edges(stripped)
            if not stripped:
                continue
            offset = segment.find(stripped)
            real_start = seg_start + offset
            real_end = real_start + len(stripped)
            if is_symbolic_token(stripped):
                sym_spans.append((real_start, real_end))

    return sym_spans


def find_propagated_symbols(
    text: str,
    layer1_spans: list[tuple[int, int]],
    layer1_is_math: list[bool],
    sym_atomic_spans: list[tuple[int, int]],
    layer1_tipos: list[str] | None = None,
) -> list[tuple[int, int]]:
    """A.6 — propagação contextual.

    Letras isoladas (1-3 chars latino/grego) em palavras-token de prosa
    pura adquirem status SYM quando IMEDIATAMENTE adjacentes (gap
    whitespace-only) a entidade matemática reconhecida:
        - átomo Camada 1 cujo tipo é matemático (não QTY pura — QTY já
          é entidade canônica de interesse próprio; OP/IDX/CONST/SYM são
          ativadores válidos)
        - padrão SYM atômico já identificado

    Args:
        text: texto cru.
        layer1_spans: spans Camada 1 ordenados.
        layer1_is_math: paralelo a layer1_spans; True se o átomo é
            entidade ativadora de propagação (OP/IDX/CONST). QTY pura
            não ativa (variável adjacente a QTY tipicamente é também
            variável: `l = 5 m` → l propagado por `=` operador, não por
            `5 m`; cobertura redundante).
        sym_atomic_spans: SYM atômicos já identificados (marcas A.1-A.5).

    Returns:
        Lista (start, end) de letras isoladas promovidas a SYM.
    """
    word_spans = tokenize_into_word_tokens(text)

    # Construir lista combinada de "ativadores": Camada 1 math + SYM atomic
    activators: list[tuple[int, int]] = []
    for (s, e), is_math in zip(layer1_spans, layer1_is_math):
        if is_math:
            activators.append((s, e))
    activators.extend(sym_atomic_spans)
    activators.sort()

    propagated: list[tuple[int, int]] = []

    # Indexa por span → tipo (para A.6.2 — definição lhs/rhs)
    if layer1_tipos is None:
        layer1_tipos = ["?"] * len(layer1_spans)
    tipo_by_span = dict(zip(layer1_spans, layer1_tipos))

    layer1_sorted = sorted(layer1_spans)

    def _overlaps_layer1(s: int, e: int) -> bool:
        return any(ps < e and s < pe for ps, pe in layer1_sorted)

    def _connected_via_text_mediator(start: int, end: int) -> bool:
        """A.6.4 — letra latina conectada a ativador via mediador textual.

        Diferente de A.6.2/A.6.3 que exigem OperadorFormal no layer1,
        esta regra detecta operadores composicionais ASCII (`+`, `-`,
        `*`, `/`) DIRETAMENTE no texto bruto via is_operator_mediator,
        conectando a letra a SYM/QTY/ConstanteUniversal do outro lado.

        Princípio ontológico: `K` em `(K - ω²·M)·φ` é variável porque
        está em expressão matemática composta — mediador `-` conecta
        a `ω²·M` que é SYM. Operadores ASCII composicionais não viram
        OperadorFormal por ambiguidade com prosa, mas dentro de
        expressão (vizinhos a SYM/QTY) são inequivocamente matemáticos.
        """
        # Pequeno raio de busca: até MAX_GAP chars (whitespace + op + whitespace + …)
        # para evitar falsos positivos em prosa distante.
        MAX_GAP = 6  # cobre " - " com folga; bloqueia "K e a flecha"
        # Procura ativador à DIREITA (start_act > end)
        for a_start, a_end in activators + sym_atomic_spans:
            if a_start < end:
                continue
            gap = text[end:a_start]
            if len(gap) > MAX_GAP:
                continue
            if not gap.strip():
                # Ativador IMEDIATO sem operador — já tratado em A.6.1/A.6.3
                continue
            if is_operator_mediator(gap):
                return True
        # Procura ativador à ESQUERDA (a_end <= start)
        for a_start, a_end in activators + sym_atomic_spans:
            if a_end > start:
                continue
            gap = text[a_end:start]
            if len(gap) > MAX_GAP:
                continue
            if not gap.strip():
                continue
            if is_operator_mediator(gap):
                return True
        return False

    def _adjacent_to_unicode_op(start: int, end: int) -> bool:
        """True se o PRIMEIRO char não-whitespace vizinho é math Unicode.

        Consulta direta do UCD via is_math_unicode_char — independente do
        operador estar no lexicon OP da Camada 1. Cobre ∀ ∃ ∈ ∧ ∨ ⇒ ⊂
        ⊃ ∪ ∩ ≈ uniformemente, mesmo os que faltem no lexicon.
        """
        # Vizinho à DIREITA: pula whitespace e olha primeiro char
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and is_math_unicode_char(text[cursor]):
            return True
        # Vizinho à ESQUERDA: pula whitespace e olha último char
        cursor = start - 1
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        if cursor >= 0 and is_math_unicode_char(text[cursor]):
            return True
        return False

    def _adjacent_to_activator(start: int, end: int) -> bool:
        for a_start, a_end in activators:
            # Vizinho à esquerda (gap entre activator.end e word.start é whitespace)
            if a_end <= start:
                gap = text[a_end:start]
                if gap == "" or gap.strip() == "":
                    return True
            # Vizinho à direita (gap entre word.end e activator.start é whitespace)
            if end <= a_start:
                gap = text[end:a_start]
                if gap == "" or gap.strip() == "":
                    return True
        return False

    # Padrão de número puro (literal numérico): suporta sinal, decimal PT/EN,
    # notação científica. Aceita como RHS/LHS de definição matemática
    # — admite adimensionais como `R = 3,0`, `n = 5`, `b = -3`.
    _NUM_LITERAL_RE = re.compile(
        r"^\s*[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?\s*$"
    )

    def _gap_is_number(s: int, e: int) -> bool:
        """True se o slice [s:e] é um número puro (literal numérico)."""
        if s >= e:
            return False
        return bool(_NUM_LITERAL_RE.match(text[s:e]))

    # Padrão de "filler" entre `=` e o RHS: whitespace, sinais, parens/
    # brackets/braces (abertos OU fechados), dígitos, operadores composicionais
    # Cobre `Q = -K·A·...`, `q = -(2·π·...)`, `f = 1/(2·π·R·C)`,
    # `D = (1/2)·ρ·...`, `Q = (1/n)·A·R_h^(2/3)·S^(1/2)`.
    # Janela: até N chars após `=` (suficiente para expressões médias),
    # parando em quebra forte (`\n`, `;`) ou fim de sentence (`. ` + maiúscula).
    _MAX_RHS_WINDOW = 120
    _STRONG_BREAK_RE = re.compile(r"[\n;]|\.\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])")

    def _is_definition_config(start: int, end: int) -> bool:
        """A.6.2 — letra latina em config `<letra> = <RHS>` ou `<LHS> = <letra>`.

        RHS/LHS aceitos como ativadores matemáticos:
            - GrandezaFisica (QTY com unidade) — caso clássico `L = 1,80 m`
            - ExpressaoSimbolica (SYM já identificado) — caso `H = η·U·I/v`
            - ConstanteUniversal
            - NÚMERO PURO (literal numérico) — adimensionais `R = 3,0`, `n = 5`

        Princípio: se há ativador matemático DENTRO da janela RHS
        (até _MAX_RHS_WINDOW chars após `=`, sem quebra forte), a config
        é matemática. NÃO testamos filler char-a-char — variáveis
        intermediárias (`(1/n)·...`) NÃO bloqueiam a propagação porque
        elas próprias são ativadores.

        O operador adjacente deve ser OperadorFormal mediador (relacional típico).
        """
        def _is_math_activator(s: int, e: int) -> bool:
            tipo = tipo_by_span.get((s, e), "?")
            return tipo in (
                "GrandezaFisica",
                "ExpressaoSimbolica",
                "ConstanteUniversal",
            )

        def _window_clean(seg: str) -> bool:
            """Janela é matematicamente contínua (sem quebra forte)."""
            return _STRONG_BREAK_RE.search(seg) is None

        # Vizinho à DIREITA: <letra> = <RHS-ativador> ... dentro da janela
        for op_s, op_e in layer1_spans:
            if tipo_by_span.get((op_s, op_e), "?") != "OperadorFormal":
                continue
            if op_s < end:
                continue
            gap_left = text[end:op_s]
            if gap_left and gap_left.strip() != "":
                continue
            op_text = text[op_s:op_e]
            if not is_operator_mediator(op_text):
                continue
            win_end = min(len(text), op_e + _MAX_RHS_WINDOW)
            # 1. Ativador Camada 1 dentro da janela
            for q_s, q_e in layer1_spans:
                if not _is_math_activator(q_s, q_e):
                    continue
                if op_e <= q_s < win_end and _window_clean(text[op_e:q_s]):
                    return True
            # 2. SYM atômico dentro da janela
            for q_s, q_e in sym_atomic_spans:
                if op_e <= q_s < win_end and _window_clean(text[op_e:q_s]):
                    return True
            # 3. RHS é NÚMERO PURO imediato (adimensionais)
            after = op_e
            while after < len(text) and text[after].isspace():
                after += 1
            num_end = after
            while num_end < len(text) and text[num_end] in "0123456789.,-+eE":
                num_end += 1
            if num_end > after and _gap_is_number(after, num_end):
                return True
            break

        # Vizinho à ESQUERDA: <LHS-ativador> = <letra>
        for op_s, op_e in layer1_spans:
            if tipo_by_span.get((op_s, op_e), "?") != "OperadorFormal":
                continue
            if op_e > start:
                continue
            gap_right = text[op_e:start]
            if gap_right and gap_right.strip() != "":
                continue
            op_text = text[op_s:op_e]
            if not is_operator_mediator(op_text):
                continue
            for q_s, q_e in layer1_spans:
                if not _is_math_activator(q_s, q_e):
                    continue
                if q_e > op_s:
                    continue
                gap_left2 = text[q_e:op_s]
                if gap_left2 and gap_left2.strip() != "":
                    continue
                return True
            for q_s, q_e in sym_atomic_spans:
                if q_e > op_s:
                    continue
                gap_left2 = text[q_e:op_s]
                if gap_left2 and gap_left2.strip() != "":
                    continue
                return True
            # LHS é número puro
            before = op_s - 1
            while before >= 0 and text[before].isspace():
                before -= 1
            num_start = before
            while num_start >= 0 and text[num_start] in "0123456789.,-+eE":
                num_start -= 1
            if before > num_start and _gap_is_number(num_start + 1, before + 1):
                return True
            break
        return False

    # ITERA SUB-ÁTOMOS, não word-tokens inteiros.
    # Variáveis dentro de palavras-token contíguas (`U`, `I`, `v` em
    # `η·U·I/v`) precisam ser candidatos individuais à propagação.
    sub_atom_candidates: list[tuple[int, int]] = []
    for w_start, w_end in word_spans:
        token = text[w_start:w_end]
        sub_atoms = _atomize_variables(token, base_offset=w_start)
        if sub_atoms:
            sub_atom_candidates.extend(sub_atoms)
        else:
            # Palavra-token sem sub-átomos VAR (ex.: número puro `5`,
            # operador puro `∪`): mantém o token inteiro como candidato.
            sub_atom_candidates.append((w_start, w_end))

    for real_start, real_end in sub_atom_candidates:
        if _overlaps_layer1(real_start, real_end):
            continue
        token_stripped = text[real_start:real_end].rstrip(".,;:!?").lstrip(".,;:!?()")
        if not token_stripped:
            continue
        # Já capturado por marca intrínseca?
        if is_symbolic_token(token_stripped):
            continue
        # Já está nos sym_atomic_spans?
        if any(real_start == s and real_end == e for s, e in sym_atomic_spans):
            continue
        # Critério A.6: propagação contextual — múltiplos regimes.
        #
        # A.6.1 (geral) — APENAS letras gregas isoladas adjacentes a
        # qualquer ativador matemático. Letras gregas (σ, α, π, etc.)
        # não aparecem em prosa natural PT-BR → adjacência a contexto
        # math é sinal inequívoco.
        #
        # A.6.2 (def matemática) — qualquer TOKEN ALFABÉTICO (1 letra ou
        # palavra inteira) configurado como `<token> = <QTY>` ou
        # `<QTY> = <token>`. Operador relacional adjacente E lado
        # oposto é ativador matemático (QTY/SYM/CONST/número puro).
        # Sintaxe declarativa de definição: se engenheiro escreveu
        # `<palavra> = <grandeza>` está NOMEANDO a grandeza — vale tanto
        # para `L = 1,80 m` (1 char) quanto `largura = 1,80 m` (palavra).
        # Restrição: token deve ser ALFABÉTICO PURO (Unicode `\p{L}` +
        # subscript `_` + primes `'″‴`) — exclui mistura com dígitos ou
        # pontuação que indicaria outro tipo de entidade.
        #
        # A.6.3 / A.6.4 (geral) — letras latinas curtas (1-4 chars)
        # adjacentes a operador Unicode não-ASCII OU conectadas via
        # mediador textual. Limite restrito para evitar falso positivo
        # em prosa natural longa.
        if not token_stripped:
            continue
        # Chars aceitos em qualquer regime de propagação:
        #   - letras (qualquer alfabeto Unicode)
        #   - `_` subscript
        #   - primes notação Lagrange (`'`, `″`, `‴`) — derivada de função
        if not all(c.isalpha() or c in "_'″‴" for c in token_stripped):
            continue
        has_greek = bool(_GREEK_RE.search(token_stripped))
        if has_greek:
            # A.6.1: letra matemática Unicode — basta adjacência a ativador.
            # Aceita qualquer tamanho (palavras gregas inteiras improváveis
            # em prosa PT-BR, mas inofensivas se aparecerem).
            if _adjacent_to_activator(real_start, real_end):
                propagated.append((real_start, real_end))
            continue
        # A.6.2: token alfabético em config def matemática
        # `<token> = QTY/SYM/num` — SEM limite de comprimento.
        # Sintaxe declarativa sobrepõe heurística lexical.
        if _is_definition_config(real_start, real_end):
            propagated.append((real_start, real_end))
            continue
        # Para A.6.3 e A.6.4, manter limite estrito para evitar falso
        # positivo em prosa natural longa.
        if not (1 <= len(token_stripped) <= 4):
            continue
        # A.6.3: latina ADJACENTE A OPERADOR UNICODE NÃO-ASCII (lógica,
        # conjuntos, cálculo). `∀x`, `x ∈ ℝ`, `p ∧ q`, `f: ℝ → ℝ`.
        # Operadores Unicode `\\p{Sm}` são inequivocamente matemáticos —
        # letra latina ao lado é sempre variável/proposição.
        if _adjacent_to_unicode_op(real_start, real_end):
            propagated.append((real_start, real_end))
            continue
        # A.6.4: latina conectada a SYM/QTY/CONST via MEDIADOR TEXTUAL.
        # Caso `(K - ω²·M)·φ`: K adjacente a ω²·M via gap ` - ` (ws +
        # ASCII op + ws). Operadores composicionais ASCII `+ - * /` não
        # estão na Camada 1 mas SÃO mediadores matemáticos no contexto.
        # is_operator_mediator(gap) detecta isso sobre texto bruto.
        if _connected_via_text_mediator(real_start, real_end):
            propagated.append((real_start, real_end))

    return propagated


def find_clusters_from_atoms(
    text: str,
    atom_spans: list[tuple[int, int]],
    atom_is_operator: list[bool] | None = None,
    atom_is_quantity: list[bool] | None = None,
) -> list[tuple[int, int]]:
    """Cluster derivado: ≥2 termos-símbolo mediados por operador binário.

    Termo-símbolo = átomo não-OP e não-QTY (i.e., IDX, CONST, SYM ou
    palavra-token simbólica propagada). QTYs puras (grandezas físicas com
    unidade canônica) NÃO contam — preservadas separadas para o LLM
    extrair valores individuais (`30 cm × 40 cm` permanece como dois QTYs).

    Args:
        text: texto cru.
        atom_spans: spans de todos os átomos da Camada 1 + tokens SYM
            (atomic + propagados) ordenados por start.
        atom_is_operator: paralelo; True se OperadorFormal.
        atom_is_quantity: paralelo; True se GrandezaFisica.

    Returns:
        Lista de (cluster_start, cluster_end).
    """
    n = len(atom_spans)
    if n < 2:
        return []
    if atom_is_operator is None:
        atom_is_operator = [False] * n
    if atom_is_quantity is None:
        atom_is_quantity = [False] * n

    clusters: list[tuple[int, int]] = []
    i = 0
    while i < n:
        cluster_start = atom_spans[i][0]
        cluster_end = atom_spans[i][1]
        term_count = 0 if atom_is_operator[i] else 1
        symbolic_count = 0
        if not atom_is_operator[i] and not atom_is_quantity[i]:
            symbolic_count = 1
        has_compositional = False
        j = i + 1
        while j < n:
            prev_end = atom_spans[j - 1][1]
            next_start = atom_spans[j][0]
            gap = text[prev_end:next_start]

            # Cluster ESTENDE SOMENTE via operador COMPOSICIONAL.
            # Operadores RELACIONAIS (=, ≈, ≠, ≤, ≥, ⇒, ⇔) NÃO estendem cluster
            # — eles SEPARAM LHS de RHS (definição/comparação). Resultado:
            # `H = η·U·I/v = 396 J/mm` produz 3 entidades atômicas separadas
            # `[SYM:H] = [SYM:η·U·I/v] = [QTY ...]` em vez de um cluster único.
            extends = False
            mediator_is_compositional = False
            if is_compositional_operator(gap):
                extends = True
                mediator_is_compositional = True
            elif gap.strip() == "":
                # gap só whitespace: estende SE um lado é OP COMPOSICIONAL capturado
                if atom_is_operator[j - 1] or atom_is_operator[j]:
                    op_idx = j - 1 if atom_is_operator[j - 1] else j
                    op_text = text[atom_spans[op_idx][0]:atom_spans[op_idx][1]]
                    if is_compositional_operator(op_text):
                        extends = True
                        mediator_is_compositional = True

            if extends:
                cluster_end = atom_spans[j][1]
                if not atom_is_operator[j]:
                    term_count += 1
                if not atom_is_operator[j] and not atom_is_quantity[j]:
                    symbolic_count += 1
                if mediator_is_compositional:
                    has_compositional = True
                j += 1
            else:
                break

        # Cluster válido exige TRÊS condições:
        #   1. ≥2 termos (não-OP)
        #   2. ≥2 termos simbólicos (não-QTY, não-OP)
        #   3. ≥1 mediador COMPOSICIONAL
        #
        # Princípio ontológico:
        #   - `σ_x + σ_y`: 2 IDX + compositional → cluster (expressão)
        #   - `σ_y = 215 MPa`: 1 IDX + rel + 1 QTY → preserva (definição)
        #   - `fck = C25`: 2 IDX + rel SEM compositional → preserva (definição)
        #   - `30 cm × 40 cm`: 0 simbólico → preserva (regra QTY-pura)
        if term_count >= 2 and symbolic_count >= 2 and has_compositional:
            clusters.append((cluster_start, cluster_end))
        i = j if j > i else i + 1
    return clusters


def derive_sym_regions(
    text: str,
    layer1_spans: list[tuple[int, int]],
    layer1_tipos: list[str],
) -> list[tuple[int, int, bool]]:
    """Pipeline unificado da Camada 2: emerge TODAS as regiões SYM.

    Algoritmo:
        1. SYM ATÔMICO DELIMITADO — padrões com estrutura sintática
           inequívoca (funções, LaTeX, integrais, valor absoluto, etc.).
        2. SYM ATÔMICO POR MARCA INTRÍNSECA — palavras-token com
           marcas A.1-A.5 (subscript, grego, operador Unicode, dígito
           justaposto a letra).
        3. SYM POR PROPAGAÇÃO (A.6) — letras isoladas adjacentes a
           ativador matemático (OP, IDX, CONST, SYM já identificado).
        4. CLUSTER DERIVADO — ≥2 termos-símbolo (incluindo tokens das
           etapas 2 e 3) mediados por operador.

    Args:
        text: texto cru.
        layer1_spans: spans da Camada 1, ordenados.
        layer1_tipos: tipo de cada átomo Camada 1 (string nome do tipo).

    Returns:
        Lista (start, end, atomic_bool) sem overlap interno.
    """
    layer1_is_math = [
        t in ("OperadorFormal", "IdentificadorTecnico", "ConstanteUniversal")
        for t in layer1_tipos
    ]
    layer1_is_operator = [t == "OperadorFormal" for t in layer1_tipos]
    layer1_is_quantity = [t == "GrandezaFisica" for t in layer1_tipos]

    # Etapa 1: delimitados (absorventes)
    delimited = find_delimited_sym_patterns(text, protected_atom_spans=None)

    # Etapa 2: marca intrínseca em palavras-token
    intrinsic = find_sym_tokens(text, layer1_spans)

    # Etapa 2.5: variáveis em listas matemáticas `[X, Y]`, `(x, y)`, `{p, q}`
    list_vars = _find_list_var_spans(text)

    # Combinar com delimitados como SYM atômicos (sem overlap)
    sym_atomic = _dedupe_overlaps(delimited + intrinsic + list_vars, prefer_longer=True)

    # Etapa 3: propagação em FIXED-POINT (até estabilizar).
    # Passe N+1 vê SYMs do passe N como ativadores → captura cadeias como
    # `x + y = 0` onde y vira SYM no passe 1 (via `y = 0`) e x vira SYM
    # no passe 2 (adjacente via `+` a SYM `y`).
    sym_atomic_with_propagated = list(sym_atomic)
    for _ in range(3):  # 3 passes máx; converge em ≤2 passes na prática
        propagated = find_propagated_symbols(
            text, layer1_spans, layer1_is_math, sym_atomic_with_propagated,
            layer1_tipos=layer1_tipos,
        )
        new_count = len(sym_atomic_with_propagated)
        sym_atomic_with_propagated = _dedupe_overlaps(
            sym_atomic_with_propagated + propagated, prefer_longer=True
        )
        if len(sym_atomic_with_propagated) == new_count:
            break  # ponto fixo atingido

    # Etapa 4: clusters derivados
    # Construir lista combinada de "átomos" para clusterização:
    # Camada 1 atoms + SYM atomic (todos como termos-símbolo não-OP).
    combined_spans = list(layer1_spans)
    combined_is_op = list(layer1_is_operator)
    combined_is_qty = list(layer1_is_quantity)
    for s, e in sym_atomic_with_propagated:
        combined_spans.append((s, e))
        combined_is_op.append(False)
        combined_is_qty.append(False)
    # Ordenar por start
    triples = sorted(
        zip(combined_spans, combined_is_op, combined_is_qty),
        key=lambda t: t[0][0],
    )
    combined_spans = [t[0] for t in triples]
    combined_is_op = [t[1] for t in triples]
    combined_is_qty = [t[2] for t in triples]

    # OPÇÃO A — cluster greedy DESABILITADO. Variáveis ficam atômicas
    # como SYMs separados. Operadores entre elas (·, +, -, /) ficam
    # passthrough, preservando estrutura sintática explícita para o
    # modelo/LLM. Compostos sintaticamente coesos (funções aplicadas,
    # integrais, derivadas) continuam emergindo como SYM ATÔMICO via
    # `find_delimited_sym_patterns` (Etapa 1).
    clusters: list[tuple[int, int]] = []

    # Combinar resultados: somente SYM atomic (sub-atomizados).
    candidates: list[tuple[int, int, bool]] = []
    for s, e in sym_atomic_with_propagated:
        candidates.append((s, e, True))

    # Longest-match dentro de SYM (atomic preferido em empate, depois start asc)
    candidates.sort(key=lambda c: (-(c[1] - c[0]), 0 if c[2] else 1, c[0]))
    accepted: list[tuple[int, int, bool]] = []
    for cand in candidates:
        s, e, _ = cand
        if any(ps < e and s < pe for ps, pe, _ in accepted):
            continue
        accepted.append(cand)
    accepted.sort(key=lambda c: c[0])
    return accepted


def _dedupe_overlaps(
    spans: list[tuple[int, int]],
    prefer_longer: bool = True,
) -> list[tuple[int, int]]:
    """Remove overlaps mantendo preferencialmente os mais longos."""
    if not spans:
        return []
    if prefer_longer:
        spans = sorted(set(spans), key=lambda s: (-(s[1] - s[0]), s[0]))
    else:
        spans = sorted(set(spans), key=lambda s: (s[0], -(s[1] - s[0])))
    accepted: list[tuple[int, int]] = []
    for s, e in spans:
        if any(ps < e and s < pe for ps, pe in accepted):
            continue
        accepted.append((s, e))
    accepted.sort()
    return accepted


# ---------------------------------------------------------------------------
# COMPATIBILIDADE — APIs antigas mantidas como wrappers finos
# (para não quebrar classifier.py durante a transição)
# ---------------------------------------------------------------------------


def find_atomic_sym_patterns(
    text: str,
    protected_atom_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """[DEPRECATED] Use derive_sym_regions.

    Mantido para compatibilidade temporária; retorna apenas patterns
    delimitados (sem propagação nem marca intrínseca).
    """
    return find_delimited_sym_patterns(text, protected_atom_spans)


def derive_sym_from_atoms(
    text: str,
    atom_spans: list[tuple[int, int]],
    atomic_patterns: list[tuple[int, int]] | None = None,
    atom_is_operator: list[bool] | None = None,
    atom_is_quantity: list[bool] | None = None,
) -> list[tuple[int, int, bool]]:
    """[DEPRECATED] API antiga. Use derive_sym_regions com tipos.

    Wrapper de compatibilidade: combina clusters + patterns atômicos
    sem a etapa de propagação contextual.
    """
    if atomic_patterns is None:
        atomic_patterns = []

    cluster_spans = find_clusters_from_atoms(
        text, atom_spans, atom_is_operator, atom_is_quantity
    )
    candidates: list[tuple[int, int, bool]] = []
    candidates.extend((s, e, False) for s, e in cluster_spans)
    candidates.extend((s, e, True) for s, e in atomic_patterns)

    candidates.sort(key=lambda c: (0 if c[2] else 1, -(c[1] - c[0]), c[0]))
    accepted: list[tuple[int, int, bool]] = []
    for cand in candidates:
        s, e, _ = cand
        if any(ps < e and s < pe for ps, pe, _ in accepted):
            continue
        accepted.append(cand)
    accepted.sort(key=lambda c: c[0])
    return accepted
