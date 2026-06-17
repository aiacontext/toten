"""ExpressaoSimbolica — 7º tipo da OEE.

**Definição ontológica composicional.** Uma ExpressaoSimbolica é
**composição mediada** (princípio P3) de:

- IdentificadorTecnico (símbolos de variável: `p`, `l`, `σ_pa`)
- OperadorFormal (`/`, `·`, `^`, `√`, `∛`, parens, sup/sub)
- Número literal

…ao longo de uma sequência contígua sem prosa intermediária, sujeita a:

- **Princípio P3 (composição mediada)**: cluster contém ≥1 operador visível,
  transição num↔letra, OU subscript.
- **Princípio P6 (convenção tipográfica)**: clusters alfabéticos ≤4 chars,
  OU contêm `_`, OU contêm letra grega.
- **Princípio P8 (marca matemática)**: presença distintiva de dígito, sub,
  sup, parens, operador Unicode ou símbolo de cálculo.
- **dim_table veto**: cluster com APENAS átomos SI é unidade composta
  (QTY), não SYM.

**Atomicidade real: token preserva a STRING ORIGINAL do engenheiro.**

Decisão de design (2026-05-20): SymPy ABANDONADO do tokenizer. O
objetivo do Modo B é PRESERVAR a notação do autor para o LLM frozen
consumir, não canonizar. SymPy é parser algébrico — força
`V_c(t) → V_c*t`, `ΔT → Δ*T`, `(pl²/32)·∛2 → p*l**2*2**(1/3)/32` —
destruindo fidelidade textual e introduzindo erros semânticos
(função aplicada vs produto, variável composta vs multiplicação).
Canonicalização algébrica é trabalho do grader (Trilha E), não do
tokenizer. O LLM lê a forma original perfeitamente.

Resultado: tokens fiéis ao texto:
- `pl/12` → `[SYM:pl/12]` (literal)
- `V_c(t)` → `[SYM:V_c(t)]` (preserva função aplicada)
- `ΔT/R_total` → `[SYM:ΔT/R_total]` (preserva prefixo diferencial)
- `(pl²/32)·∛2` → `[SYM:(pl²/32)·∛2]` (preserva radical Unicode)
- `dM/dx` → `[SYM:dM/dx]` (preserva notação diferencial)
- `∫₀ˡ q(x) dx` → `[SYM:∫₀ˡ q(x) dx]` (preserva integral completa)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import regex as re

from toten.classifier.region import Region
from toten.dimensional.table import default_dim_table
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# =============================================================================
# Constantes ontológicas — derivadas, não enumeradas
# =============================================================================
#
# Princípio de design: evitar listas paralelas que duplicam o critério.
# Cada constante abaixo representa uma categoria ONTOLÓGICA distinta:
#
# - _GREEK: alfabeto grego (notação técnica universal)
# - _VISIBLE_OPERATORS: operadores composicionais P3 (composição mediada)
# - _SUB_SUP_UNICODE: subscript/superscript Unicode (notação tipográfica)
# - _CALC_OPERATORS: operadores de cálculo Unicode (∫∑∂∇∏∮√∛∜)
# - _RELATIONAL_OPERATORS: relações Unicode (≈≠≡≤≥⇒⇔→↔∝)
#
# P8 (marca matemática) é DERIVADO destas categorias + estruturas
# reconhecíveis (subscript ASCII, função aplicada, derivada, integral) —
# NÃO é uma lista separada que requer manutenção.

_GREEK_LOWER = "αβγδεζηθικλμνξοπρςστυφχψω"
_GREEK_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
_GREEK = frozenset(_GREEK_LOWER + _GREEK_UPPER)

# Operadores composicionais visíveis (P3). Inclui sup/sub Unicode (`x²` =
# potenciação implícita, operação composicional). NÃO inclui `-` ASCII
# (ambíguo com hífen de identificador).
_VISIBLE_OPERATORS = frozenset(
    "/·*^()√∛∜×÷"
    "²³⁰¹⁴⁵⁶⁷⁸⁹⁻⁺"     # sup Unicode (potenciação)
    "₀₁₂₃₄₅₆₇₈₉"     # sub Unicode (índice)
    "|⟨⟩"             # delimitadores Dirac/abs (matemática quântica/abs)
    "{}"               # delimitadores LaTeX subscript composto (k|k-1)
    "\\"               # backslash LaTeX (comando markup matemático)
    # Relacionais Unicode (≈ ≠ ≡ ≤ ≥ → ⇒) NÃO entram aqui — são
    # OPERADORES FORMAIS (separadores de equação LHS/RHS), capturados
    # como tipo OPERADOR_FORMAL distinto. Decisão ontológica: relacional
    # ≠ operador composicional.
)

# Sub/sup Unicode (notação tipográfica matemática intrínseca)
_SUB_SUP_UNICODE = frozenset("²³⁰¹⁴⁵⁶⁷⁸⁹⁻⁺₀₁₂₃₄₅₆₇₈₉")

# Operadores de cálculo Unicode (estruturas matemáticas reconhecíveis)
_CALC_OPERATORS = frozenset("∫∑∂∇∏∮")

# Operadores relacionais Unicode (presença em equação)
_RELATIONAL_OPERATORS = frozenset("≈≠≡≤≥⇒⇔→↔∝")

# Marcadores romanos comuns em listas (não viram SYM mesmo em parens).
_ROMAN_NUMERAL_TOKENS = frozenset({
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "a", "b", "c", "d", "e", "f", "g", "h",  # marcadores letra única
})


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolicExpression:
    """Composição simbólica reconhecida ontologicamente.

    Atomicidade textual: emite `[SYM:<original>]` análogo a `[IDX:slug]`.
    Conteúdo é a forma LITERAL do autor (engenheiro), sem canonicalização.
    """

    original: str
    free_vars: frozenset[str]   # variáveis livres extraídas do texto

    def render_b(self) -> str:
        """Modo B atômico: `[SYM:<original_sem_espacos>]`.

        Remove apenas espaços para preservar atomicidade textual (BPE não
        fragmenta em espaços internos do token), mantendo todos os
        operadores, subscript, sup/sub Unicode, parens etc. do autor.
        """
        canonical = self.original.replace(" ", "").rstrip(".,;:")
        return f"[SYM:{canonical}]"

    @property
    def text(self) -> str:
        """Alias uniforme — mesma interface dos outros Tokens (QTY, IDX, ...)."""
        return self.render_b()


# ---------------------------------------------------------------------------
# Critérios construtivos (P3 + P6 + P8 da OEE)
# ---------------------------------------------------------------------------


# Padrões estruturais (usando `\p{L}\p{M}` Unicode letter category — substitui
# enumeração latim/grego/acentuado/Ø). Derivada: d<var>/d<var> ou ∂<var>/∂<var>.
# Integral: ∫ ... d<var>. Função aplicada: <letra>(<arg>).
# Regexes estruturais — `\p{L}` letras, `\p{M}` marks combinatórias (Ĥ, σ̂),
# `\p{N}` números. Letras de variável aceitam marks combinatórias após.
_DERIVATIVE_RE = re.compile(
    r"(?:\bd|∂)[²³⁴]?[\p{L}\p{M}][_\p{L}\p{M}\p{N}]*\s*/\s*(?:d|∂)[\p{L}\p{M}][_\p{L}\p{M}\p{N}]*[²³⁴]?"
)
_INTEGRAL_PATTERN_RE = re.compile(r"∫.*?\bd[\p{L}\p{M}]\b")
_FUNCTION_APPLICATION_RE = re.compile(r"[\p{L}\p{M}]\(")
_NUM_LETTER_TRANSITION_RE = re.compile(r"\d\s?[\p{L}\p{M}]|[\p{L}\p{M}]\s?\d")
_LETTER_CLUSTER_RE = re.compile(r"[\p{L}\p{M}_]+")


def _satisfies_p3_composicao_mediada(text: str) -> bool:
    """P3: composição mediada — cluster tem operador composicional OU
    transição num↔letra OU subscript ASCII (qualquer mediador estrutural)."""
    if any(c in _VISIBLE_OPERATORS for c in text):
        return True
    if "_" in text:
        return True
    return bool(_NUM_LETTER_TRANSITION_RE.search(text))


def _satisfies_p6_convencao_tipografica(text: str) -> bool:
    """P6: cada cluster alfabético contíguo tem ≤4 chars OU `_` OU grego
    OU é função aplicada (seguido de `(`) OU é comando LaTeX (precedido por `\\`).

    Estruturas sintáticas matemáticas inequívocas fazem bypass P6:
    - Função aplicada `arccos(x)`, `sqrt(a+b)`, `tanh(t)` — clusters >4 chars
      reconhecíveis pela aplicação a parens.
    - Comando LaTeX `\\alpha`, `\\nabla`, `\\partial`, `\\sigma` — clusters >4
      chars precedidos por backslash, markup matemático inequívoco.
    """
    for m in _LETTER_CLUSTER_RE.finditer(text):
        word = m.group()
        start_pos = m.start()
        end_pos = m.end()
        # Comando LaTeX: cluster precedido por backslash
        if start_pos > 0 and text[start_pos - 1] == "\\":
            continue
        # Função aplicada: cluster imediatamente seguido de `(`
        if end_pos < len(text) and text[end_pos] == "(":
            continue
        if "_" in word or any(c in _GREEK for c in word):
            continue
        if len(word) > 4:
            return False
    return True


# Operadores Unicode INEQUIVOCAMENTE matemáticos (P8). ASCII `/`, `*`, `^`
# ficam DE FORA — são ambíguos com prosa (`e/ou`, `a/b`, `*nota*`). Operadores
# Unicode (`·`, `×`, `÷`, `√`, `∛`) só aparecem em notação matemática.
_UNAMBIGUOUS_MATH_OPERATORS = frozenset("·×÷√∛∜")

# Parens balanceadas com operador aritmético interno: estrutura
# inequivocamente matemática `(a+b)`, `(x-y)`, `(p/q)`, `(m*n)`.
_PAREN_WITH_OP_RE = re.compile(r"\([^()]*[+\-*/^][^()]*\)")


def _has_mathematical_mark(text: str) -> bool:
    """P8: marca matemática INEQUÍVOCA — complemento de P3.

    Princípio: P3 (composição mediada) e P8 (marca matemática) são
    COMPLEMENTARES. P3 detecta composição via qualquer operador visível,
    incluindo `/`, `*`, `^`, `(` ASCII que são AMBÍGUOS (`e/ou`, `a*b` em
    prosa). P8 exige marca INEQUIVOCAMENTE matemática que distingue notação
    técnica de prosa natural.

    Marcas inequívocas (qualquer uma é suficiente):
    1. Subscript ASCII `_X` — notação tipográfica matemática.
    2. Sub/sup Unicode `²³₁₂` — não aparece em prosa.
    3. Letra grega — alfabeto técnico.
    4. Operador matemático Unicode (`·×÷√∛`) — exclusivamente notacional.
    5. Operador relacional/cálculo Unicode (`≈≠≤≥∫∑∂∇`).
    6. Dígito — justaposto a letra ou em composição implica notação.
    7. Função aplicada `<letra>(` — estrutura sintática matemática.
    8. Estruturas reservadas: derivada `d/d<var>`, integral `∫...d<var>`.

    Operadores ASCII ambíguos (`/`, `*`, `^`, `+`, `-`, `(`, `)`) NÃO contam
    sozinhos — eles satisfazem P3 mas exigem P8 corroborante (alguma das
    marcas acima) para confirmar natureza matemática.
    """
    if "_" in text:
        return True
    if "\\" in text:
        # Backslash LaTeX: markup matemático inequívoco (\\alpha, \\frac, \\sum)
        return True
    if any(c in _SUB_SUP_UNICODE for c in text):
        return True
    if any(c in _GREEK for c in text):
        return True
    if any(c in _UNAMBIGUOUS_MATH_OPERATORS for c in text):
        return True
    if any(c in _CALC_OPERATORS for c in text):
        return True
    if any(c in _RELATIONAL_OPERATORS for c in text):
        return True
    if any(ch.isdigit() for ch in text):
        return True
    if _FUNCTION_APPLICATION_RE.search(text):
        return True
    if _PAREN_WITH_OP_RE.search(text):
        return True
    if _DERIVATIVE_RE.search(text):
        return True
    return bool(_INTEGRAL_PATTERN_RE.search(text))


# ---------------------------------------------------------------------------
# dim_table veto: cluster com apenas átomos SI é unidade, não SYM
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _atom_pattern_re() -> re.Pattern:
    """Regex que casa átomos do dim_table com boundary."""
    atoms = sorted(default_dim_table().atoms.keys(), key=lambda a: (-len(a), a))
    pattern = "|".join(re.escape(a) for a in atoms)
    return re.compile(rf"(?<![\w_])(?:{pattern})(?![\w_])", flags=re.UNICODE)


def _extract_free_variables(text: str) -> frozenset[str]:
    """Extrai variáveis livres do texto: clusters alfabéticos.

    Sem SymPy. Captura todos os símbolos alfa (incluindo subscript Unicode/ASCII).
    NÃO filtra átomos do dim_table aqui — letras únicas como `t`, `m`, `s`, `A`
    são átomos no SI mas também aparecem frequentemente como VARIÁVEIS em
    equações de engenharia (tempo, massa, segundo, área). O filtro de
    unidade-composta vs expressão é feito em `_all_alpha_are_atoms` antes.
    """
    return frozenset(
        _FREE_VAR_RE.findall(text)
    )


# Free variable: letras Unicode + dígitos ASCII (sup/sub Unicode são
# operadores P3, não parte do nome da variável). `pl²` → free_var = `pl`.
_FREE_VAR_RE = re.compile(r"[\p{L}](?:[\p{L}\p{M}\d]|_[\p{L}\p{M}\d,]+)*")
_COMPOSITIONAL_OPERATORS_RE = re.compile(r"[·*/^()²³⁰¹⁴⁵⁶⁷⁸⁹⁻⁺₀₁₂₃₄₅₆₇₈₉√∛∜∫∑∂∇∏∮×÷]+")


def _all_alpha_are_atoms(text: str) -> bool:
    """Cluster contém APENAS átomos do dim_table (= unidade composta, não SYM).

    Segmenta o cluster pelos OPERADORES COMPOSICIONAIS de P3 (`·`, `*`, `/`,
    `^`, parens, sup/raiz) e checa se cada segmento resultante é átomo SI.
    Subscript `_` é convenção tipográfica P6 (qualificador), NÃO operador
    composicional — `A_t` é UM segmento (variável `A` qualificada por `t`),
    não dois átomos.

    Exemplos:
        `kg/m²`      → ['kg', 'm'] → ambos átomos → veta (GrandezaFisica)
        `t·m`        → ['t', 'm']  → ambos átomos → veta
        `A_t`        → ['A_t']     → não é átomo SI → não veta
        `σ_y·A_p`    → ['σ_y','A_p'] → nenhum átomo → não veta
        `R_g·T`      → ['R_g','T']  → nenhum átomo (T não é, R_g não é) → não veta
    """
    segments = [s for s in _COMPOSITIONAL_OPERATORS_RE.split(text) if s]
    if not segments:
        return False
    dim_atoms = default_dim_table().atoms
    return all(seg in dim_atoms for seg in segments)


# ---------------------------------------------------------------------------
# Pipeline composicional — sem SymPy
# ---------------------------------------------------------------------------


def try_compose_symbolic(text: str) -> SymbolicExpression | None:
    """Tenta interpretar `text` como ExpressaoSimbolica.

    Critérios derivados de princípios da OEE — SEM SymPy:
    1. Comprimento mínimo (≥3 chars)
    2. P3 (composição mediada): tem op visível OU transição num↔letra OU `_`
    3. P6 (convenção tipográfica): clusters alfa ≤4 OU `_` OU grego
    4. P8 (marca matemática): dígito/sub/sup/parens/op Unicode
    5. dim_table veto: NÃO é unidade composta (kg/m², tf·m)
    6. Tem ≥1 variável livre (não-átomo)

    Atomicidade: preserva string ORIGINAL do autor. Sem fragmentação,
    sem reordenação, sem canonicalização.
    """
    original = text.strip()
    if len(original) < 2:
        return None

    if not _satisfies_p3_composicao_mediada(original):
        return None
    if not _satisfies_p6_convencao_tipografica(original):
        return None
    if not _has_mathematical_mark(original):
        return None

    # dim_table veto: se TODOS os tokens alfa são átomos do dim_table,
    # é unidade composta (GrandezaFisica composta), não SYM.
    if _all_alpha_are_atoms(original):
        return None

    # Rejeitar marcadores de lista: `(a)`, `(ii)`, `(iii)` — original
    # despojado de parens é apenas letras alfa ≤4 chars iguais ou em
    # sequência romana (ii, iii, iv, v, vi, vii, viii, ix, x). Marca
    # matemática `(...)` é falso positivo nesse contexto.
    bare = original.strip("()").replace(" ", "")
    if (
        bare.isalpha()
        and len(bare) <= 4
        and (len(set(bare)) == 1 or bare.lower() in _ROMAN_NUMERAL_TOKENS)
    ):
        return None

    free_vars = _extract_free_variables(original)
    if not free_vars:
        return None

    return SymbolicExpression(original=original, free_vars=free_vars)


@lru_cache(maxsize=4096)
def _cached_compose(text: str) -> SymbolicExpression | None:
    return try_compose_symbolic(text)


# Alias retro-compat (alguns tests usam o nome antigo).
try_parse_symbolic = try_compose_symbolic


# ---------------------------------------------------------------------------
# Detecção de candidatos via expression_lexicon_v0.json
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXPRESSION_LEXICON_PATH = PACKAGE_ROOT / "data" / "expression_lexicon_v0.json"


@lru_cache(maxsize=1)
def _load_expression_lexicon() -> list[tuple[str, re.Pattern, bool, bool]]:
    """Carrega expression_lexicon_v0.json e compila regexes por classe.

    Retorna: (classe, pattern, bypass_p8, atomic_over_protected).
    """
    with DEFAULT_EXPRESSION_LEXICON_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    compiled: list[tuple[str, re.Pattern, bool, bool]] = []
    for entry in data.get("classes", []):
        classe = entry.get("classe")
        regex = entry.get("regex")
        if not classe or not regex:
            continue
        bypass_p8 = bool(entry.get("bypass_p8", False))
        atomic_over_protected = bool(entry.get("atomic_over_protected", False))
        compiled.append(
            (classe, re.compile(regex, re.UNICODE | re.V1), bypass_p8, atomic_over_protected)
        )
    return compiled


_DOLLAR_DELIMITED_RE = re.compile(r"\$([^$\n]+?)\$", re.UNICODE)


def find_symbolic_candidates(
    text: str,
    protected_ranges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, SymbolicExpression, bool]]:
    """Identifica candidatos a ExpressaoSimbolica via lexicon estrutural.

    Args:
        text: texto a escanear.
        protected_ranges: regiões já capturadas como IDX/CONST/GF que NÃO
            devem ser engolidas por SYM clusters maiores.

    Vias:
    1. `$...$` LaTeX explícito (bypass P8).
    2. Padrões estruturais do expression_lexicon (longest-match precedence).

    Validação final: P3+P6+P8+dim_table veto (sem SymPy).
    """
    if protected_ranges is None:
        protected_ranges = []
    results: list[tuple[int, int, SymbolicExpression, bool]] = []
    captured_ranges: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in captured_ranges)

    def _overlaps_protected(span_start: int, span_end: int) -> bool:
        return any(
            ps < span_end and span_start < pe
            for ps, pe in protected_ranges
        )

    def _split_around_protected(
        span_start: int, span_end: int
    ) -> list[tuple[int, int]]:
        sorted_protected = sorted(
            (ps, pe) for ps, pe in protected_ranges
            if ps < span_end and span_start < pe
        )
        pieces: list[tuple[int, int]] = []
        cursor = span_start
        for ps, pe in sorted_protected:
            if ps > cursor:
                pieces.append((cursor, ps))
            cursor = max(cursor, pe)
        if cursor < span_end:
            pieces.append((cursor, span_end))
        return pieces

    def _make_candidate(
        sub_start: int, sub_end: int, bypass_p8: bool, atomic: bool = False
    ) -> tuple[int, int, SymbolicExpression, bool] | None:
        sub_token = text[sub_start:sub_end]
        # Strip extremidades por direção: fim preserva nada além do cluster
        # (operadores no fim são órfãos); início preserva `+`/`-` (sinal unário).
        sub_token = sub_token.rstrip(" \t.,;:·*/+-^=≈≤≥<>")
        sub_token = sub_token.lstrip(" \t.,;:·*/^=≈≤≥<>")
        # P8 com `(`/`)`: marca matemática exige parens balanceadas. Strip
        # parens órfãs das extremidades para isolar o cluster matemático
        # do parêntese de prosa que o delimita.
        while sub_token.count("(") < sub_token.count(")") and sub_token.endswith(")"):
            sub_token = sub_token[:-1]
        while sub_token.count(")") < sub_token.count("(") and sub_token.startswith("("):
            sub_token = sub_token[1:]
        # Re-strip após parens (preserva sinal unário inicial)
        sub_token = sub_token.rstrip(" \t.,;:·*/+-^=≈≤≥<>")
        sub_token = sub_token.lstrip(" \t.,;:·*/^=≈≤≥<>")
        if len(sub_token) < 2:
            return None
        new_start = text.find(sub_token, sub_start, sub_end)
        if new_start < 0:
            return None
        real_end = new_start + len(sub_token)
        if not bypass_p8 and not _has_mathematical_mark(sub_token):
            return None
        analysis = _cached_compose(sub_token)
        if analysis is None:
            return None
        return (new_start, real_end, analysis, atomic)

    # Via 1: `$...$` markup explícito (bypass P8). Atômico — `$...$` é
    # delimitador inviolável, absorve protected_ranges internos.
    all_candidates: list[tuple[int, int, SymbolicExpression, bool]] = []
    for m in _DOLLAR_DELIMITED_RE.finditer(text):
        inner = m.group(1).strip()
        if len(inner) < 3:
            continue
        analysis = _cached_compose(inner)
        if analysis is not None:
            all_candidates.append((m.start(), m.end(), analysis, True))

    # Via 2: padrões estruturais — coleta global, sem decidir overlap aqui.
    # Resolução por longest-match (princípio das notas_design do lexicon).
    for _classe, pattern, bypass_p8, atomic_over_protected in _load_expression_lexicon():
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()

            # Classes atômicas (função aplicada) absorvem CONST/IDX internos:
            # a aplicação de função é o mediador P3 indissociável.
            if atomic_over_protected:
                cand = _make_candidate(start, end, bypass_p8, atomic=True)
                if cand is not None:
                    all_candidates.append(cand)
                continue

            if _overlaps_protected(start, end):
                for sub_start, sub_end in _split_around_protected(start, end):
                    cand = _make_candidate(sub_start, sub_end, bypass_p8)
                    if cand is not None:
                        all_candidates.append(cand)
                continue

            cand = _make_candidate(start, end, bypass_p8)
            if cand is not None:
                all_candidates.append(cand)

    # Resolução de overlap: longest-match precedence (princípio ontológico
    # documentado nas notas_design do expression_lexicon). Cluster mais longo
    # é a interpretação mais específica do mesmo material textual.
    # Tiebreakers: atomic > non-atomic; start asc.
    all_candidates.sort(key=lambda c: (-(c[1] - c[0]), 0 if c[3] else 1, c[0]))
    accepted_ranges: list[tuple[int, int]] = []
    for cand in all_candidates:
        c_start, c_end = cand[0], cand[1]
        if any(s < c_end and c_start < e for s, e in accepted_ranges):
            continue
        results.append(cand)
        accepted_ranges.append((c_start, c_end))

    results.sort(key=lambda r: r[0])
    return results


# ---------------------------------------------------------------------------
# Instanciador
# ---------------------------------------------------------------------------


class ExpressaoSimbolicaInstantiator:
    """Adapter entre Region(EXPRESSAO_SIMBOLICA) e representação canônica."""

    def instantiate(
        self,
        region: Region,
        mode: Literal["A", "B"] = "B",
    ) -> SymbolicExpression:
        """Formata a região como SymbolicExpression. SEM re-validação.

        Princípio arquitetural: a Camada 2 é a ÚNICA autoridade decisória de
        tipo. Este instanciador apenas formata. O método `try_compose_symbolic`
        existe para análise opcional de variáveis livres (P3+P6+P8); falha
        dele NÃO invalida o tipo SYM — apenas indica que não extraímos
        variáveis livres estruturadas.
        """
        if region.tipo is not TipoNome.EXPRESSAO_SIMBOLICA:
            msg = (
                f"ExpressaoSimbolicaInstantiator espera EXPRESSAO_SIMBOLICA, "
                f"recebeu {region.tipo}"
            )
            raise TypeError(msg)
        canonical = region.content.strip().rstrip(".,;:")
        analysis = try_compose_symbolic(canonical)
        if analysis is not None:
            return analysis
        # Token curto, atômico, ou sem composição mediada estrutural:
        # mantém wrap [SYM:<original>] confiando na Camada 2.
        return SymbolicExpression(
            original=canonical,
            free_vars=frozenset({canonical}),
        )

    def instantiate_text(self, content: str, mode: Literal["A", "B"] = "B") -> str:
        # Strip delimitadores `$...$` se presentes
        inner = content.strip()
        if len(inner) >= 2 and inner.startswith("$") and inner.endswith("$"):
            inner = inner[1:-1].strip()
        analysis = try_compose_symbolic(inner)
        if analysis is not None:
            return analysis.render_b()
        # SymPy validation falhou. Comportamento depende do contexto de uso:
        # - Quando chamado pelo classifier (SYM region derivada por camadas),
        #   o classifier garantiu que é SYM legítimo → emitir tag.
        # - Quando chamado DIRETAMENTE com conteúdo arbitrário (uso fora
        #   do pipeline), e o conteúdo não tem marca matemática inerente,
        #   retornar verbatim (compatibilidade com API direta).
        # Heurística: presença de marca matemática determina o modo.
        from toten.classifier.derivation import is_symbolic_token
        # is_symbolic_token cobre marcas A.1-A.5
        # adicionalmente: parens/operadores em geral implicam SYM
        has_math_signal = (
            is_symbolic_token(inner)
            or any(c in inner for c in "+-*/·×÷^=≈≤≥<>()_")
        )
        if has_math_signal:
            return f"[SYM:{inner}]"
        return content
