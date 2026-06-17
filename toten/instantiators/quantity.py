"""Instanciador de GrandezaFisica — o tipo central do framework.

Decompõe a região classificada `<valor> [± <incerteza>] <unidade>` em
componentes estruturados que honram os invariantes da OEE §2.2:

- *valor* canonicalizado para `float` consistente com locale (PT-BR
  vírgula decimal, US ponto decimal, separadores de milhar tratados);
- *unidade* decomposta em sequência de `UnitTerm(symbol, power)` via
  parser recursivo descendente que aceita átomos + operadores
  (`·`, `*`, `/`), expoentes (superscript Unicode `²³…` ou caret
  `^N` / `^-N`) e parênteses;
- *incerteza* opcional na mesma escala do valor.

O parser de unidades é estrutural: qualquer composição de átomos
conhecidos é parseável, mesmo que o composto exato nunca tenha sido
enumerado. `tf/(m²·°C)` decompõe em `[(tf,1), (m,-2), (°C,-1)]`
preservando a estrutura para o consumidor (Modo B) e para o tokenizer
de Modo A.

Modo B (default, model-agnostic): formato textual
`[QTY value=<canonical> unit=<original> unc=<incerteza>?]` consumível
por qualquer LLM frozen. Modo A (futuro completo Dia 6+) materializa
em token IDs do vocab.

Limitações declaradas:
- Tabela dimensional ℤ⁷ não está disponível ainda; `dim_vector` em
  UnitTerm é `None` até Dia 6. Estrutura preservada; semântica
  dimensional chega depois.
- Conversão para SI base não é feita aqui (`350 MPa` mantém
  `unit=MPa`, não `Pa`); canonização SI é Dia 6.
- Separador de milhar ambíguo (`123,456` standalone) resolvido por
  heurística "último separador é decimal" — documentado abaixo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import regex as re

from toten.classifier.region import Region
from toten.dimensional.algebra import combine_terms
from toten.dimensional.aliases import default_unit_aliases
from toten.dimensional.table import DimensionalTable, default_dim_table
from toten.instantiators._r2l import r2l_group
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------


# UnitTerm canonicamente definido em dimensional/algebra.py — é conceito
# dimensional, não de instanciador. Re-exportado aqui por compatibilidade.
from toten.dimensional.algebra import UnitTerm  # noqa: E402


@dataclass(frozen=True, slots=True)
class Quantity:
    """GrandezaFisica decomposta em valor + unidade estrutural.

    Em OEE v1.1, `value` pode ser `None` para representar unidade-isolada
    (string só-unidade como `"kgf/cm²"` ou `"kN·m"` sem magnitude
    associada). Quando `value=None`, `uncertainty`, `si_value` e
    `si_uncertainty` também são `None` (não há valor para converter);
    `dim_vector`, `factor` e `si_unit` permanecem disponíveis como
    expressão da dimensão pura.

    Campos opcionais `dim_vector`, `factor`, `si_unit` ficam populados
    quando todos os átomos da unidade estão na DimensionalTable; senão
    permanecem None (graceful degradation — Modo B cai no fallback
    verbatim).
    """

    value: float | None
    unit_text: str
    unit_terms: tuple[UnitTerm, ...]
    uncertainty: float | None = None
    raw: str = ""
    dim_vector: tuple[int, int, int, int, int, int, int] | None = None
    factor: float | None = None
    si_value: float | None = None
    si_unit: str | None = None
    si_uncertainty: float | None = None
    ambig_kind: str | None = None
    ambig_alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.unit_text.strip():
            msg = "Quantity exige unit_text não-vazio"
            raise ValueError(msg)
        if not self.unit_terms:
            msg = "Quantity exige ao menos um UnitTerm"
            raise ValueError(msg)
        # v1.1: value=None implica uncertainty/si_value/si_uncertainty=None
        if self.value is None:
            if self.uncertainty is not None:
                msg = "Quantity sem value não pode ter uncertainty"
                raise ValueError(msg)
            if self.si_value is not None:
                msg = "Quantity sem value não pode ter si_value derivado"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class QuantityToken:
    """Representação tokenizada de uma GrandezaFisica."""

    quantity: Quantity
    mode: Literal["A", "B"]

    @property
    def text(self) -> str:
        if self.mode == "B":
            return self._render_b()
        return self._render_a()

    def _render_b(self) -> str:
        """Modo B simples: preserva unidade original do engenheiro e
        anexa `dim` para identidade dimensional. Conversão SI (`si_value`,
        `si_unit`) fica disponível programaticamente nos campos da
        Quantity para downstream, mas NÃO entra no texto por default —
        verbose desnecessário; LLMs de fronteira fazem `MPa → Pa` com
        confiança alta a partir do dim signature.

        Para átomo desconhecido (graceful degradation): só value+unit
        verbatim, sem dim.

        Em OEE v1.1, `value=None` representa unidade-isolada — tag emitida
        como `[QTY unit=X dim=Y]` sem `value=` (semântica clara: "esta
        é a dimensão de uma unidade isolada, sem magnitude associada").
        """
        q = self.quantity
        parts: list[str] = []
        if q.value is not None:
            parts.append(f"value={_format_value(q.value)}")
            # R2L grouping opcional — SOTA Number Cookbook (Yang ICLR 2025)
            # Singh-Strouse 2024: agrupamento explícito de dígitos sobe
            # aritmética posicional de LLMs em ~22 pontos. Emitido apenas
            # quando muda visivelmente o value (|value|≥1000).
            r2l = r2l_group(q.value)
            if r2l is not None:
                parts.append(f'r2l="{r2l}"')
        parts.append(f"unit={q.unit_text}")
        if q.dim_vector is not None:
            parts.append(f"dim={_format_dim(q.dim_vector)}")
        if q.uncertainty is not None:
            parts.append(f"unc={_format_value(q.uncertainty)}")
        # SPEC_07 — marcador de ambiguidade declarada (opcional).
        # Quando o classifier detecta ambiguidade estrutural (unit-letter,
        # domain-semantic, typography-degraded, locale-thousands), emite
        # `ambig=<tipo>` + `alternatives="<a1>|<a2>|..."` para que o
        # agente downstream inicie diálogo com o usuário.
        if q.ambig_kind:
            parts.append(f'ambig="{q.ambig_kind}"')
            if q.ambig_alternatives:
                alts = "|".join(q.ambig_alternatives)
                parts.append(f'alternatives="{alts}"')
        return f"[QTY {' '.join(parts)}]"

    def _render_a(self) -> str:
        q = self.quantity
        if q.dim_vector is not None and q.si_unit is not None:
            # Modo A trinitário com SI canônico
            val = _format_value(q.si_value)
            unc = _format_value(q.si_uncertainty) if q.si_uncertainty is not None else None
            dim_text = q.si_unit if q.si_unit else _render_terms(q.unit_terms)
        else:
            val = _format_value(q.value)
            unc = _format_value(q.uncertainty) if q.uncertainty is not None else None
            dim_text = _render_terms(q.unit_terms)
        # v1.1: unidade-isolada (value=None) emite <VAL/> auto-fechado
        val_tag = f"<VAL>{val}</VAL>" if q.value is not None else "<VAL/>"
        chunks = [f"<QTY>{val_tag}", f"<DIM>{dim_text}</DIM>"]
        if unc is not None:
            chunks.append(f"<UNC>{unc}</UNC>")
        else:
            chunks.append("<UNC/>")
        chunks.append("</QTY>")
        return "".join(chunks)


def _format_dim(dim: tuple[int, ...]) -> str:
    return "[" + ",".join(str(d) for d in dim) + "]"


# ---------------------------------------------------------------------------
# Canonicalização numérica (locale-aware)
# ---------------------------------------------------------------------------


_SUPERSCRIPT_DIGIT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
}

# Conversão de variantes científicas para forma canônica `eN`:
# - `× 10⁻³⁵` / `· 10⁻³⁵` (Unicode superscript com ×, x, ·, *)
# - `× 10^-3` / `× 10^(-3)` (caret ASCII)
# - `10⁶` standalone (notação científica BR sem mantissa explícita)
_SCI_UNICODE_RE = re.compile(
    r"\s*[×x·*]\s*10\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
)
_SCI_CARET_RE = re.compile(
    r"\s*[×x·*]\s*10\s*\^\s*\(?\s*([+\-−]?\d+)\s*\)?"
)
# Notação científica BR standalone: `10⁶ K`, `10¹² Pa·s`. Sem mantissa
# explícita — mantissa virtual = 1.
_SCI_BARE_RE = re.compile(
    r"\A\s*([+\-−]?)10\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*\Z"
)


def _normalize_scientific_notation(raw: str) -> str:
    """Normaliza variantes Unicode/caret de notação científica para `eN`.

    Engenharia brasileira/internacional usa `1,5 × 10⁻³` mais que `1.5e-3`.
    O framework aceita ambas — esta função é self-contained, determinística,
    invocada antes do parse decimal.

    Examples
    --------
    >>> _normalize_scientific_notation("1,5 × 10⁻³")
    '1,5e-3'
    >>> _normalize_scientific_notation("2 × 10^5")
    '2e5'
    >>> _normalize_scientific_notation("3.0 · 10⁻¹⁵")
    '3.0e-15'
    """

    def _unicode_to_e(m: re.Match) -> str:
        exp_uni = m.group(1)
        exp = "".join(_SUPERSCRIPT_DIGIT_MAP.get(c, c) for c in exp_uni)
        return f"e{exp}"

    def _caret_to_e(m: re.Match) -> str:
        exp = m.group(1).replace("−", "-")
        return f"e{exp}"

    # Standalone primeiro (antes das versões com mantissa) — caso `10⁶`
    # standalone, prepende mantissa virtual `1`.
    bare = _SCI_BARE_RE.match(raw)
    if bare is not None:
        sign, exp_uni = bare.groups()
        exp = "".join(_SUPERSCRIPT_DIGIT_MAP.get(c, c) for c in exp_uni)
        return f"{sign}1e{exp}"
    raw = _SCI_UNICODE_RE.sub(_unicode_to_e, raw)
    raw = _SCI_CARET_RE.sub(_caret_to_e, raw)
    return raw


def canonicalize_number(raw: str) -> float:
    """Devolve o valor float canônico de uma string numérica.

    Aceita variantes:
    - Locale PT-BR `287,4` e US `287.4` (último separador é decimal).
    - Thousands PT-BR `1.234,5` ou US `1,234.5`.
    - Científica `e/E`: `1.5e-3`, `2E+5`.
    - Científica Unicode: `1,5 × 10⁻³` (superscript).
    - Científica caret: `2 × 10^5`, `3.0 × 10^(-3)`.
    - Operadores múltiplos para ×: `×`, `x`, `·`, `*`.
    - Sinal Unicode `−` aceito como menos.

    Examples
    --------
    >>> canonicalize_number("350")
    350.0
    >>> canonicalize_number("287,4")
    287.4
    >>> canonicalize_number("1.5e-3")
    0.0015
    >>> canonicalize_number("1,616 × 10⁻³⁵")
    1.616e-35
    """
    if raw is None:
        msg = "canonicalize_number exige string não-nula"
        raise ValueError(msg)
    s = raw.strip().replace("−", "-")
    if not s:
        msg = "canonicalize_number exige string não-vazia"
        raise ValueError(msg)
    s = _normalize_scientific_notation(s)
    if "e" in s.lower():
        match = re.fullmatch(r"\s*([+-]?[\d.,]+)\s*[eE]\s*([+-]?\d+)\s*", s)
        if not match:
            msg = f"canonicalize_number não conseguiu parsear: {raw!r}"
            raise ValueError(msg)
        mantissa_raw, exp_raw = match.groups()
        mantissa = _canonicalize_decimal(mantissa_raw)
        return mantissa * (10 ** int(exp_raw))
    return _canonicalize_decimal(s)


def _canonicalize_decimal(s: str) -> float:
    """Canonicaliza string SEM expoente, retornando float.

    Regras para resolver locale PT-BR vs US:
    - Múltiplos `.` sem `,`: PT-BR thousands (`1.234.567` = 1234567).
    - Múltiplos `,` sem `.`: US thousands (`1,234,567` = 1234567).
    - Ambos `.` e `,`: o último é decimal, o outro é thousands.
    - Único `,`: decimal PT-BR (`287,4` = 287.4).
    - Único `.`: ambíguo. Heurística: se EXATAMENTE 3 dígitos seguem o
      ponto E a parte inteira é não-zero, interpreta como thousands
      PT-BR (`1.900` = 1900). Senão é decimal US (`287.4` = 287.4,
      `0.500` = 0.5). Engenharia BR usa rotineiramente `1.900 MPa`
      para 1900 MPa em textos técnicos formais.
    """
    sign = 1.0
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        s = s[1:]
        sign = -1.0

    n_commas = s.count(",")
    n_dots = s.count(".")

    if n_commas == 0 and n_dots == 0:
        return sign * float(s)

    if n_dots > 1 and n_commas == 0:
        # Múltiplos pontos: PT-BR thousands sem decimal
        return sign * float(s.replace(".", ""))

    if n_commas > 1 and n_dots == 0:
        # Múltiplos vírgulas: US thousands sem decimal (raro)
        return sign * float(s.replace(",", ""))

    if n_commas >= 1 and n_dots >= 1:
        # Ambos presentes: último é decimal, outro é thousands
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            decimal_pos = last_comma
            thousands = "."
        else:
            decimal_pos = last_dot
            thousands = ","
        integer_part = s[:decimal_pos].replace(thousands, "")
        decimal_part = s[decimal_pos + 1 :]
        if not integer_part:  # pragma: no cover — defensivo para inputs degenerados (".,5")
            integer_part = "0"
        return sign * float(f"{integer_part}.{decimal_part}")

    if n_commas == 1 and n_dots == 0:
        # Única vírgula: decimal PT-BR
        return sign * float(s.replace(",", "."))

    # n_dots == 1 and n_commas == 0: ambíguo decimal US vs thousands PT-BR
    dot_pos = s.find(".")
    integer_part = s[:dot_pos]
    decimal_part = s[dot_pos + 1 :]
    if (
        len(decimal_part) == 3
        and len(integer_part) >= 1
        and integer_part.isdigit()
        and integer_part != "0"
    ):
        # Heurística PT-BR thousands: 1.900 → 1900
        return sign * float(integer_part + decimal_part)
    # Default: decimal US
    return sign * float(s)


def _format_value(value: float | None) -> str:
    """Repr canônica, idêntica em Modo A e Modo B, consistente entre QTY e NUM.

    Decisão v0.7 (consistência arquitetural com `numero._format_value`):
    inteiros até `1e15` (limite seguro de IEEE 754 sem perda de precisão)
    ficam compactos como `str(int(value))`; valores não-inteiros muito
    pequenos (`<1e-3`) ou muito grandes (`≥1e6`) em notação científica
    `:.6g`; intermediários como `repr(float)`. O R2L grouping no Modo B
    é informação complementar (não substitui o `value=`); Modo A não
    emite R2L mas mantém o mesmo `<VAL>X</VAL>` canônico.

    Princípio: NENHUMA divergência de formatação entre modos ou
    instanciadores — um valor numérico tem uma única forma canônica
    em todo o framework.
    """
    if value is None:
        return ""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if value != 0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
        return f"{value:.6g}"
    return repr(value)


# ---------------------------------------------------------------------------
# Parser recursivo descendente de unidades
# ---------------------------------------------------------------------------

_SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-",
}
_SUPERSCRIPT_CHARS = frozenset(_SUPERSCRIPT_MAP)


@dataclass
class _UnitParser:
    text: str
    pos: int = field(default=0)

    def parse(self) -> tuple[UnitTerm, ...]:
        self._skip_ws()
        terms = self._composition(sign=1)
        self._skip_ws()
        if self.pos < len(self.text):
            msg = (
                f"caractere inesperado em '{self.text}' "
                f"pos {self.pos}: {self.text[self.pos]!r}"
            )
            raise ValueError(msg)
        if not terms:  # pragma: no cover — defensivo; _factor sempre retorna ≥1 termo ou levanta
            msg = f"unidade vazia em '{self.text}'"
            raise ValueError(msg)
        return tuple(terms)

    # ---- gramática

    def _composition(self, sign: int) -> list[UnitTerm]:
        terms = self._factor(sign)
        while True:
            pos_before_ws = self.pos
            self._skip_ws()
            had_space = self.pos > pos_before_ws
            op = self._peek()
            if op in ("·", "*"):
                self.pos += 1
                terms.extend(self._factor(sign))
            elif op == "/":
                self.pos += 1
                terms.extend(self._factor(-sign))
            elif had_space and op is not None and (
                op.isalpha() or op in ("(", "µ", "μ", "Ω", "°")
            ):
                # Espaço como operador composicional multiplicativo
                # implícito (BIPM 9th §5.2 / ISO 80000-1): `m s⁻²`
                # equivalente a `m·s⁻²`. Reconhecido apenas quando o
                # próximo token inicia novo símbolo de unidade
                # (consistente com classifier que só captura átomos do
                # oráculo após espaço).
                terms.extend(self._factor(sign))
            else:
                break
        return terms

    def _factor(self, sign: int) -> list[UnitTerm]:
        self._skip_ws()
        if self._peek() == "(":
            self.pos += 1
            inner = self._composition(sign)
            self._skip_ws()
            if self._peek() != ")":
                msg = f"esperado ')' em '{self.text}' pos {self.pos}"
                raise ValueError(msg)
            self.pos += 1
            return inner
        symbol = self._symbol()
        if not symbol:
            msg = f"esperado símbolo em '{self.text}' pos {self.pos}"
            raise ValueError(msg)
        power = self._power()
        return [UnitTerm(symbol=symbol, power=power * sign)]

    # ---- léxica

    def _peek(self) -> str | None:
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _symbol(self) -> str:
        start = self.pos
        # Chars adicionais aceitos em símbolo de unidade:
        # - µ/μ: micro (prefixo SI U+00B5 / U+03BC)
        # - Ω: ohm
        # - °: grau angular
        # - %, ‰: razões puras (unidades dimensionless do whitelist)
        _EXTRA_SYMBOL_CHARS = ("µ", "μ", "Ω", "°", "%", "‰")
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isalpha() or c in _EXTRA_SYMBOL_CHARS:
                self.pos += 1
            else:
                break
        return self.text[start : self.pos]

    def _power(self) -> int:
        # superscript Unicode
        if self._peek() in _SUPERSCRIPT_CHARS:
            digits = ""
            while self._peek() in _SUPERSCRIPT_CHARS:
                digits += _SUPERSCRIPT_MAP[self.text[self.pos]]
                self.pos += 1
            return int(digits)
        # caret style
        if self._peek() == "^":
            self.pos += 1
            start = self.pos
            if self._peek() == "-":
                self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            if start == self.pos or (self.text[start] == "-" and start + 1 == self.pos):
                msg = f"expoente sem dígitos após '^' em '{self.text}' pos {start}"
                raise ValueError(msg)
            return int(self.text[start : self.pos])
        return 1


def parse_unit_composition(unit_text: str) -> tuple[UnitTerm, ...]:
    """Parse `kgf/cm²` em `(UnitTerm('kgf',1), UnitTerm('cm',-2))`.

    Pré-normaliza aliases conhecidos (data/unit_aliases_v0.json) ANTES da
    decomposição: `mt → tf·m`, `t/m → tf/m`, `kgf/cm2 → kgf/cm²`, etc.
    Convenção dos aliases elimina ambiguidade entre uso técnico e SI
    moderno em composições com `t`.

    Fast-path: se o input inteiro (após alias) já é átomo conhecido do
    oráculo (e.g., `mmHg`, `mmH2O`, `inHg`), devolve `[(input, 1)]`
    direto sem tentar decompor — evita falso positivo em parser que
    interpretaria os dígitos internos como expoentes.
    """
    if not unit_text or not unit_text.strip():
        msg = "parse_unit_composition exige string não-vazia"
        raise ValueError(msg)
    normalized = default_unit_aliases().normalize(unit_text.strip())
    # Fast-path: átomo único conhecido
    if normalized in default_dim_table().atoms:
        return (UnitTerm(symbol=normalized, power=1),)
    return _UnitParser(normalized).parse()


def _render_terms(terms: tuple[UnitTerm, ...]) -> str:
    return "·".join(_render_term(t) for t in terms)


def _render_term(term: UnitTerm) -> str:
    if term.power == 1:
        return term.symbol
    return f"{term.symbol}^{term.power}"


# ---------------------------------------------------------------------------
# Instanciador
# ---------------------------------------------------------------------------

# Núcleo numérico — sincronizado com classifier._NUMBER_RE. Aceita
# locale PT-BR/US e científica em três formas.
_NUM_INTERNAL = (
    r"[+\-−]?"
    r"(?:"
    r"10[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"                 # notação científica BR standalone: 10⁶, 10¹²
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d+(?:[.,]\d+)?"
    r")"
    r"(?:"
    r"[eE][+\-−]?\d+"
    r"|\s*[×x·*]\s*10\s*[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"
    r"|\s*[×x·*]\s*10\s*\^\s*\(?\s*[+\-−]?\d+\s*\)?"
    r")?"
)

_GRANDEZA_INTERNAL_RE = re.compile(
    r"\s*"
    rf"(?P<num>{_NUM_INTERNAL})"
    rf"(?:\s*(?:±|\+/-)\s*(?P<unc>{_NUM_INTERNAL}))?"
    # \s* (zero ou mais): consistente com Camada 2, que aceita `540°C`,
    # `350MPa` sem espaço. A unit DEVE começar com caractere não-numérico
    # (letra, °, µ, etc.) para impedir backtrack ambíguo.
    r"\s*"
    r"(?P<unit>[^\d\s±+/\-×x·*]\S*(?:.*?\S)?)"
    r"\s*$"
)


def parse_grandeza(
    content: str, dim_table: DimensionalTable | None = None
) -> Quantity:
    """Decompõe `<num> [± <unc>] <unit>` em `Quantity`.

    Em OEE v1.1, aceita também unidade-isolada (`"kgf/cm²"`, `"kN·m"`)
    sem valor associado — devolve `Quantity` com `value=None`.

    Quando `dim_table` é fornecida (ou se default disponível), realiza
    análise dimensional: combina os termos, computa `si_value` e
    `si_unit` em SI base, popula `dim_vector`. Se algum átomo não está
    na tabela, esses campos ficam `None` (Modo B então cai no
    fallback verbatim).
    """
    if not content or not content.strip():
        msg = "parse_grandeza exige string não-vazia"
        raise ValueError(msg)

    # Tenta primeiro o padrão completo value+unit (compatibilidade v1.0)
    match = _GRANDEZA_INTERNAL_RE.fullmatch(content)
    if match:
        num_raw = match.group("num")
        unc_raw = match.group("unc")
        unit_raw = match.group("unit").strip()
        value = canonicalize_number(num_raw)
        uncertainty = canonicalize_number(unc_raw) if unc_raw else None
    else:
        # Fallback v1.1: tenta como unidade-isolada (sem value)
        stripped = content.strip()
        try:
            unit_terms_check = parse_unit_composition(stripped)
            if not unit_terms_check:
                msg = f"parse_grandeza não conseguiu casar: {content!r}"
                raise ValueError(msg)
            unit_raw = stripped
            value = None
            uncertainty = None
        except (ValueError, TypeError):
            msg = f"parse_grandeza não conseguiu casar: {content!r}"
            raise ValueError(msg) from None

    unit_terms = parse_unit_composition(unit_raw)

    dim_vector = factor = si_value = si_unit = si_uncertainty = None
    table = dim_table if dim_table is not None else default_dim_table()
    analysis = combine_terms(unit_terms, table)
    if analysis is not None:
        dim_vector = analysis.dim
        factor = analysis.factor
        si_unit = analysis.si_unit
        # si_value e si_uncertainty só fazem sentido quando há value
        if value is not None:
            si_value = value * factor
            si_uncertainty = uncertainty * factor if uncertainty is not None else None

    return Quantity(
        value=value,
        unit_text=unit_raw,
        unit_terms=unit_terms,
        uncertainty=uncertainty,
        raw=content,
        dim_vector=dim_vector,
        factor=factor,
        si_value=si_value,
        si_unit=si_unit,
        si_uncertainty=si_uncertainty,
    )


class GrandezaFisicaInstantiator:
    """Camada 3 — instancia regiões de GrandezaFisica.

    Carrega `DimensionalTable` por default; pode receber tabela
    customizada via construtor (útil em testes).
    """

    def __init__(self, dim_table: DimensionalTable | None = None) -> None:
        self._table = dim_table if dim_table is not None else default_dim_table()

    def instantiate(
        self, region: Region, mode: Literal["A", "B"] = "B"
    ) -> QuantityToken:
        if region.tipo is not TipoNome.GRANDEZA_FISICA:
            msg = (
                "GrandezaFisicaInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava GrandezaFisica"
            )
            raise TypeError(msg)
        quantity = parse_grandeza(region.content, dim_table=self._table)
        # SPEC_07 — propaga marcador de ambiguidade do classifier
        if region.ambig_kind:
            from dataclasses import replace
            quantity = replace(
                quantity,
                ambig_kind=region.ambig_kind,
                ambig_alternatives=region.ambig_alternatives,
            )
        return QuantityToken(quantity=quantity, mode=mode)

    def instantiate_text(self, content: str, mode: Literal["A", "B"] = "B") -> str:
        """Atalho que aceita string diretamente.

        Suporta range `<num1>-<num2> <unit>` (intervalo BR): emite dois QTYs
        com `-` literal entre eles. Princípio: range é estrutura "intervalo"
        que liga dois valores compartilhando a unidade.
        """
        range_match = _RANGE_GRANDEZA_RE.fullmatch(content.strip())
        if range_match is not None:
            num1_raw = range_match.group("num1")
            num2_raw = range_match.group("num2")
            unit_raw = range_match.group("unit")
            q1 = parse_grandeza(f"{num1_raw} {unit_raw}", dim_table=self._table)
            q2 = parse_grandeza(f"{num2_raw} {unit_raw}", dim_table=self._table)
            tok1 = QuantityToken(quantity=q1, mode=mode).text
            tok2 = QuantityToken(quantity=q2, mode=mode).text
            return f"{tok1}-{tok2}"
        quantity = parse_grandeza(content, dim_table=self._table)
        return QuantityToken(quantity=quantity, mode=mode).text


# Padrão de range numérico `<num1>-<num2> <unit>`: estrutura "intervalo"
# (ex: `0,4-0,8 s`, `5-20 mA`, `80-100 kvar`). O instantiator detecta e
# emite dois QTYs adjacentes preservando o `-` literal.
_RANGE_GRANDEZA_RE = re.compile(
    rf"\s*(?P<num1>{_NUM_INTERNAL})\s*-\s*(?P<num2>{_NUM_INTERNAL})"
    rf"\s*(?P<unit>[^\d\s±+/\-×x·*]\S*(?:.*?\S)?)\s*"
)
