"""NumeroInstantiator — Camada 3 para o 8º tipo da OEE (v1.1).

Numero é entidade primária do mundo abstrato: valor escalar real SEM
dimensão física associada. Tipado em `data/oee-v1.yaml` (v1.1) com
4 propriedades intrínsecas:

    - valor: escalar real (IEEE 754 double)
    - locale: pt-br | en | mixed | none
    - representacao: decimal | cientifica | inteira | fracionaria | percentual
    - sinal: positivo | negativo | zero

Modo B emite tag rica:
    [NUM value=1234.56 locale=pt-br repr=decimal original="1.234,56"]

Decisão de design (preservação literal + SOTA Number Cookbook):
- `original=` preserva a string que o engenheiro escreveu — LLM
  consumidor vê notação real, não normalizada.
- `value=` expõe o valor parseado em representação canônica IEEE 754 —
  downstream pode usar para aritmética sem re-parser.
- `locale=` declarado explicitamente — sem inferência silenciosa.
- `repr=` permite raciocínio sobre estrutura representacional.

Endereça lacuna SOTA documentada em LIMITATIONS_AND_OPEN_QUESTIONS §3.1
(Yang ICLR 2025 Number Cookbook + Singh-Strouse 2024).

Princípio do framework preservado: nenhuma álgebra simbólica.
Fração `3/4` é parseada para `value=0.75` via divisão direta (não SymPy).
Percentual `50%` é parseado para `value=0.5` (convenção: divide por 100).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from toten.classifier.region import Region
from toten.instantiators._r2l import r2l_group
from toten.ontology.types import TipoNome


Locale = Literal["pt-br", "en", "mixed", "none"]
Representacao = Literal[
    "decimal",
    "cientifica",
    "inteira",
    "fracionaria",
    "percentual",
    "ordinal",
]
Sinal = Literal["positivo", "negativo", "zero"]

# Marcadores ordinais PT-BR — derivados de UCD por propriedade Unicode
# (`FEMININE/MASCULINE ORDINAL INDICATOR`, decomposição `<super>` + letra
# latina a/o). NÃO é enumeração ad-hoc: são os ÚNICOS caracteres do
# Unicode com nome "ORDINAL INDICATOR". `°` (DEGREE SIGN, U+00B0) é
# explicitamente excluído — pertence à categoria So (Symbol Other) e
# significa grau angular, não ordinal.
_ORDINAL_MARKERS = frozenset(["ª", "º"])


@dataclass(frozen=True, slots=True)
class Numero:
    """Numero decomposto em propriedades intrínsecas da OEE v1.1."""

    valor: float
    locale: Locale
    representacao: Representacao
    sinal: Sinal
    original: str

    def __post_init__(self) -> None:
        if not self.original.strip():
            msg = "Numero exige original não-vazio"
            raise ValueError(msg)

    @property
    def sinal_inferido(self) -> Sinal:
        """Sinal derivado de valor — invariante de consistência."""
        if self.valor > 0:
            return "positivo"
        if self.valor < 0:
            return "negativo"
        return "zero"


@dataclass(frozen=True, slots=True)
class NumeroToken:
    """Representação tokenizada de um Numero (Modo A ou B)."""

    numero: Numero
    mode: Literal["A", "B"]

    @property
    def text(self) -> str:
        if self.mode == "B":
            return self._render_b()
        return self._render_a()

    def _render_b(self) -> str:
        """Modo B — tag rica preservando original + valor canônico.

        Formato canônico:
            [NUM value=<float> locale=<l> repr=<r> original="<orig>"]

        `original=` entre aspas para permitir conteúdo com espaços e
        caracteres especiais sem ambiguidade de parsing.
        """
        n = self.numero
        parts = [f"value={_format_value(n.valor)}"]
        # R2L grouping opcional — SOTA Number Cookbook (Yang ICLR 2025).
        # Singh-Strouse 2024: agrupamento posicional explícito sobe aritmética
        # em ~22 pontos. Emitido apenas quando |value|≥1000 (caso onde muda
        # visivelmente a string).
        r2l = r2l_group(n.valor)
        if r2l is not None:
            parts.append(f'r2l="{r2l}"')
        parts.extend(
            [
                f"locale={n.locale}",
                f"repr={n.representacao}",
                f'original="{n.original}"',
            ]
        )
        return f"[NUM {' '.join(parts)}]"

    def _render_a(self) -> str:
        """Modo A — formato compacto para SLM treinado."""
        n = self.numero
        return f"<NUM>{_format_value(n.valor)}</NUM>"


def _format_value(value: float) -> str:
    """Formato canônico de valor — compacto para int, IEEE para float."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if value != 0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
        return f"{value:.6g}"
    return repr(value)


# ---------------------------------------------------------------------------
# Parsing locale-aware
# ---------------------------------------------------------------------------


_SUPERSCRIPT_DIGITS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
}


def _normalize_scientific(s: str) -> str:
    """Converte notação científica para notação `eN` parseável por float().

    Cobre três variantes:
    - `× 10⁻²³` / `· 10⁻²³` etc.: mantissa + operador + 10 + superscript
    - `× 10^-3`: mantissa + operador + 10 + caret + dígitos
    - `10⁶` standalone: notação científica BR sem mantissa explícita
      (mantissa virtual = 1). Convenção corrente em engenharia BR
      (`E_a = 10⁶ J/mol`, `pop ≈ 10⁹ pessoas`).
    """
    def _uni_to_e(m: re.Match) -> str:
        exp = "".join(_SUPERSCRIPT_DIGITS.get(c, c) for c in m.group(1))
        return f"e{exp}"

    s = re.sub(r"\s*[×x·*]\s*10\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", _uni_to_e, s)
    s = re.sub(
        r"\s*[×x·*]\s*10\s*\^\s*\(?\s*([+\-−]?\d+)\s*\)?",
        lambda m: f"e{m.group(1).replace('−', '-')}",
        s,
    )
    # `10⁶` standalone (sem mantissa) → `1e6`
    def _bare_sci_to_e(m: re.Match) -> str:
        sign = m.group(1).replace("−", "-")
        exp = "".join(_SUPERSCRIPT_DIGITS.get(c, c) for c in m.group(2))
        return f"{sign}1e{exp}"

    s = re.sub(
        r"^([+\-−]?)10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)$",
        _bare_sci_to_e,
        s,
    )
    return s


def _detect_locale(original: str) -> Locale:
    """Detecta locale por estrutura da string.

    - Tem `,` E `.` → PT-BR (1.234,56) ou US (1,234.56)
      Heurística: posição relativa decide. PT-BR tem `.` antes de `,`.
    - Só `,` → PT-BR (decimal)
    - Só `.` → ambíguo (en preferido, mas pode ser milhar PT-BR)
    - Nem `,` nem `.` → none
    """
    has_comma = "," in original
    has_dot = "." in original
    if has_comma and has_dot:
        last_comma = original.rfind(",")
        last_dot = original.rfind(".")
        return "pt-br" if last_comma > last_dot else "en"
    if has_comma:
        return "pt-br"
    if has_dot:
        # Pode ser US decimal OU PT-BR milhar (1.234, 1.234.567, ...).
        # Regra ESTRUTURAL ontológica: notação de milhar PT-BR exige
        # primeiro dígito ≠ 0 (sistema posicional sem zero à esquerda
        # em referências de grandeza — "0.500" significando 500 não
        # existe em PT-BR; seria escrito como "500"). Quando o token
        # começa por "0.", o ponto é decimal EN inequivocamente.
        # Ponto(s) separando grupos de 3 dígitos exatos com primeiro
        # grupo iniciando por dígito 1-9 = milhar PT-BR.
        m = re.match(r"^[-−+]?[1-9]\d{0,2}(?:\.\d{3})+$", original.strip())
        if m:
            return "pt-br"
        return "en"
    return "none"


def _detect_representacao(original: str) -> Representacao:
    """Detecta representação por marcadores na string."""
    s = original.strip()
    # Ordinal: dígito(s) seguido(s) de ª (feminino) ou º (masculino).
    # Verificado ANTES de outros padrões (marcador estrutural inequívoco
    # via UCD FEMININE/MASCULINE ORDINAL INDICATOR).
    if s and s[-1] in _ORDINAL_MARKERS and s[:-1].lstrip("-−+").isdigit():
        return "ordinal"
    if "%" in s:
        return "percentual"
    if "/" in s and not re.search(r"\d+/\d+/\d+", s):  # exclui datas
        return "fracionaria"
    if re.search(r"[eE×x·*]\s*10|[eE][+\-−]?\d", s):
        return "cientifica"
    # Superscript Unicode standalone (notação BR `10⁶`, `10²³`) — científica
    if re.search(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", s):
        return "cientifica"
    if "," in s or "." in s:
        return "decimal"
    return "inteira"


def parse_numero(content: str) -> Numero:
    """Decompõe `content` (string já reconhecida como número) em `Numero`.

    Aceita locale PT-BR e US, notação científica Unicode e ASCII,
    sinal negativo Unicode (−) e ASCII (-), fração simples e percentual.
    """
    if not content or not content.strip():
        msg = "parse_numero exige string não-vazia"
        raise ValueError(msg)

    original = content.strip()
    locale = _detect_locale(original)
    repr_kind = _detect_representacao(original)

    # Limpeza textual para parsing numérico
    s = original.replace("−", "-").strip()

    if repr_kind == "ordinal":
        # Ordinal: extrai dígitos antes do marcador (`2ª` → 2, `21º` → 21).
        # Marcador descartado do valor — informação preservada em `original`
        # e em `representacao=ordinal` para o consumidor downstream.
        digits = s[:-1].replace("−", "-").strip()
        try:
            value = float(digits)
        except ValueError as exc:
            msg = f"ordinal mal-formado: {original!r}"
            raise ValueError(msg) from exc
    elif repr_kind == "fracionaria":
        # n/d → float
        m = re.match(r"^\s*([-+]?\d+)\s*/\s*(\d+)\s*$", s)
        if m:
            num = int(m.group(1))
            den = int(m.group(2))
            if den == 0:
                msg = f"divisão por zero em fração: {original!r}"
                raise ValueError(msg)
            value = num / den
        else:
            msg = f"fração mal-formada: {original!r}"
            raise ValueError(msg)
    elif repr_kind == "percentual":
        # X% → X/100
        num_str = s.rstrip("%").strip()
        num_str = _localize_to_us(num_str, locale)
        try:
            value = float(num_str) / 100.0
        except ValueError as exc:
            msg = f"percentual mal-formado: {original!r}"
            raise ValueError(msg) from exc
    elif repr_kind == "cientifica":
        normalized = _normalize_scientific(s)
        normalized = _localize_to_us(normalized, locale)
        try:
            value = float(normalized)
        except ValueError as exc:
            msg = f"científica mal-formada: {original!r}"
            raise ValueError(msg) from exc
    else:
        # decimal ou inteira
        normalized = _localize_to_us(s, locale)
        try:
            value = float(normalized)
        except ValueError as exc:
            msg = f"número mal-formado: {original!r}"
            raise ValueError(msg) from exc

    # Sinal derivado do valor (invariante)
    if value > 0:
        sinal: Sinal = "positivo"
    elif value < 0:
        sinal = "negativo"
    else:
        sinal = "zero"

    return Numero(
        valor=value,
        locale=locale,
        representacao=repr_kind,
        sinal=sinal,
        original=original,
    )


def _localize_to_us(s: str, locale: Locale) -> str:
    """Converte string locale-specific para formato US (parseável por float())."""
    if locale == "pt-br":
        # Remove pontos de milhar, vírgula vira decimal
        if "," in s and "." in s:
            return s.replace(".", "").replace(",", ".")
        if "," in s:
            return s.replace(",", ".")
        # Só ponto(s): milhar PT-BR (1.234, 1.234.567, etc.).
        # Estrutural: primeiro dígito ≠ 0 (sem zero à esquerda em
        # grandezas posicionais PT-BR; "0.500" → 500 não existe).
        m = re.match(r"^([-+]?[1-9]\d{0,2}(?:\.\d{3})+)$", s)
        if m:
            return s.replace(".", "")
    elif locale == "en":
        # US format: vírgulas são milhar, ponto é decimal
        if "," in s:
            return s.replace(",", "")
    return s


# ---------------------------------------------------------------------------
# Instanciador
# ---------------------------------------------------------------------------


class NumeroInstantiator:
    """Camada 3 — instancia regiões do tipo Numero (OEE v1.1)."""

    def instantiate(
        self, region: Region, mode: Literal["A", "B"] = "B"
    ) -> NumeroToken:
        if region.tipo is not TipoNome.NUMERO:
            msg = (
                "NumeroInstantiator recebeu região de tipo "
                f"{region.tipo}; esperava Numero"
            )
            raise TypeError(msg)
        numero = parse_numero(region.content)
        return NumeroToken(numero=numero, mode=mode)

    def instantiate_text(
        self, content: str, mode: Literal["A", "B"] = "B"
    ) -> str:
        numero = parse_numero(content)
        return NumeroToken(numero=numero, mode=mode).text
