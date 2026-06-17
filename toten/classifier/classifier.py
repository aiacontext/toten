"""Classificador ontológico (Camada 2) — versão Dia 3.

Identifica regiões de TODOS os tipos da OEE v1.0 em texto bruto. Saída
é sequência ordenada de `Region(tipo, start, end, content)` cobrindo
trechos não-triviais do texto.

Algoritmo:

1. Cada tipo declarado em `data/<tipo>_lexicon_v0.json` produz um
   conjunto de regiões candidatas via casamento regex.
2. Resolução de sobreposição segue a ordem de especificidade declarada
   em `oee-v1.yaml.resolucao_ambiguidade.ordem` (mais específico ganha
   sobre menos específico; sobreposições estritamente parciais
   também eliminam o candidato menos específico).
3. Texto remanescente entre regiões explícitas é emitido como
   `ProsaTecnica` quando contém pelo menos um caractere não-whitespace.

Decisão de design: optei por scanner regex priorizado em vez de
gramática lark porque a saída da Camada 2 é apenas a sequência de
regiões tipadas — não árvore sintática. Gramática lark passa a ser
útil quando a Camada 3 (instanciadores) precisar parsear estruturas
compositionais (e.g., `σ_y = 350 ± 10 MPa` como id + op + grandeza
incerta). A interface `classify(text) -> list[Region]` é estável.

Limitações declaradas:
- Separadores de milhar (`1.234,5` PT-BR / `1,234.5` US) não são
  parseados como número único. Canonicalização locale chega no
  instanciador `GrandezaFisicaInstantiator` (Dia 4).
- Operadores ASCII unários `+`, `-`, `*`, `/` não são emitidos —
  conflitam com unidades compostas e números. Cobertura adicional
  é decisão para versões posteriores da v1.x.
- ConstanteUniversal de letra única (`e`, `c`, `G`, `R`) emitida só
  em contexto math-like (delimitada por whitespace + `=` / `≈`)
  para evitar falsos positivos em prosa.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import regex as re

from toten.classifier.region import Region
from toten.ontology.loader import load_oee
from toten.ontology.schema import OEE
from toten.ontology.types import TipoNome

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PACKAGE_ROOT / "data"

DEFAULT_IDENTIFIER_LEXICON_PATH = DATA_DIR / "identifier_lexicon_v0.json"
DEFAULT_OPERATOR_LEXICON_PATH = DATA_DIR / "operator_lexicon_v0.json"
DEFAULT_CONSTANT_LEXICON_PATH = DATA_DIR / "constant_lexicon_v0.json"
DEFAULT_RELATION_LEXICON_PATH = DATA_DIR / "relation_lexicon_v0.json"

# Núcleo numérico — sinal opcional, mantissa com locale (PT-BR ou US),
# expoente opcional em três formas:
#   `eN`         — programming style: `1.5e-3`
#   `× 10ⁿ`      — Unicode superscript com ×, x, ·, * (engenharia formal)
#   `× 10^N`     — caret ASCII, com parens opcionais
#
# Mantissa aceita:
#   `1.234,56` PT-BR thousands `.` + decimal `,`
#   `1,234.56` US thousands `,` + decimal `.`
#   `1.234` PT-BR thousands sem decimal (heurística: 3 dígitos após `.`)
#   `287,4` PT-BR decimal simples
#   `287.4` US decimal simples
# A canonicalização para `float` acontece em `canonicalize_number` (Camada 3).
_NUMBER_RE = (
    r"[+\-−]?"
    r"(?:"
    r"10[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"                 # notação científica BR standalone: 10⁶, 10¹², 10⁻³
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"        # PT-BR thousands com . + decimal opcional com ,
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"       # US thousands com , + decimal opcional com .
    r"|\d+(?:[.,]\d+)?"                     # simples: inteiro ou decimal único
    r")"
    r"(?:"
    r"[eE][+\-−]?\d+"
    r"|\s*[×x·*]\s*10\s*[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"
    r"|\s*[×x·*]\s*10\s*\^\s*\(?\s*[+\-−]?\d+\s*\)?"
    r")?"
)

# Operador de incerteza: ± unicode ou `+/-` ASCII, com whitespace
# tolerante em torno.
_UNCERT_OP_RE = r"\s*(?:±|\+/-)\s*"

# Caracteres que indicam continuação de "unidade" — usado em lookahead
# negativo após o símbolo casado. Inclui letras Unicode, sobrescritos
# numéricos e caracteres especiais comuns em unidades compostas.
_UNIT_CONTINUATION = r"\p{L}|[²³⁰¹⁴⁵⁶⁷⁸⁹·°²]"


@dataclass(frozen=True, slots=True)
class _Candidate:
    """Região candidata antes de resolução de sobreposição."""

    tipo: TipoNome
    start: int
    end: int
    atomic: bool = False
    canonical_slug: str | None = None
    ambig_kind: str | None = None
    ambig_alternatives: tuple[str, ...] = ()

    @property
    def length(self) -> int:
        return self.end - self.start


# === SPEC_07 §2.1 — Catálogo `unit-letter` ===
#
# Letras ASCII maiúsculas que TAMBÉM são símbolos SI (A=ampere,
# V=volt, K=kelvin, W=watt, N=newton, J=joule, T=tesla, H=henry,
# F=farad, C=coulomb, S=siemens). Quando aparecem isoladas como
# unidade de 1 caractere em texto PT-BR, podem ser confundidas com
# letras de seção/capítulo. Listas FINITAS fechadas derivadas da
# gramática PT-BR.
_AMBIGUOUS_1CHAR_UPPER_UNITS: frozenset[str] = frozenset("AVKWNJTHFCS")

# Enumeradores PT-BR: palavras que precedem REFERÊNCIA numerada
# (`Tópico 2A`, `Item 3B`). Sinal forte: quando enumerador antecede
# `<num> <letra-ambígua>`, rejeitar QTY direto (estrutural inequívoco).
_PT_ENUMERATORS: frozenset[str] = frozenset([
    "tópico", "topico", "item", "capítulo", "capitulo",
    "seção", "secao", "subseção", "subsecao",
    "figura", "fig", "tabela", "tab",
    "exemplo", "ex", "exercício", "exercicio",
    "problema", "questão", "questao",
    "letra", "alínea", "alinea",
    "anexo", "apêndice", "apendice",
    "parte", "volume", "tomo", "livro",
    "quadro", "gráfico", "grafico",
    "esquema", "diagrama", "imagem", "foto",
])

# Enumeradores NORMATIVOS PT-BR (leis, normas, regulamentos):
# referenciam parágrafos/artigos/incisos, NUNCA designam medidas.
# Quando antecedente é normativo, REJEITA QTY independente da unidade
# (inclui Unicode como `°` em "§ 5°" que é ordinal mal-formado, não
# grau angular). Distinção semântica forte: textos legais não medem
# coisas, apenas referenciam estrutura.
_PT_ENUMERATORS_NORMATIVE: frozenset[str] = frozenset([
    "parágrafo", "paragrafo",
    "art", "art.", "artigo",
    "inciso", "incisos",
    "alínea", "alinea",
    "lei", "decreto", "norma",
    "súmula", "sumula",
    "regulamento", "regimento",
    "portaria", "resolução", "resolucao",
])

# Símbolo §  (U+00A7 SECTION SIGN) — tratado como enumerador normativo
# implícito quando aparece imediatamente antes do número (com ou sem
# espaço).
_SECTION_SIGN = "§"


def _is_preceded_by_normative_enumerator(text: str, start: int) -> bool:
    """SPEC_07 — antecedente é enumerador normativo (parágrafo, §, art.,
    inciso, alínea, lei, decreto, norma, etc.)?

    Detecta tanto palavra quanto símbolo `§` (com ou sem espaço).
    """
    prefix = text[:start].rstrip()
    # Símbolo § colado ou separado
    if prefix.endswith(_SECTION_SIGN):
        return True
    last_word = _last_word_before(text, start)
    return last_word is not None and last_word in _PT_ENUMERATORS_NORMATIVE


# === SPEC_07 §2 + OEE v1.2 — Referência hierárquica normativa ===
#
# Padrão estrutural: `\d+(\.\d+)+` (números separados por pontos)
# capturado APENAS quando precedido por enumerador normativo
# (parágrafo, §, art., inciso, alínea, …). Sem antecedente normativo,
# o mesmo padrão poderia ser número decimal ou versão de software —
# contexto determina interpretação inequívoca.
#
# Exemplos cobertos:
#   `§ 8.2.1`     → [REF:8.2.1]
#   `parágrafo 14.2.3`  → [REF:14.2.3]
#   `art. 5.3.1.2`      → [REF:5.3.1.2]
#   `inciso 1.2`        → [REF:1.2]
#   `8.2.1` standalone  → NÃO captura (poderia ser versão)
_REFERENCE_HIERARCHICAL_RE = re.compile(r"\d+(?:\.\d+)+")


# Enumeradores adicionais que comumente apresentam referência
# hierárquica (`item 4.3.1`, `capítulo 2.1`, `anexo 5.2`). Diferente
# dos NORMATIVOS, esses NÃO rejeitam QTY de qualquer unidade —
# apenas habilitam REF quando o padrão é hierárquico (`\d+(\.\d+)+`).
_PT_ENUMERATORS_HIERARCHICAL_CAPABLE: frozenset[str] = _PT_ENUMERATORS_NORMATIVE | frozenset([
    "item", "capítulo", "capitulo",
    "seção", "secao", "subseção", "subsecao",
    "anexo", "apêndice", "apendice",
    "tópico", "topico",
])


def _is_preceded_by_hierarchical_enumerator(text: str, start: int) -> bool:
    """Antecedente é enumerador que LICENCIA referência hierárquica?
    (normativos + alguns gerais como `item`, `capítulo`, `seção`)."""
    prefix = text[:start].rstrip()
    if prefix.endswith(_SECTION_SIGN):
        return True
    last_word = _last_word_before(text, start)
    return last_word is not None and last_word in _PT_ENUMERATORS_HIERARCHICAL_CAPABLE


def _find_normative_references(
    text: str,
) -> list[tuple[int, int, str]]:
    """SPEC_07 §2 — captura referências hierárquicas precedidas por
    enumerador que licencia referência (normativo OU hierárquico geral).

    Retorna lista de `(start, end, content)` para cada referência
    detectada estruturalmente. Conteúdo preserva forma original.
    """
    out: list[tuple[int, int, str]] = []
    for m in _REFERENCE_HIERARCHICAL_RE.finditer(text):
        if _is_preceded_by_hierarchical_enumerator(text, m.start()):
            out.append((m.start(), m.end(), m.group()))
    return out


def _is_after_sym_equation(text: str, start: int) -> bool:
    """SPEC_07 §2.1 Fase 2 — verifica se o match está IMEDIATAMENTE
    após uma equação `<letra(s)> [=≤≥≈<>] ` (com whitespace tolerante).

    Quando True, a unidade na posição é INEQUIVOCAMENTE técnica
    (contexto de equação matemática); suprime `ambig="unit-letter"`.

    Padrão: `<letra(s) Unicode + subscripts/primes>\\s*[op]\\s*$` ANTES
    do match. Operadores: `=`, `≤`, `≥`, `≈`, `<`, `>`, `≠`.

    Exemplos:
        `P = 5 W`     → True  (P + = antes)
        `σ_y = 350 N` → True  (σ_y + = antes)
        `consome 5 W` → False (palavra antes, sem operador relacional)
    """
    prefix = text[:start].rstrip()
    # `<token-simbólico>\s*[op]` no final do prefix
    return bool(
        re.search(
            r"[\p{L}_][\p{L}_0-9'″‴]*\s*[=≤≥≈<>≠]\s*$",
            prefix,
        )
    )

_UNIT_NAMES_BY_LETTER: dict[str, str] = {
    "A": "amperes (corrente elétrica)",
    "V": "volts (tensão elétrica)",
    "K": "kelvin (temperatura absoluta)",
    "W": "watts (potência)",
    "N": "newtons (força)",
    "J": "joules (energia)",
    "T": "teslas (campo magnético)",
    "H": "henries (indutância)",
    "F": "farads (capacitância)",
    "C": "coulombs (carga elétrica)",
    "S": "siemens (condutância)",
}


# === Âncoras técnicas por letra-unidade (SPEC_07 §2.1 sinal positivo) ===
#
# Quando palavra-âncora associada à dimensão física da unidade aparece
# na janela próxima ao QTY (antes ou depois), confirma USO TÉCNICO
# inequívoco e suprime `ambig="unit-letter"`.
#
# Implementação ROBUSTA via stemmer RSLP (NLTK) — Algoritmo Porter
# adaptado para PT-BR (Orengo & Huyck 2001), padrão canônico em NLP
# brasileiro. Reduz qualquer flexão (gênero, número, derivação) ao
# RADICAL lemmático. Listas abaixo contêm radicais pré-computados;
# matching no texto também passa pelo stemmer. Princípio respeitado:
# "oráculo externo > enumeração" (memória do projeto).
#
# Cada radical está unicamente associado à dimensão da unidade — não
# dicionário arbitrário. Cobertura inclui automaticamente todas as
# flexões reais PT-BR (frio/fria/frios/frias → todos `fri`; térmico/
# térmica/térmicos/térmicas → todos `térm`). Falsos positivos
# homofônicos não ocorrem (frita→`frit`, frigorífico→`frigoríf`
# distintos de `fri`).
_UNIT_TECHNICAL_ANCHOR_STEMS: dict[str, frozenset[str]] = {
    # Stems derivados via RSLPStemmer das formas lemmáticas
    # canonicalmente associadas a cada dimensão.
    "A": frozenset([
        "corr",         # corrente, correntes
        "intens",       # intensidade(s), intensifica
        "amp", "ampera",  # amperagem(ns), amperes
        "elétric", "eletric",  # elétrica/eletrica (atributo elétrico)
    ]),
    "V": frozenset([
        "tens",         # tensão, tensões
        "volt",         # voltagem, volts
        "potenc",       # potencial(is) — não confundir com "pot" de potência (W)
        "ddp",
        "elétric", "eletric",
    ]),
    "W": frozenset([
        "pot",          # potência, potências
        "dissip",       # dissipação, dissipações
        "consum",       # consumo, consumos
        "watt",         # watts
    ]),
    "N": frozenset([
        "forç",         # força, forças
        "pes",          # peso, pesos (cuidado: também 'pesos' R$ — geralmente ambíguo)
        "traç",         # tração, trações
        "newt",         # newtons
    ]),
    "K": frozenset([
        "temperat",    # temperatura, temperaturas
        "fri",          # frio, fria, frios, frias
        "quent",        # quente, quentes
        "térm", "term",  # térmico/a/s, termico/a/s
        "ambi",         # ambiente, ambientes (cuidado: ambiental, ambiente)
        "congel",       # congelamento(s), congelado, congela
        "fus",          # fusão, fusões (cuidado: fusível também)
        "ebul",         # ebulição, ebulições
        "gel",          # gelado, gelada, gelo
        "kelvin",
    ]),
    "J": frozenset([
        "energ",        # energia, energias, energético, energética
        "trabalh",      # trabalho, trabalhos
        "calor",        # calor, calores, calorífico, calorífica
        "joul",         # joules
    ]),
    "T": frozenset([
        "induç",        # indução, induções (cuidado: condução também)
        "magnét", "magnet",  # magnético/a/s, magnetico/a/s
        "camp",         # campo, campos (cuidado: muito genérico — manter)
        "tesl",         # teslas
    ]),
    "H": frozenset([
        "indut",        # indutância, indutâncias, indutor
        "henr",         # henries
    ]),
    "F": frozenset([
        "capacit",      # capacitância, capacitâncias, capacitor
        "capacid",      # capacidade, capacidades
        "farad",        # farads
    ]),
    "C": frozenset([
        "carg",         # carga, cargas (cuidado: também carga mecânica)
        "coulomb", "coul",
        "elétric", "eletric",  # carga elétrica
    ]),
    "S": frozenset([
        "condut",       # condutância, condutâncias (cuidado: também condução)
        "admit",        # admitância, admitâncias
        "siemen",       # siemens
    ]),
}

# Janela de busca por âncora ao redor do match (chars antes E depois).
# 150 chars cobre tipicamente o parágrafo próximo, permitindo que
# âncoras estejam separadas por palavras intermediárias, parênteses,
# vírgulas e outras estruturas naturais do texto técnico PT-BR.
_TECHNICAL_ANCHOR_WINDOW = 150

# Stemmer RSLP lazy-loaded (download uma vez se necessário).
_RSLP_STEMMER = None


def _get_rslp_stemmer():
    """Lazy load do RSLPStemmer. Baixa dados RSLP do NLTK se ausente."""
    global _RSLP_STEMMER
    if _RSLP_STEMMER is None:
        from nltk.stem import RSLPStemmer
        try:
            _RSLP_STEMMER = RSLPStemmer()
        except LookupError:
            import nltk
            nltk.download("rslp", quiet=True)
            _RSLP_STEMMER = RSLPStemmer()
    return _RSLP_STEMMER


# Pattern para extrair tokens-palavra do texto (latina + acentuação PT-BR).
_WORD_TOKEN_RE = re.compile(r"\p{L}+")


def _has_technical_anchor_nearby(
    text: str, match_start: int, match_end: int, unit_letter: str
) -> bool:
    """Verifica se palavra cujo RADICAL lemmático pertence ao conjunto
    de âncoras da dimensão aparece na janela próxima ao match.

    Estrutural via stemmer RSLP (oráculo externo PT-BR): cada palavra
    do texto é reduzida ao seu radical e comparada contra o conjunto
    pré-computado da dimensão. Cobre flexões automaticamente sem
    enumerar manualmente; falsos positivos homofônicos (`frita`,
    `frigorífico`) ficam excluídos por terem stems distintos de `fri`.
    """
    stems = _UNIT_TECHNICAL_ANCHOR_STEMS.get(unit_letter)
    if not stems:
        return False
    before_start = max(0, match_start - _TECHNICAL_ANCHOR_WINDOW)
    after_end = min(len(text), match_end + _TECHNICAL_ANCHOR_WINDOW)
    window_text = (
        text[before_start:match_start] + " " + text[match_end:after_end]
    ).lower()
    stemmer = _get_rslp_stemmer()
    for word_match in _WORD_TOKEN_RE.finditer(window_text):
        word = word_match.group()
        if stemmer.stem(word) in stems:
            return True
    return False


def _last_word_before(text: str, pos: int) -> str | None:
    """Devolve a última palavra (lowercase) antes da posição `pos`,
    ou None se não houver palavra-precedente significativa.

    Tolerante a pontuação (`.`, `,`, `;`, `:`) que possa aparecer ao
    final da palavra: `art.` → `art`; `inciso,` → `inciso`.
    """
    prefix = text[:pos].rstrip()
    m = re.search(r"(\w+)[.,;:]*\s*$", prefix)
    return m.group(1).lower() if m else None


def _is_preceded_by_enumerator(text: str, start: int) -> bool:
    """SPEC_07 §2.1: antecedente é enumerador PT-BR?"""
    last_word = _last_word_before(text, start)
    return last_word is not None and last_word in _PT_ENUMERATORS


def _detect_unit_letter_ambiguity(
    text: str, match_start: int, match_end: int, unit_text: str
) -> tuple[bool, str | None, tuple[str, ...]]:
    """SPEC_07 §2.1 — detecta dimensão `unit-letter` e contexto normativo.

    Retorna `(reject, ambig_kind, alternatives)`:
    - `reject=True`: candidato deve ser REMOVIDO (não emite QTY)
    - `reject=False, ambig_kind="unit-letter"`: emite QTY com marca
    - `reject=False, ambig_kind=None`: emite QTY sem marca (caso claro)

    Critérios de rejeição (qualquer um → reject=True):
    1. Antecedente é enumerador normativo (parágrafo, §, art., etc.)
       → referência legal, NUNCA é medida (qualquer unit afetada)
    2. Antecedente é enumerador geral (Tópico, Item, ...) E unit é
       1-char ASCII MAIÚSC ambíguo → referência subdivisional

    Critérios de supressão de ambig (sem rejeitar):
    3. Antecedente é equação matemática (`<sym> = `) → unit confirmado
       pelo contexto técnico inequívoco

    Default (sem rejeição nem supressão): emite ambig=unit-letter para
    1-char ASCII MAIÚSC (preserva incerteza para diálogo agente).
    """
    # SINAL 1 — enumerador normativo (parágrafo, §, art., …) rejeita
    # QTY de QUALQUER unidade. Textos legais/normativos não medem;
    # apenas referenciam estrutura. Cobre `§ 5°` (não é grau angular).
    if _is_preceded_by_normative_enumerator(text, match_start):
        return True, None, ()

    # Sinal aplica apenas a 1-char ASCII MAIÚSC ambíguo
    if len(unit_text) != 1 or unit_text not in _AMBIGUOUS_1CHAR_UPPER_UNITS:
        return False, None, ()

    # SINAL 2 — enumerador geral + unidade ambígua: referência subdivisional
    if _is_preceded_by_enumerator(text, match_start):
        return True, None, ()

    # SINAL 3 — equação matemática antes: SYM ou letra(s) seguida de
    # operador relacional confirma contexto técnico. Suprime `ambig`.
    if _is_after_sym_equation(text, match_start):
        return False, None, ()

    # SINAL 4 — âncora técnica da dimensão na janela próxima: confirma
    # uso técnico inequívoco. `temperatura igual a 40 K` → temperatura
    # ancora K=kelvin; suprime ambig. Janela cobre ~sentença próxima.
    if _has_technical_anchor_nearby(text, match_start, match_end, unit_text):
        return False, None, ()

    # Caso default: marca ambiguidade. Alternativas catalogadas:
    unit_descr = _UNIT_NAMES_BY_LETTER[unit_text]
    alternatives = (
        f"unit:{unit_descr}",
        f"reference:letra_{unit_text}_em_referência",
    )
    return False, "unit-letter", alternatives


# ---------------------------------------------------------------------------
# Helpers de carga
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_unit_symbols_from_dim_table() -> list[str]:
    """Carrega símbolos atômicos de unidade do oráculo completo.

    Fontes (responsabilidade única por domínio):
    - dim_table.json: unidades SI dimensionais (ℤ⁷) + auxiliares mesclados
      via load_dim_table (dimensionless + info)
    - unit_aliases_v0.json: chaves do mapeamento ASCII/notacional →
      canônico (e.g., `Torr` → `torr`, `mmH₂O` → `mmH2O`, `KB` → `kB`)
      são adicionadas para que o regex reconheça AMBAS formas; o parser
      depois normaliza para a forma canônica antes da decomposição.

    Composições (`kgf/cm²`, `J/(kg·K)`, `tf·m`) NÃO precisam ser
    enumeradas — são computadas pelo regex compositional.
    """
    from toten.dimensional.aliases import default_unit_aliases
    from toten.dimensional.table import default_dim_table
    symbols = set(default_dim_table().atoms.keys())
    # Adiciona chaves de aliases (formas notacionais variantes
    # reconhecidas pelo classifier; valor canônico já é/será atom).
    symbols.update(default_unit_aliases().string_aliases.keys())
    return sorted(symbols)


def default_units() -> tuple[str, ...]:
    """Retorna a tupla canônica de símbolos atômicos derivada da dim_table."""
    return tuple(_load_unit_symbols_from_dim_table())


def _build_unit_alternation(symbols: Iterable[str]) -> str:
    ordered = sorted(set(symbols), key=lambda s: (-len(s), s))
    return "|".join(re.escape(s) for s in ordered)


def _build_literal_alternation(literals: Iterable[str]) -> str:
    ordered = sorted(set(literals), key=lambda s: (-len(s), s))
    return "|".join(re.escape(s) for s in ordered)


def _build_unit_compositional_pattern(symbols: Iterable[str]) -> str:
    """Regex que casa unidade-literal OU composição de átomos conhecidos.

    Cada átomo é um símbolo da lista, possivelmente seguido de potência
    (superscript Unicode `²³…⁻¹` incluindo sinal, ou caret `^N` / `^-N`).
    Termos podem ser átomos ou expressões parentizadas.

    **Operadores composicionais aceitos (norma SI BIPM 9th ed. §5.2):**
    - `·` ponto centrado (canônico)
    - `*` asterisco (notação ASCII)
    - `/` divisão
    - espaço simples ` ` (também canônico — ISO 80000-1 / BIPM permite
      `m s⁻²` equivalente a `m·s⁻²`); o regex só promove espaço a
      operador quando o token seguinte é átomo do oráculo, então
      prosa após unidade não é capturada acidentalmente.

    Guardrail anti-falso-positivo: TODO átomo da composição precisa estar
    no conjunto `symbols`. `350 MPa·foo` não casa porque `foo` não é
    átomo conhecido; `10 m está` não casa porque `está` não é átomo;
    `10 m s⁻²` casa inteiro porque `m`, `s` estão no oráculo.
    """
    literal_alt = _build_unit_alternation(symbols)
    # Expoente unicode (²³⁻¹), caret (^N, ^-N), OU dígito ASCII justaposto
    # (m2, cm3, mm2). O dígito ASCII cobre a notação técnica brasileira
    # antiga (kgf/cm2, t/m3) sem precisar de aliases por composição.
    atomic = (
        rf"(?:{literal_alt})"
        rf"(?:[⁻⁺]?[²³⁰¹⁴⁵⁶⁷⁸⁹]+|\^-?\d+|[2-9])?"
    )
    # Espaço dentro de parens NÃO conecta — força operador explícito ali
    # para legibilidade (parens já são marcador de subexpressão estruturada).
    parens = rf"\((?:{atomic}(?:[·*/]{atomic})*)\)"
    term = rf"(?:{atomic}|{parens})"
    # Conector entre termos: `·`, `*`, `/`, OU 1+ espaços ASCII (BIPM §5.2).
    # `\s+` excluiríamos quebras de linha — usamos só ` ` literal para
    # restringir a separador horizontal único de unidade composta.
    connector = r"(?:[·*/]| +)"
    return rf"{term}(?:{connector}{term})*"


# ---------------------------------------------------------------------------
# Classificador
# ---------------------------------------------------------------------------


class OntologicalClassifier:
    """Scanner ontológico v0 — Dia 3.

    Detecta GrandezaFisica, ConstanteUniversal, IdentificadorTecnico,
    OperadorFormal, RelacaoEstrutural e emite ProsaTecnica em gaps.
    Prioridade de resolução de sobreposição vem de `oee-v1.yaml`.
    """

    def __init__(
        self,
        oee: OEE | None = None,
        unit_symbols: list[str] | None = None,
        identifier_lexicon_path: Path | None = None,
        operator_lexicon_path: Path | None = None,
        constant_lexicon_path: Path | None = None,
        relation_lexicon_path: Path | None = None,
    ) -> None:
        self._oee = oee if oee is not None else load_oee()
        self._priority: dict[TipoNome, int] = {
            tipo: rank for rank, tipo in enumerate(self._oee.resolucao_ambiguidade.ordem)
        }

        symbols = (
            list(unit_symbols)
            if unit_symbols is not None
            else _load_unit_symbols_from_dim_table()
        )
        if not symbols:
            msg = "OntologicalClassifier exige ao menos um símbolo de unidade"
            raise ValueError(msg)
        self._unit_symbols = tuple(symbols)

        self._grandeza_re = self._build_grandeza_pattern(symbols)
        # v1.1: pattern complementar para unit-only (sem value antecedente)
        self._unit_only_re = self._build_unit_only_pattern(symbols)
        # v1.1: pattern para número solto (Numero — 8º tipo OEE)
        self._numero_re = self._build_numero_pattern()
        self._identifier_res = self._build_identifier_patterns(
            identifier_lexicon_path or DEFAULT_IDENTIFIER_LEXICON_PATH
        )
        self._operator_re, self._operator_eq_re = self._build_operator_patterns(
            operator_lexicon_path or DEFAULT_OPERATOR_LEXICON_PATH
        )
        (
            self._constant_multi_re,
            self._constant_single_re,
        ) = self._build_constant_patterns(
            constant_lexicon_path or DEFAULT_CONSTANT_LEXICON_PATH
        )
        self._relation_re = self._build_relation_pattern(
            relation_lexicon_path or DEFAULT_RELATION_LEXICON_PATH
        )

    # ------------------------------------------------------------------
    # Compilação de padrões
    # ------------------------------------------------------------------

    def _build_grandeza_pattern(self, symbols: list[str]) -> re.Pattern:
        unit_pattern = _build_unit_compositional_pattern(symbols)
        # Lookbehind impede iniciar número dentro de token contíguo:
        # `4V` em `Ti-6Al-4V` não vira GrandezaFisica.
        # Range `<num>-<num> <unit>` (estrutura intervalo BR) é capturado
        # como UMA região; o instantiator emite dois QTYs adjacentes
        # com `-` literal entre eles.
        pattern = (
            rf"(?<![\p{{L}}\d\-_])"
            rf"(?P<num>{_NUMBER_RE})"
            rf"(?:\s*-\s*(?P<num2>{_NUMBER_RE}))?"
            rf"(?:{_UNCERT_OP_RE}(?P<unc>{_NUMBER_RE}))?"
            rf"\s*"
            rf"(?P<unit>{unit_pattern})"
            rf"(?!{_UNIT_CONTINUATION})"
        )
        return re.compile(pattern)

    def _build_numero_pattern(self) -> re.Pattern:
        """Pattern para número solto (v1.1 — 8º tipo OEE).

        Detecta valores numéricos puros (sem unidade física adjacente)
        em texto livre, com suporte locale-rich PT-BR + notação científica
        Unicode/ASCII + sinal negativo + fração simples + percentual.

        Endereça lacuna SOTA Number Cookbook (Yang ICLR 2025) — números
        soltos sem unidade, frequentes em texto técnico e de avaliação.

        Reusa `_NUMBER_RE` como núcleo (já testado em GrandezaFisica) e
        adiciona variantes:
        - Fração `n/d` (3/4, 22/7) — apenas inteiros simples sem locale
        - Percentual `X%` (50%, 3,5%)

        Lookbehind/lookahead críticos (anti-falso-positivo):
        - `(?<![\\p{L}\\d_])` — não dentro de token contíguo
          (`AISI1045`, `iso9001` ficam intactos)
        - Negativo lookahead após o número: NÃO seguido por unidade
          (esse caso é capturado por _grandeza_re como QTY completa)
        """
        # Núcleo: _NUMBER_RE + fração + percentual + ordinal
        # Fração ontológica: razão entre inteiros pequenos. Estruturalmente,
        # fração matemática em texto técnico opera com numerador 1-3 dígitos
        # e denominador 1-3 dígitos. Denominadores ≥4 dígitos (`12/2024`,
        # `15/1989`) são referências temporais/normativas, NÃO frações —
        # têm domínio ontológico distinto (referência, não razão). Limitar
        # estruturalmente evita colisão polimórfica do operador `/`.
        # Word boundary explícito no final do denominador: impede match
        # parcial em referência multi-dígito (`12/2024` → `12/202`).
        fraction = r"[+\-−]?\d{1,3}\s*/\s*\d{1,3}(?!\d)"
        percent = rf"(?:{_NUMBER_RE})%"
        # Ordinal: dígito(s) seguido(s) de marcador UCD `ª` (FEMININE
        # ORDINAL INDICATOR) ou `º` (MASCULINE ORDINAL INDICATOR).
        # `°` (DEGREE SIGN) explicitamente excluído — pertence a outro
        # domínio ontológico (grau angular = QTY).
        ordinal = r"[+\-−]?\d+[ªº]"
        # Padrão final: qualquer das variantes (ordinal antes para
        # ganhar prioridade sobre `\d+` puro do _NUMBER_RE).
        num_pattern = rf"(?:{ordinal}|{percent}|{fraction}|{_NUMBER_RE})"
        # Anti-falso-positivo: não dentro de token contíguo, não imediatamente
        # seguido por unidade (que vira QTY completa via _grandeza_re).
        # Lookahead permissivo: aceita whitespace + unit (deixa _grandeza_re
        # ganhar por length); rejeita unit imediatamente colado.
        pattern = (
            rf"(?<![\p{{L}}\d_])"
            rf"({num_pattern})"
        )
        return re.compile(pattern)

    def _build_unit_only_pattern(self, symbols: list[str]) -> re.Pattern:
        """Pattern para unidade-isolada sem valor numérico antecedente.

        v1.1 (P5 extensibilidade da OEE) — emite Region(GrandezaFisica,
        value=None) para strings como `"kgf/cm²"`, `"kN·m"`, `"m/s"`.

        Lookbehind crítico (anti-falso-positivo):
        - `(?<![\\p{L}\\d\\-_])` — não dentro de token contíguo
        - `(?<![\\d][\\s])` — não imediatamente após um número (esse
          caso já é coberto por _grandeza_re como QTY completa)

        Quando existe `350 MPa`, _grandeza_re casa "350 MPa" inteiro;
        _unit_only_re tentaria casar "MPa" mas é bloqueado pelo
        lookbehind (precedido por dígito+espaço). Mesmo se casasse,
        o overlap resolver escolheria a região mais longa (full QTY).

        Lookahead: não seguido de continuação de unit (evita parsing
        parcial de composições).
        """
        unit_pattern = _build_unit_compositional_pattern(symbols)
        pattern = (
            rf"(?<![\p{{L}}\d\-_])"
            rf"(?<!\d\s)"
            rf"(?<!\d)"  # exclui `4kgf` (sem espaço) também
            rf"(?P<unit>{unit_pattern})"
            rf"(?!{_UNIT_CONTINUATION})"
        )
        return re.compile(pattern)

    def _build_identifier_patterns(
        self, path: Path
    ) -> list[tuple[re.Pattern, str | None]]:
        """Compila padrões de identificador com canonical_slug opcional.

        Cada entry do lexicon pode declarar `canonical_slug` (string fixa)
        para casos onde a CLASSE normativa tem canônico único (ex.: material
        com variantes notacionais). Quando ausente, o slug é derivado do
        conteúdo capturado via canonicalize_identifier.
        """
        payload = _load_json(path)
        patterns = []
        for entry in payload.get("padroes", []):
            regex = entry.get("regex")
            if not regex:
                continue
            canonical_slug = entry.get("canonical_slug")  # opcional
            patterns.append((re.compile(regex), canonical_slug))
        return patterns

    def _build_operator_patterns(self, path: Path) -> tuple[re.Pattern, re.Pattern]:
        payload = _load_json(path)
        unicode_symbols: list[str] = []
        ascii_eq_present = False
        for entry in payload.get("operadores", []):
            simbolo = entry.get("simbolo")
            if not simbolo:
                continue
            if simbolo == "=":
                ascii_eq_present = True
                continue
            unicode_symbols.append(simbolo)
        if not unicode_symbols:
            msg = "operator_lexicon sem operadores Unicode"
            raise ValueError(msg)
        unicode_alt = _build_literal_alternation(unicode_symbols)
        unicode_re = re.compile(unicode_alt)
        # `=` ASCII só conta quando não está em token contíguo (lookbehind/ahead
        # negam letras e dígitos — evita acertar `==`, `!=`, `x=1`).
        eq_pattern = r"(?<![=\p{L}\d])=(?![=\p{L}\d])" if ascii_eq_present else r"(?!x)x"
        eq_re = re.compile(eq_pattern)
        return unicode_re, eq_re

    def _build_constant_patterns(
        self, path: Path
    ) -> tuple[re.Pattern, re.Pattern]:
        payload = _load_json(path)
        multi: list[str] = []
        single: list[str] = []
        for entry in payload.get("constantes", []):
            simbolo = entry.get("simbolo")
            if not simbolo:
                continue
            if entry.get("requer_contexto"):
                single.append(simbolo)
            else:
                multi.append(simbolo)

        multi_alt = _build_literal_alternation(multi) if multi else r"(?!x)x"
        # Princípio ontológico (uso vs menção):
        # CONST é entidade nomeada referenciada como objeto (constante universal
        # com valor canônico). Quando adjacente a marca matemática (`²³…`, `·`,
        # `*`, `/`, `^`), é operando de composição P3 — parte de SYM, não CONST.
        # Lookahead/lookbehind excluem contextos de composição matemática.
        # `\d` no lookbehind: aceita `2π` (coeficiente justaposto, cluster SYM
        # ainda engloba).
        _MATH_ADJACENT = r"·*/^²³⁰¹⁴⁵⁶⁷⁸⁹"
        multi_re = re.compile(
            rf"(?<![\p{{L}}_{_MATH_ADJACENT}])(?:{multi_alt})"
            rf"(?![\p{{L}}\d_{_MATH_ADJACENT}])"
        )

        if single:
            single_alt = _build_literal_alternation(single)
            # contexto math-like: símbolo adjacente a `=` ou `≈` (com whitespace
            # tolerante de cada lado), evitando emparelhar prosa.
            single_pattern = (
                rf"(?<![\p{{L}}\d_{_MATH_ADJACENT}])(?:{single_alt})"
                rf"(?![\p{{L}}\d_{_MATH_ADJACENT}])(?=\s*[=≈])"
                rf"|"
                rf"(?<=[=≈]\s)(?:{single_alt})(?![\p{{L}}\d_{_MATH_ADJACENT}])"
            )
            single_re = re.compile(single_pattern)
        else:
            single_re = re.compile(r"(?!x)x")
        return multi_re, single_re

    def _build_relation_pattern(self, path: Path) -> re.Pattern:
        payload = _load_json(path)
        formas: list[str] = []
        for grupo in ("pt-br", "en"):
            for entry in payload.get(grupo, []):
                forma = entry.get("forma")
                if forma:
                    formas.append(forma)
        if not formas:
            return re.compile(r"(?!x)x")
        alt = _build_literal_alternation(formas)
        # case-insensitive, com word-boundary à esquerda e à direita
        return re.compile(rf"(?<![\p{{L}}])(?:{alt})(?![\p{{L}}])", flags=re.IGNORECASE)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def unit_symbols(self) -> tuple[str, ...]:
        return self._unit_symbols

    @property
    def priority(self) -> dict[TipoNome, int]:
        """Ordem de especificidade (menor índice = mais específico)."""
        return dict(self._priority)

    def classify(self, text: str) -> list[Region]:
        """Retorna regiões ordenadas por posição, sem sobreposição.

        Inclui `ProsaTecnica` em gaps com conteúdo não-whitespace.
        """
        if not text:
            return []

        candidates: list[_Candidate] = []

        # OEE v1.2 — Referências hierárquicas normativas (§ 8.2.1).
        # Capturadas ANTES de QTY/NUM para ganhar prioridade no overlap
        # resolver. Sem antecedente normativo, padrão `\d+(\.\d+)+`
        # não captura — fica disponível para NUM/QTY decompor.
        for start, end, _content in _find_normative_references(text):
            candidates.append(
                _Candidate(TipoNome.REFERENCIA, start, end)
            )

        for m in self._grandeza_re.finditer(text):
            # SPEC_07 §2.1 — detecção de ambiguidade `unit-letter`
            matched = m.group()
            unit_part = matched.rsplit(maxsplit=1)
            unit_text = unit_part[1] if len(unit_part) == 2 else ""
            reject, ambig_kind, alternatives = _detect_unit_letter_ambiguity(
                text, m.start(), m.end(), unit_text
            )
            if reject:
                continue  # enumerador antes → não emite QTY (referência)
            candidates.append(
                _Candidate(
                    TipoNome.GRANDEZA_FISICA,
                    m.start(),
                    m.end(),
                    ambig_kind=ambig_kind,
                    ambig_alternatives=alternatives,
                )
            )
        # v1.1: unit-only candidates — overlap com QTY completa resolve por
        # length (full QTY > unit-only sub-match), mesma prioridade.
        #
        # Filtro ONTOLÓGICO (não-heurístico): unit-only QTY representa
        # dimensão pura ISOLADA de magnitude. Para distinguir
        # inequivocamente do PROSA (palavra natural), exige MARCADOR
        # ESTRUTURAL TÉCNICO intrínseco à notação de unidade:
        #
        #   - operador composicional (`/`, `·`, `*`, `^`)
        #   - superscript Unicode (`²`, `³`, `⁻¹`, ...)
        #   - char não-ASCII (`°`, `Ω`, `μ`, `Å`, ...)
        #
        # Sem ao menos um desses marcadores, um lexema alfa-ASCII puro
        # (e.g., "Em", "tf", "min", "kg") é estruturalmente
        # indistinguível de prosa PT-BR e fica para PROSA. O caso "350
        # MPa" segue funcionando como QTY-com-value (não é unit-only).
        # O caso "kgf/cm²", "kN·m", "m/s²", "°C" segue como unit-only
        # (todos têm marcadores). Esta é a definição ontológica formal
        # de "notação técnica isolada", não regra ad-hoc.
        _TECH_MARKERS = frozenset("·*/^²³⁻⁺⁰¹⁴⁵⁶⁷⁸⁹°ΩμÅ")
        for m in self._unit_only_re.finditer(text):
            unit_text = m.group()
            has_marker = any(c in _TECH_MARKERS for c in unit_text)
            has_non_ascii = any(ord(c) > 127 for c in unit_text)
            if has_marker or has_non_ascii:
                candidates.append(
                    _Candidate(TipoNome.GRANDEZA_FISICA, m.start(), m.end())
                )
        # v1.1: Numero candidates — números soltos (8º tipo OEE).
        # Overlap com GrandezaFisica (full QTY ou unit-only) resolve por
        # priority: GRANDEZA_FISICA > NUMERO na ordem de especificidade
        # da OEE v1.1. Logo "350" em "350 MPa" fica como parte do QTY,
        # não Numero standalone.
        for m in self._numero_re.finditer(text):
            candidates.append(
                _Candidate(TipoNome.NUMERO, m.start(), m.end())
            )
        for m in self._constant_multi_re.finditer(text):
            candidates.append(
                _Candidate(TipoNome.CONSTANTE_UNIVERSAL, m.start(), m.end())
            )
        for m in self._constant_single_re.finditer(text):
            candidates.append(
                _Candidate(TipoNome.CONSTANTE_UNIVERSAL, m.start(), m.end())
            )
        for pat, canonical_slug in self._identifier_res:
            for m in pat.finditer(text):
                candidates.append(
                    _Candidate(
                        TipoNome.IDENTIFICADOR_TECNICO,
                        m.start(),
                        m.end(),
                        canonical_slug=canonical_slug,
                    )
                )
        for m in self._operator_re.finditer(text):
            candidates.append(_Candidate(TipoNome.OPERADOR_FORMAL, m.start(), m.end()))
        for m in self._operator_eq_re.finditer(text):
            candidates.append(_Candidate(TipoNome.OPERADOR_FORMAL, m.start(), m.end()))
        for m in self._relation_re.finditer(text):
            candidates.append(
                _Candidate(TipoNome.RELACAO_ESTRUTURAL, m.start(), m.end())
            )

        # ExpressaoSimbolica por DERIVAÇÃO unificada (refatoração 2026-05-28):
        # Pipeline em 4 etapas com critério ontológico único de 6 condições
        # (5 marcas tipográficas intrínsecas + 1 propagação contextual).
        # Ver toten/classifier/derivation.py para o critério formal.
        from toten.classifier.derivation import derive_sym_regions

        # Resolver overlaps DENTRO da Camada 1 primeiro (átomos limpos).
        atoms_layer1 = self._resolve_overlaps(candidates)
        atom_spans = [(c.start, c.end) for c in atoms_layer1]
        atom_tipos = [c.tipo.value for c in atoms_layer1]

        sym_emergent = derive_sym_regions(text, atom_spans, atom_tipos)
        for s_start, s_end, s_atomic in sym_emergent:
            candidates.append(
                _Candidate(
                    TipoNome.EXPRESSAO_SIMBOLICA,
                    s_start,
                    s_end,
                    atomic=s_atomic,
                )
            )

        # Resolver overlap FINAL: SYM cobre átomos internos (átomos
        # viram parte do cluster ou são absorvidos por delimitado).
        resolved = self._resolve_overlaps(candidates)
        regions = self._emit_with_residual(text, resolved)
        return regions

    # ------------------------------------------------------------------
    # Resolução e residual
    # ------------------------------------------------------------------

    def _resolve_overlaps(self, candidates: list[_Candidate]) -> list[_Candidate]:
        """Mantém candidatos não-sobrepostos resolvendo ambiguidades em
        duas fases:

        1. **Containment override** — se A contém estritamente B (mesmo
           span no fim ou no início, mas A é maior) e são de tipos
           diferentes, B é descartado independente da priority. Argumento:
           o candidato mais longo engloba o curto com mais contexto, é a
           interpretação mais específica. Resolve casos como `SAE 4140 H`
           (IDX contém GF acidental de `4140 H` interpretado como Henry)
           sem regras pontuais por par de tipos.

        2. **Priority + length desc + start asc** — para overlaps parciais
           (sem containment estrito), priority asc da OEE vence (maior
           especificidade ontológica); empate por span mais longo; empate
           final por posição.
        """
        survivors = self._apply_containment_override(candidates)
        ordered = sorted(
            survivors,
            key=lambda c: (self._priority[c.tipo], -c.length, c.start),
        )
        accepted: list[_Candidate] = []
        for cand in ordered:
            if not any(self._overlaps(cand, a) for a in accepted):
                accepted.append(cand)
        accepted.sort(key=lambda c: c.start)
        return accepted

    def _apply_containment_override(
        self,
        candidates: list[_Candidate],
    ) -> list[_Candidate]:
        """Descarta candidatos em containment estrito.

        Critério: outer.start <= inner.start AND outer.end >= inner.end
        AND ao menos uma desigualdade é estrita.

        Regras (arquitetura em camadas pós-refatoração 2026-05-28):
        - Caso geral: descarta INNER. Outer mais longo é interpretação
          mais específica em contexto (`SAE 4140 H` IDX contém `4140 H`
          GF acidental → descarta GF interno).
        - **SYM sempre vence átomos internos**: tanto cluster (atomic=False)
          quanto atomic pattern (atomic=True) cobrem seus átomos da
          Camada 1. Cluster cobre porque a composição é a entidade
          ontológica relevante. Atomic absorve porque a aplicação de
          função/notação atômica é mediador P3 indissociável.
        """
        survivors: list[_Candidate] = []
        for cand in candidates:
            discarded = False
            for other in candidates:
                if cand is other:
                    continue
                if other.tipo == cand.tipo:
                    continue
                if (
                    other.start <= cand.start
                    and other.end >= cand.end
                    and (other.start < cand.start or other.end > cand.end)
                ):
                    # cand está estritamente contido em other → descarta cand.
                    discarded = True
                    break
            if not discarded:
                survivors.append(cand)
        return survivors

    @staticmethod
    def _overlaps(a: _Candidate, b: _Candidate) -> bool:
        return a.start < b.end and b.start < a.end

    def _emit_with_residual(
        self, text: str, accepted: list[_Candidate]
    ) -> list[Region]:
        regions: list[Region] = []
        cursor = 0
        for cand in accepted:
            if cand.start > cursor:
                self._maybe_emit_prosa(text, cursor, cand.start, regions)
            regions.append(
                Region(
                    tipo=cand.tipo,
                    start=cand.start,
                    end=cand.end,
                    content=text[cand.start : cand.end],
                    canonical_slug=cand.canonical_slug,
                    ambig_kind=cand.ambig_kind,
                    ambig_alternatives=cand.ambig_alternatives,
                )
            )
            cursor = cand.end
        if cursor < len(text):
            self._maybe_emit_prosa(text, cursor, len(text), regions)
        return regions

    @staticmethod
    def _maybe_emit_prosa(
        text: str, start: int, end: int, regions: list[Region]
    ) -> None:
        fragment = text[start:end]
        if fragment.strip():
            regions.append(
                Region(
                    tipo=TipoNome.PROSA_TECNICA,
                    start=start,
                    end=end,
                    content=fragment,
                )
            )
