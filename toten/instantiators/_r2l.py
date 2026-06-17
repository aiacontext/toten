"""R2L (Right-to-Left) digit grouping helper.

Adições no Modo B das tags QTY/NUM endereçando lacuna SOTA Number
Cookbook (Yang ICLR 2025) e validando Singh-Strouse 2024 ("Tokenization
counts: the impact of tokenization on arithmetic in frontier LLMs",
arXiv:2402.14903).

**Achado SOTA:** R2L grouping em números multi-dígitos sobe GPT-3.5 de
75.6% → 97.8% em soma. Tokenizers BPE genéricos fragmentam números de
forma idiossincrática (1234567 pode virar [12, 345, 67] dependendo do
vocab); R2L explicita a estrutura digital esperada pela aritmética
posicional (unidades, dezenas, centenas, milhares, ...).

**Decisão de design (separador):** Usamos ESPAÇO como separador R2L
(não `.` ou `,`) — neutralizamos conflito com locale PT-BR/US e
sinalizamos visualmente "estes são dígitos agrupados, não milhares".
Exemplo: `1234567.89` → r2l=`"1 234 567.89"`. Parte fracional fica
intocada (R2L se aplica à parte inteira posicional).

**Quando emitir r2l:** apenas quando agrupamento muda visivelmente a
string — ou seja, ≥4 dígitos na parte inteira (|value| ≥ 1000) OU
notação científica com mantissa multi-dígito. Para números pequenos
(`100`, `0.5`), r2l é igual ao value canônico — omitir economiza
verbosidade.

Reusa sem modificar `canonicalize_number` e `_format_value` existentes.
"""

from __future__ import annotations


def r2l_group(value: float | int) -> str | None:
    """Devolve representação R2L-agrupada por espaço, OU `None` quando
    o agrupamento seria visualmente idêntico ao valor canônico.

    Ex.:
        100       → None   (sem agrupamento; "100" já é claro)
        1234      → "1 234"
        1234567   → "1 234 567"
        -1234567  → "-1 234 567"
        1234.56   → "1 234.56"
        0.5       → None
        0.0015    → None   (parte inteira = 0, sem agrupamento)
        1.5e-3    → None   (científica — agrupamento na mantissa não
                              ajuda; magnitude está no expoente)
        12345.67890123 → "12 345.67890123"
    """
    if value is None:
        return None

    # Notação científica não se beneficia de R2L na mantissa
    abs_v = abs(float(value))
    if abs_v != 0 and (abs_v >= 1e15 or abs_v < 1e-3):
        return None

    # Separar parte inteira e fracional preservando sinal
    is_int = float(value) == int(value) and abs_v < 1e15
    if is_int:
        int_part = str(int(value))
        frac_part = ""
    else:
        # Representação canônica via repr (preserva precisão IEEE 754)
        s = repr(float(value))
        # repr pode dar "1.0" para inteiros float — normaliza
        if "." in s:
            int_part, frac_part = s.split(".", 1)
            frac_part = "." + frac_part
        else:
            int_part = s
            frac_part = ""

    # Manipular sinal separadamente
    sign = ""
    if int_part.startswith("-"):
        sign = "-"
        int_part = int_part[1:]
    elif int_part.startswith("+"):
        int_part = int_part[1:]

    # Sem agrupamento necessário (≤3 dígitos na parte inteira)
    if len(int_part) <= 3:
        return None

    # R2L grouping da parte inteira
    chunks = []
    while len(int_part) > 3:
        chunks.append(int_part[-3:])
        int_part = int_part[:-3]
    chunks.append(int_part)
    grouped_int = " ".join(reversed(chunks))

    return f"{sign}{grouped_int}{frac_part}"
