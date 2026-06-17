"""Tipos enumerativos da OEE (v1.0 base + v1.1 extensão).

A enumeração `TipoNome` fixa o conjunto de tipos ontológicos. Extensões
acrescentam novos membros via P5 (extensibilidade); remoções exigem
versão major nova da ontologia.

CHANGELOG:
- v1.0 (2026-05-19): 7 tipos canônicos (ExpressaoSimbolica adicionada).
- v1.1 (2026-05-31): 8º tipo `NUMERO` adicionado para escalares puros
  sem dimensão física (reconstrução numérica + lacuna SOTA Number
  Cookbook Yang ICLR 2025). GRANDEZA_FISICA agora permite valor=None
  (unidade-isolada) — não muda o enum, mudança é no instantiator.
- v1.2 (2026-06-01): 9º tipo `REFERENCIA` adicionado para referências
  hierárquicas em texto normativo (parágrafo 8.2.1, art. 5° § 3,
  inciso III). Distingue de IDX (material/norma) e NUM (valor escalar)
  — refere ESTRUTURA legal/normativa, não medida nem identificador.
"""

from enum import StrEnum


class TipoNome(StrEnum):
    GRANDEZA_FISICA = "GrandezaFisica"
    PROSA_TECNICA = "ProsaTecnica"
    IDENTIFICADOR_TECNICO = "IdentificadorTecnico"
    OPERADOR_FORMAL = "OperadorFormal"
    CONSTANTE_UNIVERSAL = "ConstanteUniversal"
    RELACAO_ESTRUTURAL = "RelacaoEstrutural"
    EXPRESSAO_SIMBOLICA = "ExpressaoSimbolica"
    NUMERO = "Numero"  # v1.1 — escalar real sem dimensão
    REFERENCIA = "Referencia"  # v1.2 — referência hierárquica normativa


TIPOS_V1: frozenset[TipoNome] = frozenset({
    TipoNome.GRANDEZA_FISICA,
    TipoNome.PROSA_TECNICA,
    TipoNome.IDENTIFICADOR_TECNICO,
    TipoNome.OPERADOR_FORMAL,
    TipoNome.CONSTANTE_UNIVERSAL,
    TipoNome.RELACAO_ESTRUTURAL,
    TipoNome.EXPRESSAO_SIMBOLICA,
})
"""Conjunto canônico de tipos da OEE v1.0 — 7 tipos. Preservado por
compatibilidade retroativa."""

TIPOS_V1_1: frozenset[TipoNome] = frozenset({
    TipoNome.GRANDEZA_FISICA,
    TipoNome.PROSA_TECNICA,
    TipoNome.IDENTIFICADOR_TECNICO,
    TipoNome.OPERADOR_FORMAL,
    TipoNome.CONSTANTE_UNIVERSAL,
    TipoNome.RELACAO_ESTRUTURAL,
    TipoNome.EXPRESSAO_SIMBOLICA,
    TipoNome.NUMERO,
})
"""Conjunto canônico de tipos da OEE v1.1 — 8 tipos. Adiciona NUMERO
sem remover nenhum dos 7 anteriores (P5 extensibilidade).

Numero é entidade primária do mundo abstrato:
escalar real sem dimensão física associada. Distingue-se de:
- GrandezaFisica (tem dimensão ℤ⁷)
- ConstanteUniversal (tem símbolo nomeado canônico)
- ProsaTecnica (sem estrutura formal)
- ExpressaoSimbolica (composição mediada, não átomo)

Justificativa: empírica (reconstrução numérica — números soltos sem um
tipo formal perdem o valor extraível) + SOTA (Yang ICLR 2025 Number
Cookbook, gap em PT-BR locale-aware tokenization)."""

TIPOS_V1_2: frozenset[TipoNome] = frozenset(TipoNome)
"""Conjunto canônico de tipos da OEE v1.2 — 9 tipos. Adiciona
REFERENCIA sem remover nenhum dos 8 anteriores (P5 extensibilidade).

Referencia é entidade primária do mundo
estrutural-normativo: referência hierárquica a parágrafos, artigos,
incisos, seções de leis/normas/regulamentos. Distingue-se de:
- IdentificadorTecnico (material/norma como AISI 1045, NBR 6118)
- Numero (escalar com valor numérico interpretável)
- ProsaTecnica (sem estrutura formal)

Formato canônico: `[REF:<hierarquia>]` onde hierarquia preserva a
forma original como escrita (8.2.1, 14.2, 5.3.1.2). Sempre precedida
estruturalmente por enumerador normativo (parágrafo, §, art., inciso,
alínea, lei, decreto, norma, regulamento).

Justificativa: empírica (textos legais/normativos em vestibulares
ENEM/BLUEX usam essa forma; sem um tipo próprio, 8.2.1 fragmenta em NUM 8.2
+ NUM 1 ou pior). Estrutura inequivocamente identificável pelo
antecedente — não é heurística."""
