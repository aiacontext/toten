"""Camada 2 da arquitetura — Classificador Ontológico.

Dado texto bruto, identifica regiões correspondentes a cada tipo
ontológico declarado na OEE e emite `ProsaTecnica` em gaps com
conteúdo. Cobertura v0 (Dia 3):

- GrandezaFisica — PT-BR/EN, com/sem incerteza, notação científica
- ConstanteUniversal — multi-char sempre; single-char em contexto
  math-like (`= e`, `c ≈`)
- IdentificadorTecnico — normas (ABNT NBR / ISO / ASTM / ASME / AWS /
  EN), materiais (AISI, SAE, ER, Al, Ti-6Al-4V), grandezas adimensionais
- OperadorFormal — Unicode (≤ ≥ ≠ ≈ ≡ ∝ ∑ ∫ ∂ ∇ ∏ √ ∆ × ÷ ∧ ∨ ¬ ∀ ∃
  ∈ ∉ ⊂ ⊃ ⊆ ⊇ → ⇒ ⇐ ↔ ⇔); `=` ASCII bounded
- RelacaoEstrutural — conectores PT-BR e EN do lexicon
- ProsaTecnica — residual emitido em gaps não-whitespace

Política de resolução de sobreposição: ordem declarada em
`data/oee-v1.yaml.resolucao_ambiguidade.ordem` (maior especificidade
prevalece). Empate favorece span mais longo.
"""

from toten.classifier.classifier import (
    DEFAULT_CONSTANT_LEXICON_PATH,
    DEFAULT_IDENTIFIER_LEXICON_PATH,
    DEFAULT_OPERATOR_LEXICON_PATH,
    DEFAULT_RELATION_LEXICON_PATH,
    OntologicalClassifier,
    default_units,
)
from toten.classifier.region import Region

__all__ = [
    "Region",
    "OntologicalClassifier",
    "default_units",
    "DEFAULT_IDENTIFIER_LEXICON_PATH",
    "DEFAULT_OPERATOR_LEXICON_PATH",
    "DEFAULT_CONSTANT_LEXICON_PATH",
    "DEFAULT_RELATION_LEXICON_PATH",
]
