"""EngQuant benchmark — schema, validator, compiler.

Estrutura externa em `Bench_EngQuant/`:
- references.yaml — bibliografia central (Bastos, Nawy, NBRs, ...)
- tags_vocabulary.yaml — vocabulário emergente de tags
- manifest.yaml — versão e metadados do benchmark
- <discipline>/cases/<id>.yaml — casos individuais
- <discipline>/_assets/ — SVGs, PDFs-fonte, tabelas
- generated/ — JSONL compilado para automação (gitignored)

Princípio organizador: disciplinas são pastas filesystem, não estratos
de análise. Análise estatística estratifica por tags emergentes que
cruzam disciplinas.

Critério de inclusão (gate automatizado):
- ≥4 sub-grandezas auditáveis (multi-step real)
- ≥1 tag de high_risk_tags do vocabulário (foco da tese)
- reference resolvível em references.yaml
- review_status=validated
"""

from toten.benchmark.schema import (
    Asset,
    BenchmarkBundle,
    Case,
    GabaritoEntry,
    Manifest,
    Reference,
    ReferencesFile,
    TagsVocabulary,
)
from toten.benchmark.validate import (
    ValidationError,
    ValidationReport,
    validate_case,
    validate_corpus,
)

__all__ = [
    "Asset",
    "BenchmarkBundle",
    "Case",
    "GabaritoEntry",
    "Manifest",
    "Reference",
    "ReferencesFile",
    "TagsVocabulary",
    "ValidationError",
    "ValidationReport",
    "validate_case",
    "validate_corpus",
]
