"""Validador — gate de inclusão de casos no benchmark.

Critérios de inclusão (todos obrigatórios para um caso entrar no
JSONL compilado):

1. Schema válido (pydantic já garante na carga).
2. ≥4 sub-grandezas no gabarito (multi-step real).
3. ≥1 tag de high_risk_tags (foco na tese).
4. Reference resolvível em references.yaml.
5. Todas as tags em tags_vocabulary.
6. review_status='validated'.

Erros de validação são acumulados em `ValidationReport`, não levantam
imediatamente — permite revisar lote inteiro antes de corrigir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from toten.benchmark.schema import (
    Case,
    ReferencesFile,
    TagsVocabulary,
)

MIN_SUBGRANDEZAS = 4


class ValidationError(Exception):
    """Levantada por validate_case_strict() quando algum critério falha."""


@dataclass
class CaseValidationResult:
    """Resultado de validação de um único caso."""

    case_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Relatório agregado de validação de um corpus."""

    results: list[CaseValidationResult] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return sum(1 for r in self.results if r.is_valid)

    @property
    def n_invalid(self) -> int:
        return sum(1 for r in self.results if not r.is_valid)

    @property
    def has_errors(self) -> bool:
        return self.n_invalid > 0

    def add(self, result: CaseValidationResult) -> None:
        self.results.append(result)

    def summary(self) -> str:
        lines = [
            f"Total: {len(self.results)} casos",
            f"  Válidos: {self.n_valid}",
            f"  Inválidos: {self.n_invalid}",
        ]
        if self.has_errors:
            lines.append("")
            lines.append("Casos com erro:")
            for r in self.results:
                if not r.is_valid:
                    lines.append(f"  [{r.case_id}]")
                    for err in r.errors:
                        lines.append(f"    ERR: {err}")
                    for warn in r.warnings:
                        lines.append(f"    WARN: {warn}")
        return "\n".join(lines)


def validate_case(
    case: Case,
    vocab: TagsVocabulary,
    references: ReferencesFile,
) -> CaseValidationResult:
    """Avalia um caso contra os critérios de inclusão.

    Returns
    -------
    CaseValidationResult com is_valid + lista de erros/warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Multi-step ≥ 4 sub-grandezas
    if len(case.gabarito) < MIN_SUBGRANDEZAS:
        errors.append(
            f"len(gabarito)={len(case.gabarito)} < {MIN_SUBGRANDEZAS} "
            "(multi-step real exige ≥4 sub-grandezas)"
        )

    # 2. ≥1 high_risk_tag
    case_tags = set(case.tags)
    high_risk_intersect = case_tags & vocab.all_high_risk_tags
    if not high_risk_intersect:
        errors.append(
            "caso não tem nenhuma tag de high_risk_tags — fora do foco da tese. "
            f"Tags do caso: {sorted(case_tags)}. "
            f"high_risk_tags conhecidas: {sorted(vocab.all_high_risk_tags)}"
        )

    # 3. Reference resolvível
    if references.get(case.reference.source_id) is None:
        errors.append(
            f"reference.source_id='{case.reference.source_id}' não encontrado em "
            "references.yaml"
        )

    # 4. Todas as tags conhecidas
    unknown_tags = case_tags - vocab.all_known_tags
    if unknown_tags:
        errors.append(
            f"tags desconhecidas em tags_vocabulary.yaml: {sorted(unknown_tags)}. "
            "Adicione ao vocabulário antes de usar."
        )

    # 5. Expected_norms referenciam sources válidos
    for en in case.expected_norms:
        if references.get(en.source_id) is None:
            errors.append(
                f"expected_norms.source_id='{en.source_id}' não encontrado em "
                "references.yaml"
            )

    # 6. review_status
    if case.review_status != "validated":
        warnings.append(
            f"review_status='{case.review_status}' — entrará no JSONL apenas se "
            "'validated'. Promova quando o caso for revisado."
        )

    # 7. Coerência interna: se has_figure, deve haver pelo menos um asset svg/png
    if case.metadata.has_figure:
        figure_assets = [a for a in case.assets if a.type in ("svg", "png")]
        if not figure_assets:
            warnings.append(
                "metadata.has_figure=true mas nenhum asset do tipo svg/png declarado"
            )

    return CaseValidationResult(
        case_id=case.id,
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def validate_corpus(
    cases: list[Case],
    vocab: TagsVocabulary,
    references: ReferencesFile,
) -> ValidationReport:
    """Valida um lote de casos, devolvendo relatório agregado."""
    report = ValidationReport()
    seen_ids: set[str] = set()
    for case in cases:
        result = validate_case(case, vocab, references)
        if case.id in seen_ids:
            result.errors.append(f"case.id '{case.id}' duplicado no corpus")
            result.is_valid = False
        seen_ids.add(case.id)
        report.add(result)
    return report


def validate_case_strict(
    case: Case,
    vocab: TagsVocabulary,
    references: ReferencesFile,
) -> None:
    """Validação que levanta `ValidationError` se algo falhar."""
    result = validate_case(case, vocab, references)
    if not result.is_valid:
        msg = f"Caso {case.id} inválido:\n  " + "\n  ".join(result.errors)
        raise ValidationError(msg)
