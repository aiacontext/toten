"""Compiler — YAML → JSONL para automação experimental.

Fluxo:

1. Carrega `references.yaml`, `tags_vocabulary.yaml`, `manifest.yaml`.
2. Varre `<discipline>/cases/*.yaml`, deserializa para `Case`.
3. Valida cada caso via `validate_corpus`.
4. Para cada caso `review_status=validated`:
   - Resolve assets (lê SVGs, embute inline se `embed_inline=true`).
   - Aplica `Tokenizer.preprocess_for_llm` ao `prompt_baseline`.
   - Empacota condições A/B/C em estruturas pré-computadas.
5. Escreve `generated/engquant_v<X.Y.Z>.jsonl` (uma linha por caso).
6. Escreve `generated/engquant_v<X.Y.Z>_manifest.json` com checksums.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Prólogo XML opcional (SVG renderiza sem ele). Strip ao embedar para
# evitar que tokens como `encoding="UTF-8"` poluam o pipeline.
_XML_PROLOG_RE = re.compile(r"^\s*<\?xml[^?]*\?>\s*", re.IGNORECASE)

import yaml
from pydantic import ValidationError as PydanticValidationError

from toten import Tokenizer
from toten.benchmark.schema import (
    Case,
    Manifest,
    ReferencesFile,
    TagsVocabulary,
)
from toten.benchmark.validate import (
    ValidationReport,
    validate_corpus,
)


@dataclass
class CompileResult:
    """Resultado de uma compilação completa."""

    jsonl_path: Path
    manifest_path: Path
    n_cases_total: int
    n_cases_compiled: int
    n_cases_rejected: int
    validation_report: ValidationReport
    tags_seen: dict[str, int] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_references(bench_dir: Path) -> ReferencesFile:
    return ReferencesFile.model_validate(_load_yaml(bench_dir / "references.yaml"))


def load_tags_vocabulary(bench_dir: Path) -> TagsVocabulary:
    return TagsVocabulary.model_validate(
        _load_yaml(bench_dir / "tags_vocabulary.yaml")
    )


def load_manifest(bench_dir: Path) -> Manifest:
    return Manifest.model_validate(_load_yaml(bench_dir / "manifest.yaml"))


def discover_case_files(bench_dir: Path) -> list[Path]:
    """Lista todos os YAMLs de casos em <bench_dir>/*/cases/*.yaml."""
    paths: list[Path] = []
    for discipline_dir in sorted(bench_dir.iterdir()):
        if not discipline_dir.is_dir():
            continue
        cases_dir = discipline_dir / "cases"
        if not cases_dir.is_dir():
            continue
        for case_path in sorted(cases_dir.glob("*.yaml")):
            paths.append(case_path)
    return paths


def load_cases(bench_dir: Path) -> tuple[list[Case], list[tuple[Path, str]]]:
    """Carrega todos os casos. Devolve (casos_válidos_estruturalmente, erros)."""
    cases: list[Case] = []
    errors: list[tuple[Path, str]] = []
    for path in discover_case_files(bench_dir):
        try:
            raw = _load_yaml(path)
            cases.append(Case.model_validate(raw))
        except (yaml.YAMLError, PydanticValidationError) as exc:
            errors.append((path, str(exc)))
    return cases, errors


def _embed_assets(case: Case, bench_dir: Path) -> str:
    """Substitui menções de assets no prompt_baseline pelo conteúdo inline.

    Convenção: o `prompt_baseline` deve mencionar cada asset por seu id.
    O conteúdo é injetado no final do prompt em uma seção "Assets:".
    Apenas assets com `embed_inline=true` são embutidos.
    """
    text = case.prompt_baseline
    inline_assets = [a for a in case.assets if a.embed_inline]
    if not inline_assets:
        return text

    # Localizar pasta da disciplina para resolver caminhos relativos
    discipline_dir = bench_dir / case.metadata.discipline_folder
    cases_dir = discipline_dir / "cases"

    blocks: list[str] = []
    for asset in inline_assets:
        # path relativo ao YAML do caso (que vive em cases/)
        asset_path = (cases_dir / asset.path).resolve()
        if not asset_path.is_file():
            msg = f"Asset {asset.id} não encontrado em {asset_path}"
            raise FileNotFoundError(msg)
        content = asset_path.read_text(encoding="utf-8")
        # Strip do prólogo XML (`<?xml ... ?>`): é declaração para parser
        # XML, opcional para SVG, e poluiria o prompt com tokens como
        # `encoding="UTF-8"` que classifiers de entidades técnicas
        # tratariam erroneamente como identificadores normativos.
        content = _XML_PROLOG_RE.sub("", content).lstrip()
        blocks.append(f"\n\n{content}")

    return text + "".join(blocks)


def _serialize_case(
    case: Case,
    tokenizer: Tokenizer,
    bench_dir: Path,
) -> dict[str, Any]:
    """Compila um caso em dict pronto para JSONL."""
    baseline_with_assets = _embed_assets(case, bench_dir)
    activated = tokenizer.preprocess_for_llm(
        baseline_with_assets, lang=case.metadata.language
    )

    # Braço D (controle fatorial 2×2): ativação no system + baseline sem
    # anotações no user. Isola efeito puro do prompt de ativação,
    # complementando o controle B (anotações sem ativação).
    activation_system = activated.activation.rstrip()
    modo_d_ready = f"{activation_system}\n\n{baseline_with_assets}"
    modo_d_messages = [
        {"role": "system", "content": activation_system},
        {"role": "user", "content": baseline_with_assets},
    ]

    return {
        "id": case.id,
        "title": case.title,
        "metadata": case.metadata.model_dump(),
        "reference": case.reference.model_dump(),
        "tags": case.tags,
        "expected_signals": case.expected_signals,
        "memoria_calculo": case.memoria_calculo or {},
        "expected_norms": [en.model_dump() for en in case.expected_norms],
        "gabarito": [g.model_dump() for g in case.gabarito],
        "prompts": {
            "baseline": baseline_with_assets,
            "modo_b": activated.preprocessed,
            "modo_b_activated": activated.ready_for_llm,
            "modo_d_activated": modo_d_ready,
        },
        "messages_for_api": {
            "modo_b_activated": activated.as_messages(),
            "modo_d_activated": modo_d_messages,
        },
    }


def _checksum_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compile_benchmark(
    bench_dir: Path,
    output_dir: Path | None = None,
    require_validated: bool = True,
) -> CompileResult:
    """Compila o corpus inteiro de `bench_dir` em JSONL.

    Parameters
    ----------
    bench_dir:
        Raiz do Bench_EngQuant (contém references.yaml, manifest.yaml, etc.).
    output_dir:
        Diretório onde escrever generated/. Default: `bench_dir / "generated"`.
    require_validated:
        Se True (default), apenas casos com review_status='validated' entram
        no JSONL. Se False, todos os casos válidos estruturalmente entram
        (útil para inspeção).
    """
    bench_dir = bench_dir.resolve()
    output_dir = (output_dir or (bench_dir / "generated")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(bench_dir)
    references = load_references(bench_dir)
    vocab = load_tags_vocabulary(bench_dir)

    cases, structural_errors = load_cases(bench_dir)
    if structural_errors:
        details = "\n".join(f"  {path}: {err}" for path, err in structural_errors)
        msg = f"Falha na carga estrutural de casos:\n{details}"
        raise ValueError(msg)

    validation = validate_corpus(cases, vocab, references)

    tokenizer = Tokenizer.from_ontology("oee-v1")

    jsonl_path = (
        output_dir / f"engquant_v{manifest.benchmark_version}.jsonl"
    )
    manifest_path = (
        output_dir / f"engquant_v{manifest.benchmark_version}_manifest.json"
    )

    tags_seen: dict[str, int] = {}
    n_compiled = 0
    n_rejected = 0
    lines: list[str] = []

    for case in cases:
        result = next(r for r in validation.results if r.case_id == case.id)
        if not result.is_valid:
            n_rejected += 1
            continue
        if require_validated and case.review_status != "validated":
            n_rejected += 1
            continue
        serialized = _serialize_case(case, tokenizer, bench_dir)
        lines.append(json.dumps(serialized, ensure_ascii=False))
        n_compiled += 1
        for tag in case.tags:
            tags_seen[tag] = tags_seen.get(tag, 0) + 1

    jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    jsonl_path.write_bytes(jsonl_bytes)

    manifest_out = {
        "benchmark_version": manifest.benchmark_version,
        "name": manifest.name,
        "description": manifest.description,
        "primary_language": manifest.primary_language,
        "n_cases_total": len(cases),
        "n_cases_compiled": n_compiled,
        "n_cases_rejected": n_rejected,
        "tags_stratification": dict(sorted(tags_seen.items())),
        "jsonl_file": jsonl_path.name,
        "jsonl_sha256": _checksum_bytes(jsonl_bytes),
        "schema_version": manifest.schema_version,
    }
    manifest_path.write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return CompileResult(
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        n_cases_total=len(cases),
        n_cases_compiled=n_compiled,
        n_cases_rejected=n_rejected,
        validation_report=validation,
        tags_seen=tags_seen,
    )
