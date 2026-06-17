"""Auto-gera `data/dimensionless_units.json` a partir de Pint.

**Domínio ontológico:** unidades adimensionais úteis — razões puras
(`%`, `‰`, `ppm`, `ppb`), logarítmicas (`dB`, `Np`), ângulos planos
(`rad`, `°`, `arcmin`, `arcsec`) e ângulos sólidos (`sr`). Sem
dimensão SI (`dim_vec = (0,…,0)`), mas com FATOR DE RAZÃO útil para
conversão (e.g., `1% = 0.01`, `1° = π/180 rad`, `1 dB = 10^(1/10)`).

**Princípio responsabilidade única:**
- `dim_table.json` → unidades SI dimensionais (massa, comprimento, …)
- `dimensionless_units.json` → adimensionais (este arquivo)
- `info_units.json` → informação/visualização (bit, byte, pixel)
- `constant_lexicon_v0.json` → constantes nomeadas (π, k_B)

**Política de inclusão — WHITELIST EXPLÍCITA por nome canônico:**

Pint expõe ~66 adimensionais. A maioria são palavras EN comuns
(`grade`, `byte`, `cycle`, `octave`, `count`, `turn`, `mil`, `decade`)
ou letras gregas isoladas (`α`, `ε`, `ζ`) que causariam falsos
positivos em prosa PT-BR técnica. A inclusão é POR ESCOLHA EXPLÍCITA
de unidades canônicas com nome técnico inequívoco.

Categorias incluídas:
- `ratio`: %, ‰, ppm, ppb, ppt (e nomes longos `percent`/`permille`)
- `logarithmic_ratio`: dB, decibel, Np, neper
- `angle`: rad, radian, °, deg, degree, arcmin, arcsec, gon, gradian
- `solid_angle`: sr, steradian, sq_deg, square_degree

Uso:
    uv run python -m scripts.build_dimensionless_units [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pint


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data/dimensionless_units.json"


# Whitelist explícita por categoria — cada inclusão tem justificativa
# técnica formal. Adicionar nova entrada exige declarar categoria E
# justificativa em comentário.
WHITELIST: dict[str, str] = {
    # === RATIO ===
    "%": "ratio",
    "‰": "ratio",
    "percent": "ratio",
    "permille": "ratio",
    "ppm": "ratio",          # parts per million (10⁻⁶)
    "ppb": "ratio",          # parts per billion (10⁻⁹)
    "ppt": "ratio",          # parts per trillion (10⁻¹²)
    # === LOGARITHMIC_RATIO ===
    "dB": "logarithmic_ratio",
    "decibel": "logarithmic_ratio",
    "Np": "logarithmic_ratio",  # neper
    "neper": "logarithmic_ratio",
    # === ANGLE (plano) ===
    "rad": "angle",
    "radian": "angle",
    "°": "angle",
    "deg": "angle",
    "degree": "angle",
    "grau": "angle",         # PT-BR
    "arcmin": "angle",
    "arcminute": "angle",
    "arc_minute": "angle",
    "arcsec": "angle",
    "arcsecond": "angle",
    "arc_second": "angle",
    "angular_degree": "angle",
    "angular_minute": "angle",
    "angular_second": "angle",
    "gon": "angle",
    "gradian": "angle",
    # === SOLID ANGLE ===
    "sr": "solid_angle",
    "steradian": "solid_angle",
    "sq_deg": "solid_angle",
    "sqdeg": "solid_angle",
    "square_degree": "solid_angle",
}


def build_table(verbose: bool = False) -> tuple[dict, dict]:
    """Constrói `dimensionless_units.json` consultando Pint para factor
    de razão de cada entrada do whitelist."""
    ureg = pint.UnitRegistry()

    atoms: dict[str, dict] = {}
    stats = {"included": 0, "pint_undefined": 0, "skipped": 0}
    missing = []

    for name, category in WHITELIST.items():
        try:
            q = ureg.parse_expression(name)
            factor = float(q.to_base_units().magnitude)
        except Exception:
            stats["pint_undefined"] += 1
            missing.append(name)
            continue

        if factor <= 0:
            stats["skipped"] += 1
            continue

        atoms[name] = {
            "factor": factor,
            "category": category,
        }
        stats["included"] += 1
        if verbose:
            print(f"  [INCLUDED] {name:15s} category={category:18s} factor={factor:.6g}")

    if missing:
        print(f"[warn] {len(missing)} entradas do whitelist não encontradas no Pint: {missing}")

    output = {
        "version": "0.1.0",
        "comment": (
            "Tabela ATÔMICA de unidades ADIMENSIONAIS úteis (razões, "
            "logarítmicas, ângulos planos, ângulos sólidos). "
            "AUTO-GERADA por scripts/build_dimensionless_units.py — não "
            "editar à mão. Whitelist explícita por nome canônico; "
            "Pint fornece factor de razão para conversão. Princípio "
            "ontológico (responsabilidade única): unidades SEM dimensão "
            "SI (`dim_vec = 0`) vivem aqui, separadas de dim_table.json "
            "(SI físico ℤ⁷), info_units.json (informação/visualização) "
            "e constant_lexicon (constantes nomeadas com valor)."
        ),
        "atoms": dict(sorted(atoms.items())),
    }
    return output, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"[build] gerando dimensionless_units.json")
    output, stats = build_table(verbose=args.verbose)

    print(f"\n[stats]")
    for k, v in sorted(stats.items()):
        print(f"  {k:25s} {v:5d}")

    print(f"\n[result] total átomos: {len(output['atoms'])}")

    if args.dry_run:
        print(f"\n[dry-run] não escrevendo arquivo")
        return 0

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[written] {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
