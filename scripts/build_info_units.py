"""Auto-gera `data/info_units.json` a partir de Pint.

**Domínio ontológico:** unidades de informação e visualização — bit,
byte, octet, prefixos IEC binários (KiB, MiB, GiB) e SI decimais (KB,
MB, GB), bits-per-second variants, pixel/dot, dpi/ppi/ppcm, bpp.

**Nota técnica sobre Pint default:** o registry padrão de Pint trata
bit/byte como `dimensionless` (factor de razão), não com dimensão
própria `[information]`. Pixel está em `[printing_unit]`. Como o
critério dimensional é misto, usamos WHITELIST EXPLÍCITA por nome
canônico (consistente com `build_dimensionless_units.py`).

**Princípio responsabilidade única:**
- `dim_table.json` → SI ℤ⁷ (massa, comprimento, …)
- `dimensionless_units.json` → razões/ângulos/log puras
- `info_units.json` → informação/visualização (este arquivo)
- `constant_lexicon_v0.json` → constantes nomeadas

Embora Pint default considere bit/byte como dimensionless, nosso
framework os SEPARA como categoria semântica própria — engenharia
de software/redes/visualização tem vocabulário técnico distinto do
domínio físico SI puro.

Uso:
    uv run python -m scripts.build_info_units [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pint


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data/info_units.json"


# Whitelist explícita por categoria semântica. Pint default fornece
# todos estes (a maioria como `dimensionless` com factor de razão).
WHITELIST: dict[str, str] = {
    # === INFORMATION (átomos base) ===
    "bit": "information",
    "byte": "information",
    "octet": "information",
    "nibble": "information",
    # === INFORMATION (prefixos SI decimais — base 10³) ===
    # KB não existe em Pint default; MB/GB sim. Adicionamos via alias.
    "kilobit": "information",
    "megabit": "information",
    "gigabit": "information",
    "terabit": "information",
    "kbit": "information",
    "Mbit": "information",
    "Gbit": "information",
    "Tbit": "information",
    "kilobyte": "information",
    "megabyte": "information",
    "gigabyte": "information",
    "terabyte": "information",
    "petabyte": "information",
    "kB": "information",
    "MB": "information",
    "GB": "information",
    "TB": "information",
    "PB": "information",
    # === INFORMATION (prefixos IEC binários — base 2¹⁰) ===
    "kibibyte": "information",
    "mebibyte": "information",
    "gibibyte": "information",
    "tebibyte": "information",
    "pebibyte": "information",
    "KiB": "information",
    "MiB": "information",
    "GiB": "information",
    "TiB": "information",
    "PiB": "information",
    "kibibit": "information",
    "mebibit": "information",
    "gibibit": "information",
    "Kibit": "information",
    "Mibit": "information",
    "Gibit": "information",
    # === VISUALIZATION (átomos) ===
    "pixel": "visualization",
    "dot": "visualization",
    # === VISUAL_DENSITY (densidade pixel/dot por comprimento) ===
    "pixels_per_inch": "visual_density",
    "pixels_per_centimeter": "visual_density",
    "dots_per_inch": "visual_density",
    "ppi": "visual_density",
    "PPI": "visual_density",
    "ppcm": "visual_density",
    "PPCM": "visual_density",
    "DPI": "visual_density",
    "printers_dpi": "visual_density",
    # NOTA: `dpi` (lowercase isolado) é DRY_PINT (volume) em Pint
    # default — falso amigo. Usuários devem usar `DPI` ou `ppi`.
    # === COLOR_DEPTH ===
    "bits_per_pixel": "color_depth",
    "bpp": "color_depth",
}


def build_table(verbose: bool = False) -> tuple[dict, dict]:
    """Constrói `info_units.json` consultando Pint para factor de cada
    entrada do whitelist."""
    ureg = pint.UnitRegistry()

    atoms: dict[str, dict] = {}
    stats = {"included": 0, "pint_undefined": 0, "invalid_factor": 0}
    missing = []

    for name, category in WHITELIST.items():
        try:
            q = ureg.parse_expression(name)
            factor = float(q.to_base_units().magnitude)
        except Exception:
            stats["pint_undefined"] += 1
            missing.append(name)
            continue

        if factor <= 0 or not (factor == factor):
            stats["invalid_factor"] += 1
            continue

        atoms[name] = {
            "factor": factor,
            "category": category,
            "pint_base_dim": str(q.dimensionality),
        }
        stats["included"] += 1
        if verbose:
            print(f"  [INCLUDED] {name:18s} {category:14s} factor={factor:.6g}  dim={q.dimensionality}")

    if missing:
        print(f"[warn] {len(missing)} entradas do whitelist não em Pint: {missing}")

    output = {
        "version": "0.1.0",
        "comment": (
            "Tabela ATÔMICA de unidades de INFORMAÇÃO e VISUALIZAÇÃO. "
            "AUTO-GERADA por scripts/build_info_units.py — não editar "
            "à mão. Domínios cobertos: information (bit, byte, KB, MB, "
            "GB, KiB, MiB...), visualization (pixel, dot), "
            "visual_density (dpi, ppi, ppcm), color_depth (bpp). "
            "Pint default trata bit/byte como dimensionless (factor), "
            "mas nosso framework os SEPARA semanticamente (vocabulário "
            "técnico de engenharia de software/redes/visualização). "
            "FLOPS/MFLOPS são padrões não-canônicos; se necessários, "
            "adicionar como extensão manual com `non_pint_extension: "
            "true`. Princípio responsabilidade única: separado de "
            "dim_table.json (SI ℤ⁷), dimensionless_units.json (razões), "
            "constant_lexicon (constantes nomeadas)."
        ),
        "atoms": dict(sorted(atoms.items())),
    }
    return output, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"[build] gerando info_units.json")
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
