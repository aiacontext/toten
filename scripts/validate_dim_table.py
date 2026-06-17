"""Sanity-check da dim_table contra Pint.

Build-time / dev script. Pint NÃO é runtime dependency (spec §4).
Aqui ele é apenas oráculo independente para auditar a tabela.

Uso:
    uv run python scripts/validate_dim_table.py
    uv run python scripts/validate_dim_table.py --tolerancia 1e-9

Para cada átomo de `data/dim_table.json`:
  1. Verifica se Pint reconhece o símbolo (alguns como `°C`, `tonelada`,
     `cv` exigem alias custom — esses são pulados com nota).
  2. Compara `factor` SI nosso vs Pint (tolerância relativa).
  3. Compara o `dim_vector` (kg, m, s, A, K, mol, cd) com a assinatura
     dimensional do Pint.

Reporta:
  - OK   : átomo bate dentro de tolerância
  - SKIP : Pint não reconhece (esperado para BR/aliases)
  - DIFF : discrepância numérica — exige revisão

Exit code: 0 se nenhum DIFF; 1 caso contrário.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pint

ROOT = Path(__file__).resolve().parent.parent
DIM_TABLE_PATH = ROOT / "data" / "dim_table.json"

# Aliases: símbolos nossos → símbolos Pint
PINT_ALIASES: dict[str, str] = {
    "µg": "microgram",
    "μg": "microgram",
    "µm": "micrometer",
    "μm": "micrometer",
    "µs": "microsecond",
    "μs": "microsecond",
    "µA": "microampere",
    "μA": "microampere",
    "µF": "microfarad",
    "μF": "microfarad",
    "µH": "microhenry",
    "μH": "microhenry",
    "µT": "microtesla",
    "μT": "microtesla",
    "Ω": "ohm",
    "kΩ": "kiloohm",
    "MΩ": "megaohm",
    "mΩ": "milliohm",
    "°C": "kelvin",  # tratado como delta-T
    "°F": "rankine",
    "°R": "rankine",
    "°": "degree",
    "tonelada": "metric_ton",
    "ton": "metric_ton",
    "t": "metric_ton",
    "tf": "tonne_force",
    "cv": "metric_horsepower",
    "HP": "horsepower",
    "hp": "horsepower",
    "dia": "day",
    "ano": "julian_year",
    "grau": "degree",
    "kVAR": "kilovolt_ampere",
    "VAR": "volt_ampere",
}

# Símbolos para os quais Pint diverge legitimamente ou não tem entrada
# análoga (e.g., °F como delta — Pint trata absoluto; rpm em Pint é
# velocidade angular rad/s, nossa interpretação é frequência 1/min).
PINT_SKIP: set[str] = {"°F", "°R", "rpm"}

# Ordem canônica do dim vector: [kg, m, s, A, K, mol, cd]
SI_BASE_PINT = ["kilogram", "meter", "second", "ampere", "kelvin", "mole", "candela"]


def pint_dim_vector(quantity: pint.Quantity) -> tuple[int, ...] | None:
    """Devolve o vetor (kg, m, s, A, K, mol, cd) do Pint Quantity."""
    try:
        base = quantity.to_base_units()
    except Exception:
        return None
    dim = base.dimensionality  # {'[mass]': 1, '[length]': -1, ...}
    mapping = {
        "[mass]": 0, "[length]": 1, "[time]": 2,
        "[current]": 3, "[temperature]": 4, "[substance]": 5,
        "[luminosity]": 6,
    }
    vec = [0] * 7
    for key, power in dim.items():
        idx = mapping.get(str(key))
        if idx is None:
            return None
        vec[idx] = int(power)
    return tuple(vec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerancia",
        type=float,
        default=1e-6,
        help="Tolerância relativa para comparação de factor (default 1e-6).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Imprime cada átomo."
    )
    args = parser.parse_args()

    ureg = pint.UnitRegistry()
    with DIM_TABLE_PATH.open("r", encoding="utf-8") as fp:
        table = json.load(fp)

    ok = skip = diff = 0
    diffs: list[str] = []

    for symbol, entry in table["atoms"].items():
        pint_name = PINT_ALIASES.get(symbol, symbol)
        if symbol in PINT_SKIP:
            skip += 1
            if args.verbose:
                print(f"SKIP {symbol:15s} (não reconcilia com Pint)")
            continue
        try:
            q = ureg.Quantity(1, pint_name)
        except Exception as exc:
            skip += 1
            if args.verbose:
                print(f"SKIP {symbol:15s} (Pint não reconhece '{pint_name}': {exc})")
            continue

        # Factor
        try:
            pint_factor = q.to_base_units().magnitude
        except Exception as exc:
            skip += 1
            if args.verbose:
                print(f"SKIP {symbol:15s} (to_base_units falhou: {exc})")
            continue

        our_factor = entry["factor"]
        rel_err = abs(pint_factor - our_factor) / max(abs(pint_factor), 1e-30)

        # Dimensão
        pint_dim = pint_dim_vector(q)
        our_dim = tuple(entry["dim"])
        dim_match = pint_dim == our_dim

        if rel_err > args.tolerancia or not dim_match:
            diff += 1
            msg = (
                f"DIFF {symbol:15s} "
                f"factor: nosso={our_factor:.6g} pint={pint_factor:.6g} "
                f"rel_err={rel_err:.2e}  "
                f"dim: nosso={our_dim} pint={pint_dim}"
            )
            diffs.append(msg)
            print(msg)
        else:
            ok += 1
            if args.verbose:
                print(f"OK   {symbol:15s} factor={our_factor:.6g} dim={our_dim}")

    total = len(table["atoms"])
    print()
    print(f"Total: {total} átomos. OK: {ok}, SKIP: {skip}, DIFF: {diff}.")
    if diff > 0:
        print()
        print("Discrepâncias requerem revisão:")
        for d in diffs:
            print(f"  {d}")
        return 1
    print("Tabela dimensional consistente com Pint dentro da tolerância.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
