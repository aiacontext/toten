"""Auto-gera `data/dim_table.json` a partir de Pint como oráculo externo.

**Princípio arquitetural (memória `[Oráculo externo > enumeração]`):**
Pint é autoridade externa canônica para unidades de medida; nosso
`dim_table.json` é APENAS materialização derivada (estática para
performance/determinismo no carregamento e para versionamento
explícito do oráculo).

**Por que materializar em JSON em vez de chamar Pint em runtime:**
- Performance: build-once vs reparse-em-toda-chamada
- Determinismo: versão de Pint conhecida e congelada no JSON
- Auditabilidade: diff visível quando Pint atualiza
- Cobertura prevista: lista de átomos vista pelo classifier é fixa
- Reduz superfície de bugs em runtime (sem dependência circular Pint)

**Política de inclusão (não-enumeração-arbitrária):**

1. Itera `ureg._units`; para cada entrada testa `parse_expression(name)`
2. Mantém apenas se dimensionalidade Pint mapeia para nosso ℤ⁷ SI
   `[kg, m, s, A, K, mol, cd]` (exclui radian/bit/count/pixel/etc.)
3. Exclui CONSTANTES FÍSICAS (Planck, Avogadro, etc.) — pertencem ao
   módulo de constantes (`data/constant_lexicon_v0.json`), não ao
   oráculo dimensional
4. Aplica BLACKLIST de símbolos ambíguos com prosa PT-BR:
   - `a` (Pint: ano; PT-BR: artigo definido feminino)
   - `c` (Pint: speed of light constant alias; ambíguo)
   - outras conforme observação empírica

5. Preserva campos custom existentes via merge com versão atual:
   - `prefixable` (manual para SI bases)
   - Unidades BR exclusivas (kgf, tf, mt) — Pint tem `kgf` mas faltam
     algumas variantes históricas brasileiras

**Saída:**
JSON determinístico (sorted, indent=2) substitui `data/dim_table.json`.

Uso:
    uv run python -m scripts.build_dim_table [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pint


REPO_ROOT = Path(__file__).resolve().parents[1]
DIM_TABLE_PATH = REPO_ROOT / "data/dim_table.json"

# Ordem ℤ⁷ canônica do framework (consistente com `dim_table.json`)
SI_BASE_ORDER = ["kg", "m", "s", "A", "K", "mol", "cd"]

# Mapeia nome Pint base-dim para índice ℤ⁷
PINT_DIM_TO_SI = {
    "[mass]": 0,
    "[length]": 1,
    "[time]": 2,
    "[current]": 3,
    "[temperature]": 4,
    "[substance]": 5,
    "[luminosity]": 6,
}

# Dimensões Pint NÃO mapeáveis para SI ℤ⁷ tradicional — átomos com
# essas dimensões são EXCLUÍDOS.
PINT_DIM_EXCLUDED = {
    "[angle]",  # radian, sr — adimensionais por construção SI
    "[information]",  # bit, byte
    "[printing_unit]",  # pixel
    "[refractive_index]",
    "[absorbance]",
    "dimensionless",  # ratio puro
}

# BLACKLIST de símbolos ambíguos com prosa PT-BR. Decisão por
# OBSERVAÇÃO EMPÍRICA, não enumeração arbitrária — cada item precisa
# de justificativa formal de conflito com prosa natural PT-BR.
SYMBOL_BLACKLIST = {
    "a",  # Pint: year alias; PT-BR: artigo definido feminino ("a viga")
    "c",  # Pint: alias para speed_of_light (constante mascarada como
          # unidade); o pattern is_likely_constant não bate em alias
          # de 1 char, então precisa blacklist explícita
    "d",  # Pint: day alias; PT-BR: artigo informal ("d'água") e
          # preposição contraída; temos "dia" literal no JSON custom
    # Constantes MATEMÁTICAS expostas como "unidades" adimensionais
    # por Pint — pertencem ao módulo de constantes universais
    # (data/constant_lexicon_v0.json), não ao oráculo dimensional.
    "π", "pi", "e",  # pi e número de Euler
    # FALSO AMIGO: `dpi` em Pint default é `dry_pint` (volume 0.55L),
    # NÃO "dots per inch". Uso comum deve resolver via `DPI`, `ppi`,
    # `dots_per_inch` no info_units.json. Bloquear `dpi` evita
    # "300 dpi" virar [QTY 300 unit=dpi dim=volume].
    "dpi",
}
# Nota explícita: `h` (hora), `u` (uma), `P` (poise), `S` (siemens)
# são unidades LEGÍTIMAS e PERMITIDAS. Sua eventual ambiguidade com
# prosa é tratada pelo classifier estrutural (precisa de número antes
# para virar QTY-com-value; só vira unit-only com marcador estrutural).

# OVERRIDES SEMÂNTICOS: para entradas onde o JSON manual difere de Pint
# por razão semântica declarada, preserva valor do JSON em vez de
# sobrescrever com Pint.
#
# Casos:
# - `°C`, `°F`, `°R`: Pint usa temperatura ABSOLUTA com offset (273.15K
#   para °C); nosso framework usa temperatura_delta (1°C = 1K em
#   variação, sem offset). Decisão semântica do framework — texto
#   "ΔT = 5°C" requer interpretação delta, não absoluta.
SEMANTIC_OVERRIDES = frozenset(["°C", "°F", "°R"])

# Princípio da responsabilidade única (decisão arquitetural 2026-05-31):
# `dim_table.json` contém APENAS unidades com dimensão SI ℤ⁷ não-nula
# (dim_vec ≠ (0,…,0)). Adimensionais úteis (razões, decibel, ângulos)
# vão para `dimensionless_units.json`. Unidades de informação (bit,
# byte, KB, …) — com dimensão própria de informação — vão para
# `info_units.json`. Cada arquivo tem propósito ontológico único.
#
# Aqui (`build_dim_table.py`): rejeita todo átomo cujo dim_vec é
# (0,…,0) — isso filtra os ~66 adimensionais de Pint sem necessidade
# de whitelist ad-hoc.


# CATEGORIAS — derivadas heuristicamente da dim_vector ℤ⁷
DIM_TO_CATEGORY = {
    (1, 0, 0, 0, 0, 0, 0): "mass",
    (0, 1, 0, 0, 0, 0, 0): "length",
    (0, 0, 1, 0, 0, 0, 0): "time",
    (0, 0, 0, 1, 0, 0, 0): "current",
    (0, 0, 0, 0, 1, 0, 0): "temperature",
    (0, 0, 0, 0, 0, 1, 0): "amount",
    (0, 0, 0, 0, 0, 0, 1): "luminosity",
    (1, 1, -2, 0, 0, 0, 0): "force",
    (1, -1, -2, 0, 0, 0, 0): "pressure",
    (1, 2, -2, 0, 0, 0, 0): "energy",
    (1, 2, -3, 0, 0, 0, 0): "power",
    (0, 0, -1, 0, 0, 0, 0): "frequency",
    (0, 0, 1, 1, 0, 0, 0): "charge",
    (1, 2, -3, -1, 0, 0, 0): "voltage",
    (1, 2, -3, -2, 0, 0, 0): "resistance",
    (-1, -2, 4, 2, 0, 0, 0): "capacitance",
    (1, 2, -2, -2, 0, 0, 0): "inductance",
    (1, 0, -2, -1, 0, 0, 0): "magnetic_flux_density",
    (1, 2, -2, -1, 0, 0, 0): "magnetic_flux",
    (0, 2, 0, 0, 0, 0, 0): "area",
    (0, 3, 0, 0, 0, 0, 0): "volume",
    (0, 1, -1, 0, 0, 0, 0): "velocity",
    (0, 1, -2, 0, 0, 0, 0): "acceleration",
    (1, -3, 0, 0, 0, 0, 0): "density",
    (1, -1, -1, 0, 0, 0, 0): "viscosity",
    (0, 2, -1, 0, 0, 0, 0): "kinematic_viscosity",
}


def pint_dim_to_si_vector(dim: pint.util.UnitsContainer) -> tuple[int, ...] | None:
    """Converte dimensionalidade Pint em vetor ℤ⁷ SI.

    Retorna None se a dimensão usa eixos não-SI (radian, bit, pixel).
    """
    vec = [0] * 7
    for pint_dim, exp in dim.items():
        if pint_dim in PINT_DIM_EXCLUDED:
            return None
        if pint_dim not in PINT_DIM_TO_SI:
            return None  # dimensão desconhecida — exclui por segurança
        idx = PINT_DIM_TO_SI[pint_dim]
        vec[idx] = int(exp)
    return tuple(vec)


def is_likely_constant(name: str, ureg: pint.UnitRegistry) -> bool:
    """Heurística simples: nomes de constantes físicas têm marcas claras."""
    constant_markers = (
        "_constant",
        "_charge",
        "_mass",
        "_radius",
        "_quantum",
        "permittivity",
        "permeability",
        "gravitational",
        "boltzmann",
        "planck",
        "avogadro",
        "faraday",
        "rydberg",
        "loschmidt",
        "stefan",
        "wien",
        "speed_of_light",
        "vacuum_",
        "electron_",
        "proton_",
        "neutron_",
        "atomic_",
        "hartree",
        "bohr",
        "compton",
        "thomson",
        "molar_gas",
        "magnetic_flux_quantum",
        "conductance_quantum",
        "fine_structure",
        "josephson",
        "von_klitzing",
        "standard_gravity",
        "tansec",  # alias misturado
    )
    n = name.lower()
    return any(m in n for m in constant_markers)


def is_likely_obscure(name: str) -> bool:
    """Filtra unidades muito obscuras (astronomia, físca de partículas
    altamente especializada) que não ocorrem em engenharia comum."""
    obscure = {
        "jansky", "sverdrup", "barn", "shed", "outhouse",
        "jerk", "snap", "crackle", "pop",  # derivadas posicionais brincadeiras
        "smoot", "wheaton", "scaramucci",  # piadas internas Pint
        "x_unit", "Xu_",
    }
    n = name.lower()
    return any(o in n for o in obscure)


def load_existing_table() -> dict:
    """Carrega o JSON atual para preservar campos custom no merge."""
    if not DIM_TABLE_PATH.exists():
        return {"version": "0.0.0", "atoms": {}}
    return json.loads(DIM_TABLE_PATH.read_text(encoding="utf-8"))


def build_table(verbose: bool = False) -> dict:
    """Constrói novo dim_table.json a partir de Pint + merge custom."""
    ureg = pint.UnitRegistry()
    existing = load_existing_table()
    existing_atoms: dict[str, dict] = existing.get("atoms", {})

    new_atoms: dict[str, dict] = {}
    stats = defaultdict(int)

    for name in ureg._units:
        # Filtros estruturais
        if name in SYMBOL_BLACKLIST:
            stats["blacklisted"] += 1
            if verbose:
                print(f"  [BLACKLIST] {name}")
            continue
        if is_likely_constant(name, ureg):
            stats["constant_excluded"] += 1
            continue
        if is_likely_obscure(name):
            stats["obscure_excluded"] += 1
            continue
        if name.startswith("_"):
            stats["private_excluded"] += 1
            continue

        try:
            q = ureg.parse_expression(name)
        except Exception:
            stats["parse_failed"] += 1
            continue

        dim_vec = pint_dim_to_si_vector(q.dimensionality)
        if dim_vec is None:
            stats["non_si_dim_excluded"] += 1
            continue
        # Princípio responsabilidade única: dim_table.json contém APENAS
        # unidades dimensionais SI (vetor ℤ⁷ não-nulo). Adimensionais
        # (razões, ângulos, dB) vão para dimensionless_units.json.
        if dim_vec == (0, 0, 0, 0, 0, 0, 0):
            stats["dimensionless_excluded"] += 1
            continue

        # Factor para SI base
        try:
            base = q.to_base_units()
            factor = float(base.magnitude)
        except Exception:
            stats["conversion_failed"] += 1
            continue

        if factor <= 0 or not (factor == factor):  # ≤0/NaN guard
            # Pint expõe constantes adimensionais negativas (g-factors:
            # g_e≈-2.0023, g_p≈5.586 com sinal). Não são unidades de
            # medida — pertencem ao módulo de constantes.
            stats["invalid_factor"] += 1
            continue
        # Filtra adimensionais sem fator-de-razão útil. `%`, `ppm`,
        # `ppb` têm dim_vec=(0,..,0) com factor>0 e SÃO incluídos
        # (categoria `ratio`). Outras entradas adimensionais (g-factors
        # já filtradas acima, refractive_index_unit já filtrada via
        # PINT_DIM_EXCLUDED) ficam fora.

        # Categoria semântica (heurística por dim_vector)
        category = DIM_TO_CATEGORY.get(dim_vec, "other")
        # SI canônico — usa nome da base unit Pint, mapeando para o
        # equivalente padrão do framework
        si_canonical = derive_si_canonical(dim_vec)

        entry = {
            "dim": list(dim_vec),
            "factor": factor,
            "si_canonical": si_canonical,
            "category": category,
        }

        # OVERRIDE SEMÂNTICO: preserva entrada manual integral quando
        # framework diverge de Pint por decisão semântica declarada
        # (e.g., temperatura como delta vs absoluta com offset).
        if name in SEMANTIC_OVERRIDES and name in existing_atoms:
            new_atoms[name] = existing_atoms[name]
            stats["semantic_override"] += 1
            continue

        # MERGE: preserva campos custom (prefixable) do JSON existente
        if name in existing_atoms:
            for custom_key in ("prefixable",):
                if custom_key in existing_atoms[name]:
                    entry[custom_key] = existing_atoms[name][custom_key]

        new_atoms[name] = entry
        stats["included"] += 1

    # EXPANSÃO POR PREFIXO SI: Pint trata muitos átomos como
    # "prefixable" dinamicamente em parse (`mmHg`, `cmHg`, `kmHg`
    # parseiam todos a partir de `mHg`), mas não registra as variantes
    # em `_units`. Para cobertura completa, geramos cada combinação
    # prefixo × átomo válido e adicionamos se Pint reconhece com
    # factor coerente. Isso multiplica cobertura sem enumerar manual.
    SI_PREFIXES = {
        "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12,
        "G": 1e9, "M": 1e6, "k": 1e3, "h": 1e2, "da": 1e1,
        "d": 1e-1, "c": 1e-2, "m": 1e-3, "µ": 1e-6, "μ": 1e-6,
        "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18,
        "z": 1e-21, "y": 1e-24,
    }
    base_atoms_snapshot = dict(new_atoms)
    for name, base_entry in base_atoms_snapshot.items():
        if name in SYMBOL_BLACKLIST:
            continue
        for prefix in SI_PREFIXES:
            prefixed = f"{prefix}{name}"
            if prefixed in new_atoms:
                continue
            if prefixed in SYMBOL_BLACKLIST:
                continue
            try:
                pq = ureg.parse_expression(prefixed)
                pfactor = float(pq.to_base_units().magnitude)
            except Exception:
                continue
            if pfactor <= 0 or not (pfactor == pfactor):
                continue
            # Verifica se a dim bate (segurança contra collision com
            # outra entrada que casualmente colide com prefix+name)
            pdim = pint_dim_to_si_vector(pq.dimensionality)
            if pdim != tuple(base_entry["dim"]):
                continue
            new_atoms[prefixed] = {
                "dim": base_entry["dim"],
                "factor": pfactor,
                "si_canonical": base_entry["si_canonical"],
                "category": base_entry["category"],
            }
            stats["prefix_expanded"] += 1

    # PRESERVA átomos custom do JSON existente que Pint não tem
    # (variantes históricas BR como `mt`, `Mt`, aliases custom).
    # Aplica os MESMOS filtros de validação: blacklist + factor>0.
    # Garante que iterações anteriores do gerador (que possam ter
    # adicionado entries inválidas como `g_e`) não sejam preservadas
    # acidentalmente no merge.
    for name, entry in existing_atoms.items():
        if name in new_atoms:
            continue
        if name in SYMBOL_BLACKLIST:
            continue
        entry_factor = entry.get("factor", 0)
        if not isinstance(entry_factor, (int, float)) or entry_factor <= 0:
            stats["existing_invalid_factor"] += 1
            continue
        entry_dim = tuple(entry.get("dim", []))
        if entry_dim == (0, 0, 0, 0, 0, 0, 0):
            # Adimensionais legados não pertencem mais a dim_table.json
            # — vão para dimensionless_units.json em arquivo próprio.
            stats["existing_dimensionless_excluded"] += 1
            continue
        new_atoms[name] = entry
        stats["preserved_custom"] += 1
        if verbose:
            print(f"  [PRESERVED-BR] {name}")

    # Estrutura final
    output = {
        "version": "0.4.0",
        "comment": (
            "Tabela dimensional ATÔMICA derivada de Pint (oráculo externo) "
            "+ preservação de unidades BR custom não-Pint. "
            "AUTO-GERADA por scripts/build_dim_table.py — não editar à mão. "
            "Princípio (memória `[Oráculo externo > enumeração]`): Pint é "
            "autoridade externa canônica; este JSON é materialização "
            "derivada para performance, determinismo e versionamento. "
            "Composições (kgf/cm², W/(m·K)) NÃO entram aqui; são "
            "computadas em dimensional/algebra.py via combine_terms(). "
            "Notação BR antiga (mt = tf·m) e variantes ASCII (kgf/cm2) "
            "são normalizadas em data/unit_aliases_v0.json. "
            "Ordem ℤ⁷: [kg, m, s, A, K, mol, cd]."
        ),
        "si_base_order": SI_BASE_ORDER,
        "atoms": dict(sorted(new_atoms.items())),
    }

    # Preserva campo `derived_si` do JSON existente se presente
    if "derived_si" in existing:
        output["derived_si"] = existing["derived_si"]

    return output, stats


def derive_si_canonical(dim_vec: tuple[int, ...]) -> str:
    """Mapeia dim_vec ℤ⁷ para nome canônico SI (string curta).

    Para dimensões nomeadas no SI, retorna o símbolo padrão (`N`, `Pa`,
    `J`); para outras, devolve a expansão dos átomos base.
    """
    named = {
        (1, 0, 0, 0, 0, 0, 0): "kg",
        (0, 1, 0, 0, 0, 0, 0): "m",
        (0, 0, 1, 0, 0, 0, 0): "s",
        (0, 0, 0, 1, 0, 0, 0): "A",
        (0, 0, 0, 0, 1, 0, 0): "K",
        (0, 0, 0, 0, 0, 1, 0): "mol",
        (0, 0, 0, 0, 0, 0, 1): "cd",
        (1, 1, -2, 0, 0, 0, 0): "N",
        (1, -1, -2, 0, 0, 0, 0): "Pa",
        (1, 2, -2, 0, 0, 0, 0): "J",
        (1, 2, -3, 0, 0, 0, 0): "W",
        (0, 0, -1, 0, 0, 0, 0): "Hz",
        (0, 0, 1, 1, 0, 0, 0): "C",
        (1, 2, -3, -1, 0, 0, 0): "V",
        (1, 2, -3, -2, 0, 0, 0): "Ω",
        (-1, -2, 4, 2, 0, 0, 0): "F",
        (1, 2, -2, -2, 0, 0, 0): "H",
        (1, 0, -2, -1, 0, 0, 0): "T",
        (1, 2, -2, -1, 0, 0, 0): "Wb",
    }
    if dim_vec in named:
        return named[dim_vec]
    # Expansão em base SI
    parts = []
    for sym, exp in zip(SI_BASE_ORDER, dim_vec, strict=True):
        if exp == 0:
            continue
        if exp == 1:
            parts.append(sym)
        else:
            parts.append(f"{sym}^{exp}")
    return "·".join(parts) or "1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Não escreve dim_table.json; só reporta")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"[build] gerando dim_table.json a partir de Pint")
    output, stats = build_table(verbose=args.verbose)

    print(f"\n[stats]")
    for k, v in sorted(stats.items()):
        print(f"  {k:30s} {v:5d}")

    print(f"\n[result] total átomos finais: {len(output['atoms'])}")

    # Diff vs anterior
    existing = load_existing_table()
    existing_atoms = set(existing.get("atoms", {}).keys())
    new_atoms_set = set(output["atoms"].keys())
    added = new_atoms_set - existing_atoms
    removed = existing_atoms - new_atoms_set
    print(f"\n[diff vs atual]")
    print(f"  átomos adicionados ({len(added)}): {sorted(added)[:30]}{'...' if len(added)>30 else ''}")
    print(f"  átomos removidos ({len(removed)}): {sorted(removed)}")

    if args.dry_run:
        print(f"\n[dry-run] não escrevendo arquivo")
        return 0

    DIM_TABLE_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[written] {DIM_TABLE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
