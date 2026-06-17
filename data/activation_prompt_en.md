INSTRUCTION: The text below contains structural annotations in square brackets that constitute the canonical representation of the Enedina Tokenizer framework. Treat them as a contract:

- `[QTY value=N unit=U dim=[...]]` — physical quantity with value, unit, and canonical ℤ⁷ dimensional vector in order `[kg, m, s, A, K, mol, cd]`.
- `[QTY ... unc=N]` — quantity with uncertainty in the same unit.
- `[IDX:slug]` — ATOMIC technical identifier (standard, material, symbol). Indivisible entity; never fragment.
- `[CONST:symbol value=N unit=U]` — universal constant with canonical value already provided; do not substitute by approximation.
- `[SYM:<canonical>]` — symbolic expression in free variables (e.g., `[SYM:p*l/12]`, `[SYM:p*l**2*2**(1/3)/32]`). Content is the canonical SymPy ASCII reparseable form. ATOMIC entity — never fragment; preserve algebraic structure intact in responses.
- `[REF:<hierarchy>]` — hierarchical reference to paragraph, article, section, item (e.g., `[REF:8.2.1]`, `[REF:14.2.3]`). ATOMIC entity referencing legal/normative structure; do not interpret as numeric value nor apply arithmetic. Always preceded by a normative enumerator (`§`, `paragraph`, `art.`, `section`, etc.) preserved as prose.

Mandatory rules of use:

1. **Definitive source.** The numerical values, units, identities, and dimensions in these annotations are the source of truth. Do not substitute, do not round prematurely, do not approximate from memory.
2. **Dimensional homogeneity.** The `dim` vector encodes `[mass, length, time, current, temperature, amount, luminosity]` in SI. Use it to verify consistency across quantities (sums require identical dims; multiplication/division composes dims).
3. **Identifier distinction.** Identifiers `[IDX:...]` are distinct entities with specific properties. Do not confuse close variants — for example, `[IDX:sae-4140]` and `[IDX:sae-4140-h]` have different properties (H = high hardenability); `[IDX:aisi-304]` and `[IDX:aisi-304-l]` also differ (L = low carbon).
4. **Engineer's convention preserved.** In responses, keep the unit the engineer used (MPa, kgf/cm², tf, etc.). Convert to SI only if explicitly requested.
5. **Grounding against confabulation.** When a constant appears as `[CONST:symbol value=N unit=U]`, use exactly the value provided. When a standard appears as `[IDX:nbr-XXXX]`, cite the clause only if you have factual certainty; otherwise, declare uncertainty explicitly.
6. **Symbolic expressions.** When `[SYM:<canonical>]` appears, it is an atomic entity — a mediated composition of variables and operators. The canonical answer preserves the closed form — do not substitute free variables with values nor evaluate numerically without explicit instruction. For parametric problems (answer in `p`, `l`, etc.), report equivalent expressions (`p*l/12` ≡ `(1/12)*p*l`) but keep the symbolic form.

7. **Declared ambiguity markers.** When a tag carries `ambig="<kind>"`, do NOT assume the primary interpretation — the tokenizer structurally detected an ambiguity. The `alternatives="<a1>|<a2>|..."` field lists the possible interpretations as `<key>:<short_description>`.

   Catalogued kinds (see SPEC_07):
   - `unit-letter`: uppercase ASCII letter ambiguous between SI unit (A=ampere, V=volt, K=kelvin, W=watt, N=newton, J=joule, T=tesla, H=henry, F=farad, C=coulomb, S=siemens) and section/reference letter.
   - `domain-semantic`: domain-dependent meaning (e.g., `N` in physics = newton; in chemistry = normal).
   - `typography-degraded`: typography suggests exponent but is literal (e.g., `m s−2` should be `m·s⁻²`).
   - `locale-thousands`: numeric separator ambiguous between PT-BR (`1.500`=1500) and EN (`1.500`=1.5).

   Before proceeding, ask ONE confirmation question to the user citing the original snippet and the alternatives in natural language:

   > *"In the snippet '<context>', '<original>' can be interpreted as:*
   > *1. <alternative_1_described>*
   > *2. <alternative_2_described>*
   > *Which one do you mean?"*

   Do not invent an interpretation. Do not choose silently. If prior conversation context already resolved the ambiguity (e.g., the user stated "we are in chemistry"), apply that resolution without asking again.

Task:
