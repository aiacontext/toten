# TOTEN

**Typed Ontological Tokenization** — knowledge-based, deterministic tokenization of physical
quantities and technical notation in Brazilian Portuguese.

---

## The problem

Statistical tokenizers (*Byte-Pair Encoding*, *WordPiece*, *SentencePiece*) are efficient for
vocabulary compression but **semantically blind** to structured technical entities: they
fragment physical quantities, locale-specific numbers, compound dimensional units, and
symbolic expressions into lexically arbitrary subwords, leaving recomposition entirely to the
downstream model. On Brazilian-Portuguese technical text the effect is worse.

## The approach

Instead of deriving the vocabulary statistically, TOTEN **declares** a formal ontology of
engineering entities (OEE) and classifies text against it. Formally, it is the triple

```
⟨ O , classify , { inst_τ } ⟩
```

— the **ontology** holds types, structural principles, composition relations, and preservable
invariants; the **classify** function maps raw text into typed regions; and the family of
**instantiators** produces a self-describing structured representation. Robustness comes from
**deterministic coupling** to three community-maintained external oracles, instead of manual
enumeration:

- **[Pint](https://pint.readthedocs.io/)** — dimensional authority (ℤ⁷ SI vector);
- **Unicode Character Database (UCD)** — typographic authority (superscripts, fractions, math marks);
- **RSLP** (*Removedor de Sufixos da Língua Portuguesa*, via NLTK) — Portuguese morphology.

## Install

Requires Python ≥ 3.11.

```bash
pip install -e .          # or:  uv sync
```

Runtime dependencies: `regex`, `lark`, `numpy`, `pydantic`, `pyyaml`, `nltk` (the RSLP model is
downloaded on first use).

## Usage

```python
from toten import Tokenizer

t = Tokenizer.from_ontology()
print(t.preprocess("viga de 5 m com carga de 12 kN/m e σ_y = 250 MPa"))
# viga de [QTY value=5 unit=m dim=[0,1,0,0,0,0,0]] com carga de
# [QTY value=12 unit=kN/m dim=[1,0,-2,0,0,0,0]] e [SYM:σ_y] =
# [QTY value=250 unit=MPa dim=[1,-1,-2,0,0,0,0]]
```

Each `[QTY …]` carries value, unit, and the ℤ⁷ dimensional vector; `[SYM:…]` preserves the
author's symbolic expression verbatim; numbers, normative identifiers, and hierarchical
references get their own tags. (Input text is Brazilian Portuguese — the domain the system
targets.)

## The OEE — Ontology of Engineering Entities

Primary types defined by **intrinsic properties** (not by enumeration), **eight** structural
principles expressed as axioms, and declared composition relations. The ontology is
*open-for-extension, closed-for-modification*: new types/oracles are added without rewriting
existing ones. The formal definition lives in [`data/oee-v1.yaml`](data/oee-v1.yaml).

## Repository layout

```
toten/                 reference implementation (Python)
  ontology/            OEE: types, axioms, ⟨T,P,R,I⟩
  classifier/          ontological classification layer (classify)
  instantiators/       indexed family of instantiators { inst_τ }
  dimensional/         ℤ⁷ dimensional table (Pint coupling)
  activation.py        activation prompt for a frozen LLM
  pipeline.py          Tokenizer facade
data/                  pre-built oracles + intrinsic property corpora
  dim_table.json       ~18k SI atoms (auto-generated from Pint)
  *_lexicon_v0.json    lexicons (operator, identifier, expression, constant, relation)
  oee-v1.yaml          formal ontology
  intrinsic_corpus/    property pairs (dimensional equivalence, typographic robustness, numerical reconstruction)
scripts/               oracle builders (Pint → dim_table, dimensionless / info units)
tests/                 core test suite
```

## Evaluation data

The deterministic property checks (dimensional equivalence, typographic robustness, numerical
reconstruction) run against the pairs in [`data/intrinsic_corpus/`](data/intrinsic_corpus/).
The larger **EngQuant** benchmark (procedurally generated with physical validation) is
distributed separately as a dataset on the Aia Context Hugging Face
(`huggingface.co/datasets/aiacontext`), since it is data rather than code.

## Citation

If you use TOTEN, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{toten2026,
  title   = {TOTEN: Typed Ontological Tokenization},
  author  = {Leit\~ao Filho, Antonio de Sousa and Barros Filho, Allan Kardec Duailibe and
             Lima, Fabr\'icio Saul and Santos, Selby Mykael Lima dos and
             Sousa, Rejani Bandeira Vieira},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/aiacontext/toten}
}
```

## License

[Apache-2.0](LICENSE) © 2026 Aia Context.
