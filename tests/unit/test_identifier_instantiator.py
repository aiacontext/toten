"""Testes Dia 3.5 — IdentificadorTecnicoInstantiator (Camada 3).

Avalia:
- Canonicalização determinística (NFKD + strip diacritics + slug).
- Preservação atômica de identidade em forma `IDX:<slug>`.
- Idempotência: canonicalizar duas vezes não muda o resultado.
- Integração com Region da Camada 2.
- Modo A vs Modo B.
"""

from __future__ import annotations

import pytest

from toten.classifier import OntologicalClassifier, Region
from toten.instantiators import (
    IdentificadorTecnicoInstantiator,
    canonicalize_identifier,
)
from toten.ontology.types import TipoNome

# ---------------------------------------------------------------------------
# canonicalize_identifier — diretamente
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "slug_esperado"),
    [
        ("Hardox 500", "hardox-500"),
        ("Niobrás 320", "niobras-320"),
        ("Niobrás", "niobras"),
        ("ABNT NBR 12655", "abnt-nbr-12655"),
        ("ABNT NBR ISO 9001:2015", "abnt-nbr-iso-9001-2015"),
        ("Ti-6Al-4V", "ti-6al-4v"),
        ("AISI 1045", "aisi-1045"),
        ("USI-AR-400", "usi-ar-400"),
        ("CP-V ARI-RS", "cp-v-ari-rs"),
        ("CA-50A", "ca-50a"),
        ("ER70S-6", "er70s-6"),
        ("Al 6061-T6", "al-6061-t6"),
        ("aço Hardox 500", "aco-hardox-500"),
    ],
)
def test_canonicalize_casos_basicos(entrada: str, slug_esperado: str) -> None:
    assert canonicalize_identifier(entrada) == slug_esperado


@pytest.mark.parametrize(
    ("entrada", "slug_esperado"),
    [
        # Princípio: underscore ASCII é marcador VISUAL de subscript em
        # notação técnica NBR (f_{ck} formal vs fck inline). Removido na
        # canonicalização para colapsar ambas variantes ao MESMO slug.
        ("σ_y", "σy"),
        ("σ_eq", "σeq"),
        ("τ_y", "τy"),
        ("f_ck", "fck"),
        ("E_s", "es"),
        ("ν_c", "νc"),
    ],
)
def test_canonicalize_preserva_letras_gregas(
    entrada: str, slug_esperado: str
) -> None:
    """Letras gregas (σ, τ, ν, π) NÃO são transliteradas — preservadas
    como Unicode no slug. NFKD as deixa intactas. Underscore ASCII removido
    (marcador de subscript visual, não separador semântico)."""
    assert canonicalize_identifier(entrada) == slug_esperado


def test_canonicalize_idempotente() -> None:
    """Canonicalizar um slug já canônico devolve o mesmo slug."""
    for entrada in ["Hardox 500", "Niobrás 320", "σ_y", "USI-AR-400"]:
        primeiro = canonicalize_identifier(entrada)
        segundo = canonicalize_identifier(primeiro)
        assert primeiro == segundo


def test_canonicalize_deterministico() -> None:
    """Mesma entrada → mesmo slug, sempre."""
    entradas = ["Hardox 500", "Niobrás 320", "ABNT NBR 12655"]
    for entrada in entradas:
        s1 = canonicalize_identifier(entrada)
        s2 = canonicalize_identifier(entrada)
        assert s1 == s2


def test_canonicalize_colapsa_separadores_consecutivos() -> None:
    """Múltiplos espaços, hífens, ou underscores viram um único `-`."""
    assert canonicalize_identifier("Hardox    500") == "hardox-500"
    assert canonicalize_identifier("Hardox---500") == "hardox-500"
    assert canonicalize_identifier("Hardox _ 500") == "hardox-500"


def test_canonicalize_strip_hifens_extremos() -> None:
    assert canonicalize_identifier("-Hardox 500-") == "hardox-500"
    assert canonicalize_identifier("- USI-AR-400 -") == "usi-ar-400"


def test_canonicalize_rejeita_entrada_vazia() -> None:
    with pytest.raises(ValueError, match="não-vazia"):
        canonicalize_identifier("")


def test_canonicalize_rejeita_slug_que_resulta_vazio() -> None:
    """Entrada só com pontuação resulta em slug vazio — rejeitado."""
    with pytest.raises(ValueError, match="slug vazio"):
        canonicalize_identifier("---")
    with pytest.raises(ValueError, match="slug vazio"):
        canonicalize_identifier("   ")


# ---------------------------------------------------------------------------
# IdentificadorTecnicoInstantiator — interface via Region
# ---------------------------------------------------------------------------


@pytest.fixture
def instantiator() -> IdentificadorTecnicoInstantiator:
    return IdentificadorTecnicoInstantiator()


def test_instantiate_modo_b(instantiator: IdentificadorTecnicoInstantiator) -> None:
    region = Region(TipoNome.IDENTIFICADOR_TECNICO, 0, 10, "Hardox 500")
    token = instantiator.instantiate(region, mode="B")
    assert token.text == "[IDX:hardox-500]"
    assert token.slug == "hardox-500"
    assert token.raw == "Hardox 500"


def test_instantiate_modo_a(instantiator: IdentificadorTecnicoInstantiator) -> None:
    region = Region(TipoNome.IDENTIFICADOR_TECNICO, 0, 10, "Hardox 500")
    token = instantiator.instantiate(region, mode="A")
    assert token.text == "<ID>hardox-500</ID>"


def test_instantiate_rejeita_tipo_errado(
    instantiator: IdentificadorTecnicoInstantiator,
) -> None:
    region = Region(TipoNome.GRANDEZA_FISICA, 0, 7, "350 MPa")
    with pytest.raises(TypeError, match="IdentificadorTecnico"):
        instantiator.instantiate(region)


def test_instantiate_text_atalho(
    instantiator: IdentificadorTecnicoInstantiator,
) -> None:
    assert instantiator.instantiate_text("Niobrás 320") == "[IDX:niobras-320]"
    assert (
        instantiator.instantiate_text("Niobrás 320", mode="A")
        == "<ID>niobras-320</ID>"
    )


# ---------------------------------------------------------------------------
# Integração Camada 2 → Camada 3 (atomicidade preservada)
# ---------------------------------------------------------------------------


def test_round_trip_camada2_camada3() -> None:
    """Pega regiões IT do classificador e canonicaliza — todas viram tokens
    atômicos, mesmo identificadores não-curados (cobertos por estrutural)."""
    classifier = OntologicalClassifier()
    instantiator = IdentificadorTecnicoInstantiator()

    texto = (
        "viga em concreto C30 com aço CA-50, cimento CP-V ARI, "
        "aço Hardox 500 abrasivo, ensaiada por UT"
    )
    regions = classifier.classify(texto)
    ids = [r for r in regions if r.tipo is TipoNome.IDENTIFICADOR_TECNICO]
    tokens = [instantiator.instantiate(r) for r in ids]
    slugs = {tok.slug for tok in tokens}

    assert "c30" in slugs
    assert "ca-50" in slugs
    assert "cp-v-ari" in slugs
    assert "hardox-500" in slugs
    assert "ut" in slugs


def test_identificadores_distintos_geram_slugs_distintos() -> None:
    """Identificadores semanticamente distintos geram slugs distintos."""
    inputs = [
        "Hardox 500",
        "Hardox 600",
        "Niobrás 320",
        "CA-50",
        "CA-60",
        "CP-V",
        "CP-V ARI",
    ]
    slugs = {canonicalize_identifier(s) for s in inputs}
    assert len(slugs) == len(inputs), f"colisão em {slugs}"


def test_canonicalize_unifica_variantes_ortograficas() -> None:
    """Variantes ortográficas (com/sem diacrítico) convergem para o mesmo
    slug — comportamento desejado: `Niobrás` e `Niobras` referem-se à
    mesma marca, então devem canonicar idêntico."""
    assert canonicalize_identifier("Niobrás") == canonicalize_identifier("Niobras")
    assert canonicalize_identifier("aço") == canonicalize_identifier("aco")
    assert (
        canonicalize_identifier("Niobrás 320")
        == canonicalize_identifier("Niobras 320")
        == "niobras-320"
    )
