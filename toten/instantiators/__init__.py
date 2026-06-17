"""Camada 3 da arquitetura — Instanciadores.

Cada tipo da OEE possui instanciador dedicado que converte uma região
do classificador na representação derivada das propriedades intrínsecas
do tipo (spec §3.4).

Estado atual (Dia 3.5): IdentificadorTecnicoInstantiator implementado
para honrar a tese de que o framework preserva identidade atomicamente,
sem fallback fragmentário. Os demais instanciadores
(GrandezaFisicaInstantiator central, ProsaTecnica, OperadorFormal,
ConstanteUniversal, RelacaoEstrutural) entram nos Dias 4-5.

Política firme: tudo que a Camada 2 detectar como IdentificadorTecnico
é preservado atomicamente pela Camada 3 — nunca fragmentado. Mesmo
identificadores fora do vocab curado recebem forma canônica
`IDX:<slug>` derivada determinística da string, garantindo que o
modelo consumidor enxergue *um* objeto tipado, não N caracteres.
"""

from toten.instantiators.constant import (
    ConstanteUniversalInstantiator,
    ConstantToken,
)
from toten.instantiators.expression import (
    ExpressaoSimbolicaInstantiator,
    SymbolicExpression,
    try_parse_symbolic,
)
from toten.instantiators.identifier import (
    IdentificadorTecnicoInstantiator,
    canonicalize_identifier,
)
from toten.instantiators.numero import (
    Numero,
    NumeroInstantiator,
    NumeroToken,
    parse_numero,
)
from toten.instantiators.reference import (
    Referencia,
    ReferenciaInstantiator,
    ReferenciaToken,
    parse_referencia,
)
from toten.instantiators.operator import (
    OperadorFormalInstantiator,
    OperatorToken,
)
from toten.instantiators.prose import (
    BPEBackend,
    ProsaTecnicaInstantiator,
    ProseToken,
)
from toten.instantiators.quantity import (
    GrandezaFisicaInstantiator,
    Quantity,
    QuantityToken,
    UnitTerm,
    canonicalize_number,
    parse_grandeza,
    parse_unit_composition,
)
from toten.instantiators.relation import (
    RelacaoEstruturalInstantiator,
    RelationToken,
)

__all__ = [
    "ExpressaoSimbolicaInstantiator",
    "SymbolicExpression",
    "try_parse_symbolic",
    "IdentificadorTecnicoInstantiator",
    "canonicalize_identifier",
    "NumeroInstantiator",
    "Numero",
    "NumeroToken",
    "parse_numero",
    "ReferenciaInstantiator",
    "Referencia",
    "ReferenciaToken",
    "parse_referencia",
    "GrandezaFisicaInstantiator",
    "Quantity",
    "QuantityToken",
    "UnitTerm",
    "canonicalize_number",
    "parse_grandeza",
    "parse_unit_composition",
    "ProsaTecnicaInstantiator",
    "ProseToken",
    "BPEBackend",
    "OperadorFormalInstantiator",
    "OperatorToken",
    "ConstanteUniversalInstantiator",
    "ConstantToken",
    "RelacaoEstruturalInstantiator",
    "RelationToken",
]
