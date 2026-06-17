"""Camada 1 da arquitetura — Ontologia declarativa.

Carrega, valida e expõe a Ontologia das Entidades em Engenharia (OEE).
A ontologia é dado (`data/oee-v1.yaml`), não código. Este módulo provê
acesso tipado via pydantic.
"""

from toten.ontology.loader import default_oee_path, load_oee
from toten.ontology.schema import OEE, Principio, TipoOntologico
from toten.ontology.types import TipoNome

__all__ = [
    "load_oee",
    "default_oee_path",
    "OEE",
    "TipoOntologico",
    "Principio",
    "TipoNome",
]
