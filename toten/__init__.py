"""Enedina Tokenizer — framework ontológico para representação determinística
de grandezas físicas em LLMs de engenharia.

API pública conforme spec §6.2:

    from toten import Tokenizer

    tok = Tokenizer.from_ontology("oee-v1")
    canonical = tok.preprocess(
        "A tensão de 350 MPa em AISI 1045 supera σ_y = 215 MPa"
    )
"""

from toten.activation import (
    ActivatedPrompt,
    default_activation_prompt,
    load_activation_prompt,
)
from toten.pipeline import Tokenizer

__version__ = "0.6.0"
__all__ = [
    "Tokenizer",
    "ActivatedPrompt",
    "load_activation_prompt",
    "default_activation_prompt",
]
