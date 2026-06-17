"""Carregamento da OEE a partir de YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from toten.ontology.schema import OEE

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OEE_PATH = PACKAGE_ROOT / "data" / "oee-v1.yaml"


def default_oee_path() -> Path:
    """Retorna o caminho da OEE v1.0 distribuída com o pacote."""
    return DEFAULT_OEE_PATH


def load_oee(path: Path | str | None = None) -> OEE:
    """Carrega e valida uma ontologia OEE.

    Parameters
    ----------
    path:
        Caminho do YAML. Se None, usa a OEE v1.0 default do pacote.

    Raises
    ------
    FileNotFoundError
        se o YAML não existir.
    pydantic.ValidationError
        se o YAML violar o esquema da OEE ou alguma invariante da v1.0.
    """
    yaml_path = Path(path) if path is not None else default_oee_path()
    if not yaml_path.is_file():
        msg = f"OEE YAML não encontrado em {yaml_path}"
        raise FileNotFoundError(msg)

    with yaml_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    return OEE.model_validate(raw)
