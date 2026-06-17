"""Activation prompts — contrato explícito que ativa o framework no LLM.

Spec §13.11. O preprocessing Modo B sozinho NÃO produz ganho mensurável
em LLMs frozen — o piloto adversarial revelou que frontier LLMs tratam
o markup como opcional quando não-anunciado. O ganho concentra-se na
condição C (Modo B + ativação explícita): system prompt que declara o
contrato (tipos das tags, regras de uso, fonte de verdade, preservação
de convenção do engenheiro).

Este módulo trata a ativação como artefato de primeira classe:

- Fontes canônicas em `data/activation_prompt_pt_br.md` e
  `data/activation_prompt_en.md` — versionadas com o framework.
- Carregamento via `load_activation_prompt(lang)` ou
  `default_activation_prompt(lang)` (cacheado).
- `ActivatedPrompt` dataclass empacota ativação + texto preprocessado,
  com helpers `ready_for_llm` (string para colar em chat) e
  `as_messages()` (formato `{role, content}` para API).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"pt-br", "en"})


def _activation_prompt_path(lang: str) -> Path:
    suffix = lang.replace("-", "_")
    return DATA_DIR / f"activation_prompt_{suffix}.md"


def load_activation_prompt(lang: str = "pt-br") -> str:
    """Carrega o prompt de ativação canônico para o idioma."""
    if lang not in SUPPORTED_LANGUAGES:
        msg = (
            f"Idioma de ativação não suportado: {lang!r}. "
            f"Disponíveis: {sorted(SUPPORTED_LANGUAGES)}."
        )
        raise ValueError(msg)
    path = _activation_prompt_path(lang)
    if not path.is_file():
        msg = f"Activation prompt não encontrado em {path}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8").rstrip() + "\n"


@lru_cache(maxsize=4)
def default_activation_prompt(lang: str = "pt-br") -> str:
    """Cacheia o prompt default por idioma."""
    return load_activation_prompt(lang)


@dataclass(frozen=True, slots=True)
class ActivatedPrompt:
    """Pacote ativação + texto preprocessado, pronto para consumo por LLM.

    - `activation` é o system prompt explicando o contrato.
    - `preprocessed` é o texto Modo B (ou Modo A) já tokenizado.
    - `language` indica o idioma da ativação ("pt-br" | "en").

    Use `ready_for_llm` para colar em interface de chat; use
    `as_messages()` para chamadas de API (Anthropic, OpenAI, Gemini).
    """

    activation: str
    preprocessed: str
    language: str

    def __post_init__(self) -> None:
        if not self.activation.strip():
            msg = "ActivatedPrompt exige activation não-vazia"
            raise ValueError(msg)
        if not self.preprocessed:
            msg = "ActivatedPrompt exige preprocessed não-vazio"
            raise ValueError(msg)
        if self.language not in SUPPORTED_LANGUAGES:
            msg = f"language deve ser um de {sorted(SUPPORTED_LANGUAGES)}"
            raise ValueError(msg)

    @property
    def ready_for_llm(self) -> str:
        """String única ativação + texto, formatada para colar em chat."""
        return f"{self.activation.rstrip()}\n\n{self.preprocessed}"

    def as_messages(self) -> list[dict[str, str]]:
        """Formato `[{role, content}]` aceito por Anthropic/OpenAI/Gemini.

        Convenção: ativação em `system`, texto preprocessado em `user`.
        Compatível com:
        - `anthropic.messages.create(system=..., messages=[...])` —
          passe `activation` direto para `system` e omita a entrada
          system desta lista.
        - `openai.chat.completions.create(messages=[...])` — passe a
          lista inteira.
        - `google.generativeai` — adaptar conforme schema do SDK.
        """
        return [
            {"role": "system", "content": self.activation.rstrip()},
            {"role": "user", "content": self.preprocessed},
        ]
