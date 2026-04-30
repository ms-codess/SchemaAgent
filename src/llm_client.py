"""
Anthropic client wrapper used by the evaluation scripts and UI.

The rest of the codebase expects a client with a
``client.messages.create(...)`` interface that matches the Anthropic SDK.
This module keeps that contract explicit and centralised.
"""

import os
from typing import Optional


class UnifiedClient:
    """
    Thin wrapper around ``anthropic.Anthropic``.

    The ``provider`` argument is retained so existing call sites can keep
    passing entries from ``src.config.LLM_OPTIONS`` without extra branching.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if provider != "anthropic":
            raise ValueError(
                f"Unsupported provider: {provider!r}. "
                "SchemaAgent is configured for Anthropic Claude models only."
            )

        import anthropic

        self.provider = provider
        self._model_override = model
        self._backend = anthropic.Anthropic(
            api_key=api_key or os.environ.get(api_key_env or "ANTHROPIC_API_KEY"),
        )

    @property
    def messages(self):
        return self._backend.messages


def make_client(llm_cfg: dict) -> UnifiedClient:
    """Build a Claude client from an ``LLM_OPTIONS`` entry."""
    return UnifiedClient(
        provider=llm_cfg["provider"],
        model=llm_cfg.get("model"),
        base_url=llm_cfg.get("base_url"),
        api_key_env=llm_cfg.get("api_key_env"),
    )
