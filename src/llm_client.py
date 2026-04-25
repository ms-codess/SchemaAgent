"""
Unified LLM client — wraps Anthropic and Hugging Face providers behind a single
interface that mirrors anthropic.Anthropic().messages.create().

This lets SQLAgent, HybridFusion, and IntentRouter remain unchanged while
supporting Claude (Anthropic SDK) and Qwen2.5-Coder-32B (HuggingFace
InferenceClient — already in requirements, no extra dependency).

Providers
---------
  "anthropic"   — anthropic.Anthropic()          requires ANTHROPIC_API_KEY
  "huggingface" — huggingface_hub.InferenceClient requires HF_TOKEN
  "openai_compat" — openai.OpenAI()              requires openai pkg + custom key

Usage
-----
from src.llm_client import UnifiedClient, make_client
from src.config import LLM_OPTIONS

client = make_client(LLM_OPTIONS["Qwen2.5-Coder-32B"])
response = client.messages.create(model=..., max_tokens=..., system=..., messages=...)
text = response.content[0].text
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Shim response types (mirror the Anthropic SDK surface used by this project) ─

@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Content:
    text: str


@dataclass
class _Response:
    content: List[_Content]
    usage: _Usage


# ── Internal messages-API facade ──────────────────────────────────────────────

class _MessagesAPI:
    def __init__(self, owner: "UnifiedClient"):
        self._owner = owner

    def create(self, model: str, max_tokens: int, system, messages: list, **kwargs):
        return self._owner._dispatch(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )


# ── Public client ─────────────────────────────────────────────────────────────

class UnifiedClient:
    """
    Drop-in replacement for anthropic.Anthropic() supporting multiple providers.

    Parameters
    ----------
    provider : "anthropic" | "huggingface" | "openai_compat"
    model : str, optional
        Override the model ID for every call.
    base_url : str, optional
        Only used by "openai_compat" provider.
    api_key_env : str, optional
        Env var name for the API key (default varies by provider).
    api_key : str, optional
        Explicit API key (overrides api_key_env lookup).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self._model_override = model
        self._backend = None

        if provider == "anthropic":
            import anthropic
            self._backend = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            )

        elif provider == "huggingface":
            from huggingface_hub import InferenceClient
            env_var = api_key_env or "HF_TOKEN"
            token = api_key or os.environ.get(env_var)
            if not token:
                raise EnvironmentError(
                    f"Hugging Face token not found. Set the {env_var!r} environment variable "
                    "(get yours at https://huggingface.co/settings/tokens)."
                )
            self._backend = InferenceClient(token=token)

        elif provider == "openai_compat":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for openai_compat provider. "
                    "Run: pip install openai"
                ) from exc
            env_var = api_key_env or "TOGETHER_API_KEY"
            resolved_key = api_key or os.environ.get(env_var) or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise EnvironmentError(
                    f"API key not found. Set the {env_var!r} environment variable."
                )
            self._backend = OpenAI(
                api_key=resolved_key,
                base_url=base_url or "https://api.together.xyz/v1",
            )

        else:
            raise ValueError(f"Unknown provider: {provider!r}")

        self._messages = _MessagesAPI(self)

    @property
    def messages(self) -> _MessagesAPI:
        return self._messages

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, model: str, max_tokens: int, system, messages: list):
        effective_model = self._model_override or model

        if self.provider == "anthropic":
            # Pass straight through — Anthropic response already has the right shape.
            return self._backend.messages.create(
                model=effective_model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )

        if self.provider == "huggingface":
            return self._call_huggingface(effective_model, max_tokens, system, messages)

        return self._call_openai_compat(effective_model, max_tokens, system, messages)

    # ── HuggingFace InferenceClient ───────────────────────────────────────────

    def _call_huggingface(
        self, model: str, max_tokens: int, system, messages: list
    ) -> _Response:
        """Call HF Inference API via huggingface_hub.InferenceClient.chat_completion."""
        system_text = self._extract_text(system)
        hf_messages = [{"role": "system", "content": system_text}]
        for msg in messages:
            hf_messages.append({
                "role": msg["role"],
                "content": self._extract_text(msg.get("content", "")),
            })

        logger.debug("HuggingFace call → model=%s, max_tokens=%d", model, max_tokens)

        resp = self._backend.chat_completion(
            model=model,
            messages=hf_messages,
            max_tokens=max_tokens,
        )

        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        return _Response(
            content=[_Content(text=text)],
            usage=_Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            ),
        )

    # ── OpenAI-compat (Together.ai, etc.) ─────────────────────────────────────

    def _call_openai_compat(
        self, model: str, max_tokens: int, system, messages: list
    ) -> _Response:
        """Convert Anthropic-style args → OpenAI format, call, wrap response."""
        system_text = self._extract_text(system)
        oai_messages = [{"role": "system", "content": system_text}]
        for msg in messages:
            oai_messages.append({
                "role": msg["role"],
                "content": self._extract_text(msg.get("content", "")),
            })

        logger.debug("OpenAI-compat call → model=%s, max_tokens=%d", model, max_tokens)

        resp = self._backend.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=oai_messages,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return _Response(
            content=[_Content(text=text)],
            usage=_Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            ),
        )

    # ── Shared helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(value) -> str:
        """Flatten a string, Anthropic block-list, or content-list to plain text."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [
                (b.get("text", "") if isinstance(b, dict) else str(b))
                for b in value
            ]
            return " ".join(p for p in parts if p)
        return str(value)


# ── Convenience factory ───────────────────────────────────────────────────────

def make_client(llm_cfg: dict) -> UnifiedClient:
    """Build a UnifiedClient from an LLM_OPTIONS entry."""
    return UnifiedClient(
        provider=llm_cfg["provider"],
        model=llm_cfg.get("model"),
        base_url=llm_cfg.get("base_url"),
        api_key_env=llm_cfg.get("api_key_env"),
    )
