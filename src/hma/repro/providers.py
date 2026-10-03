"""Credential injection at execution time, never into experiment manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hma.repro.config import Local


def names(provider: str) -> list[str]:
    prefix = "HMA_" + provider.upper()
    return [prefix + "_API_KEY", prefix + "_BASE_URL"]


def credentials(local: Local, providers: set[str]) -> dict[str, str]:
    result = {}
    for provider in sorted(providers):
        if provider not in local.providers:
            raise ValueError(f"provider not configured: {provider}")
        p = local.providers[provider]
        for name, value in zip(names(provider), (p.api_key, p.base_url)):
            value = os.environ.get(name) or value
            if not value:
                raise ValueError(
                    f"fill {provider} api_key/base_url or set {name}; templates are intentionally blank"
                )
            if name.endswith("_BASE_URL") and not value.startswith(("https://", "http://")):
                raise ValueError(f"{name} must be an HTTP(S) URL")
            result[name] = value
    return result


def bootstrap() -> None:
    """Translate this actor's provider into its native CLI configuration."""
    provider = os.environ.get("HMA_PROVIDER", "")
    if not provider:
        return  # Existing manually staged hma-run configurations still work.
    key_name, url_name = names(provider)
    key, url = os.environ[key_name], os.environ[url_name]
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = key
        config = Path.home() / ".codex/config.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            'model_provider = "repro"\n[model_providers.repro]\nname = "Reproduction endpoint"\n'
            + f'base_url = {json.dumps(url)}\nenv_key = "OPENAI_API_KEY"\nwire_api = "responses"\n'
        )
        config.chmod(0o600)
    elif provider in {"anthropic", "glm"}:
        os.environ.update(ANTHROPIC_API_KEY=key, ANTHROPIC_BASE_URL=url)
    elif provider == "kimi":
        os.environ.update(KIMI_MODEL_API_KEY=key, KIMI_MODEL_BASE_URL=url)
    elif provider == "deepseek":
        os.environ.update(DEEPSEEK_API_KEY=key, DEEPSEEK_BASE_URL=url)
        # The external harnesses use two roles on the same declared backbone.
        for role in ("CODE", "FEEDBACK"):
            os.environ[f"PAPER_{role}_API_KEY"] = key
            os.environ[f"PAPER_{role}_BASE_URL"] = url
    else:
        raise ValueError(f"unsupported provider: {provider}")
