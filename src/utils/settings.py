"""Centralized configuration management for Enlitens agents and services."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONFIG_ENV_VAR = "ENLITENS_CONFIG_FILE"


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` and return ``base``."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class ProviderSettings:
    """Settings that control how the LLM backend is accessed."""

    provider: str = "vllm"
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    monitoring_model: Optional[str] = None
    endpoints: Dict[str, str] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    local_weights_path: Optional[str] = None
    api_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def model_for(self, key: Optional[str]) -> Optional[str]:
        """Return the preferred model for ``key`` falling back to the default."""

        if key:
            normalised = key.replace("_", "-").lower()
            if normalised in self.models:
                return self.models[normalised]
        return self.default_model

    def endpoint_for(self, key: Optional[str]) -> Optional[str]:
        if key:
            normalised = key.replace("_", "-").lower()
            if normalised in self.endpoints:
                return self.endpoints[normalised]
        return self.base_url


@dataclass
class Settings:
    """Top level configuration container."""

    llm: ProviderSettings = field(default_factory=ProviderSettings)

    def model_for_agent(self, agent_name: str) -> Optional[str]:
        return self.llm.model_for(agent_name)


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - defensive guard
        logger.warning("PyYAML not installed; skipping YAML configuration load")
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError("Configuration root must be a mapping")
            return data
    except FileNotFoundError:
        logger.debug("Configuration file %s not found", path)
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load configuration from %s: %s", path, exc)
        return {}


def _load_env_overrides() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"llm": {"models": {}, "endpoints": {}}}

    llm = overrides["llm"]
    provider = os.environ.get("LLM_PROVIDER")
    if provider:
        llm["provider"] = provider

    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        llm["base_url"] = base_url

    default_model = os.environ.get("LLM_MODEL_DEFAULT")
    if default_model:
        llm["default_model"] = default_model

    monitoring_model = os.environ.get("LLM_MODEL_MONITORING")
    if monitoring_model:
        llm["monitoring_model"] = monitoring_model

    local_weights = os.environ.get("LLM_LOCAL_WEIGHTS")
    if local_weights:
        llm["local_weights_path"] = local_weights

    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        llm["api_key"] = api_key

    for key, value in os.environ.items():
        if key.startswith("LLM_MODEL__"):
            llm["models"][key[len("LLM_MODEL__"):].replace("__", "-").lower()] = value
        elif key.startswith("LLM_ENDPOINT__"):
            llm["endpoints"][key[len("LLM_ENDPOINT__"):].replace("__", "-").lower()] = value

    return overrides


def _coerce_provider_settings(data: Dict[str, Any]) -> ProviderSettings:
    llm_data = data.get("llm", {}) if isinstance(data, dict) else {}
    if isinstance(llm_data.get("models"), list):
        llm_data["models"] = {entry.get("name"): entry.get("model") for entry in llm_data["models"] if isinstance(entry, dict)}
    if isinstance(llm_data.get("endpoints"), list):
        llm_data["endpoints"] = {entry.get("name"): entry.get("url") for entry in llm_data["endpoints"] if isinstance(entry, dict)}

    models = llm_data.get("models", {}) or {}
    llm_data["models"] = {str(key).replace("_", "-").lower(): value for key, value in models.items() if value}

    endpoints = llm_data.get("endpoints", {}) or {}
    llm_data["endpoints"] = {str(key).replace("_", "-").lower(): value for key, value in endpoints.items() if value}

    return ProviderSettings(
        provider=str(llm_data.get("provider", "vllm")).lower(),
        base_url=llm_data.get("base_url"),
        default_model=llm_data.get("default_model"),
        monitoring_model=llm_data.get("monitoring_model"),
        endpoints=llm_data.get("endpoints", {}),
        models=llm_data.get("models", {}),
        local_weights_path=llm_data.get("local_weights_path"),
        api_key=llm_data.get("api_key"),
        extra={key: value for key, value in llm_data.items() if key not in {
            "provider",
            "base_url",
            "default_model",
            "monitoring_model",
            "endpoints",
            "models",
            "local_weights_path",
            "api_key",
        }},
    )


def _build_settings() -> Settings:
    data: Dict[str, Any] = {}

    config_file = os.environ.get(_CONFIG_ENV_VAR)
    if config_file:
        data = _load_yaml_config(Path(config_file))

    env_overrides = _load_env_overrides()
    _deep_update(data.setdefault("llm", {}), env_overrides.get("llm", {}))

    provider_settings = _coerce_provider_settings(data)

    if not provider_settings.default_model:
        provider_settings.default_model = "/home/antons-gs/enlitens-ai/models/llama-3.1-8b-instruct"

    if not provider_settings.base_url:
        provider_settings.base_url = "http://localhost:8000/v1"

    if not provider_settings.local_weights_path:
        provider_settings.local_weights_path = provider_settings.default_model

    return Settings(llm=provider_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached configuration settings."""

    settings = _build_settings()
    logger.debug("Loaded settings: %s", json.dumps({
        "provider": settings.llm.provider,
        "base_url": settings.llm.base_url,
        "default_model": settings.llm.default_model,
        "monitoring_model": settings.llm.monitoring_model,
    }))
    return settings


def reset_settings_cache() -> None:
    """Reset the cached settings to force a reload on next access."""

    get_settings.cache_clear()  # type: ignore[attr-defined]
