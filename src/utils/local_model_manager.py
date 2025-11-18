#!/usr/bin/env python3
"""
Helpers for starting/stopping local vLLM/SGLang servers per model.

The comparison pipeline needs to swap between MedGemma and Llama with
minimal manual steps.  This module reads `config/local_models.yaml` and
provides convenience wrappers around the start/stop commands defined
there.  If no commands are provided, the helpers simply log guidance so
the operator can start the server manually.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/local_models.yaml")


class LocalModelConfig(Dict[str, str]):
    """Typed alias for model configuration entries."""


class LocalModelManager:
    """Runtime helper for loading model configs and orchestrating servers."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        if not self.config_path.exists():
            raise FileNotFoundError(f"Local model config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.models: Dict[str, LocalModelConfig] = yaml.safe_load(handle) or {}
        if not self.models:
            raise ValueError(f"No models defined in {self.config_path}")
        logger.info("📚 Loaded %d local model configs from %s", len(self.models), self.config_path)

    def get(self, key: str) -> LocalModelConfig:
        if key not in self.models:
            raise KeyError(f"Model '{key}' not defined in {self.config_path}")
        return self.models[key]

    def start(self, key: str) -> None:
        """Invoke the configured start command for a model."""
        config = self.get(key)
        command = config.get("start_command")
        if not command:
            logger.info("ℹ️ No start_command specified for %s; assume server already running.", key)
            return
        logger.info("🚀 Starting %s via `%s`", key, command)
        self._run_shell(command)
        self.wait_until_ready(key)

    def stop(self, key: str) -> None:
        """Invoke the configured stop command for a model (if any)."""
        config = self.get(key)
        command = config.get("stop_command")
        if not command:
            logger.info("ℹ️ No stop_command for %s; skipping.", key)
            return
        logger.info("🛑 Stopping %s via `%s`", key, command)
        self._run_shell(command)

    def clear_gpu(self) -> None:
        """
        Attempt to clear GPU memory by calling `nvidia-smi --gpu-reset`
        when supported. Errors are logged but non-fatal.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--gpu-reset"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info("🔄 GPU reset requested: %s", result.stdout.strip())
            else:
                logger.debug("nvidia-smi reset returned %s: %s", result.returncode, result.stderr.strip())
        except FileNotFoundError:
            logger.debug("nvidia-smi not found; skipping GPU reset.")

    @staticmethod
    def _run_shell(command: str) -> None:
        """Execute a shell command with inherited environment."""
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as exc:
            logger.error("Command `%s` failed with code %s", command, exc.returncode)
            raise

    def describe(self, key: str) -> str:
        """Return human-readable description of a model config."""
        config = self.get(key)
        desc = config.get("description") or ""
        model_name = config.get("model_name")
        base_url = config.get("base_url")
        return f"{key} -> {model_name} ({base_url}) {desc}"

    def wait_until_ready(self, key: str, timeout: int = 180, interval: float = 2.0) -> None:
        """Poll the model's /v1/models endpoint until it responds or timeout."""
        config = self.get(key)
        base_url = config.get("base_url")
        if not base_url:
            logger.debug("No base_url configured for %s; skipping readiness check.", key)
            return
        models_url = base_url.rstrip("/") + "/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(models_url, timeout=5.0)
                if resp.status_code == 200:
                    logger.info("✅ %s is ready at %s", key, models_url)
                    return
                logger.debug("Still waiting for %s: %s -> %s", key, models_url, resp.status_code)
            except httpx.HTTPError as exc:
                logger.debug("Waiting for %s: %s", key, exc)
            time.sleep(interval)
        raise TimeoutError(f"{key} did not become ready within {timeout} seconds (checked {models_url}).")


