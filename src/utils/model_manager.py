"""Single-model manager using Llama 3.1 8B Instruct for all agents.

This module manages a single local vLLM server that hosts the Llama 3.1 8B
Instruct checkpoint. All agent tiers (context curator, science extraction,
clinical translation, verification) share this model so we do not need any
dynamic switching logic.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier - single model for all agents."""

    UNIFIED = "llama-3.1-8b"  # All agents use this model


class ModelManager:
    """Manages single Llama 3.1 8B model for all agents."""
    
    # Singleton instance
    _instance: Optional[ModelManager] = None
    
    # Model configuration
    MODEL_CONFIG = {
        "name": "Llama 3.1 8B Instruct",
        "port": 8000,
        "base_url": "http://localhost:8000",
        "script": "/home/antons-gs/enlitens-ai/scripts/start_vllm_llama_8b.sh",
        "model_path": "/home/antons-gs/enlitens-ai/models/llama-3.1-8b-instruct",
        "context": "≈58k tokens (fp8 KV cache)",
        "quality": "Llama 3.1 8B",
        "use_case": "All agents (Data, Research, Writer, QA)",
    }
    
    def __init__(self):
        """Initialize ModelManager (use get_instance() instead)."""
        self.model_started = False
        self.startup_timeout = 120  # 2 minutes
        self.health_check_interval = 2  # seconds
        
    @classmethod
    def get_instance(cls) -> ModelManager:
        """Get singleton instance of ModelManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_model_for_agent(self, agent_class_name: str) -> ModelTier:
        """Get the model tier for an agent (always returns UNIFIED).
        
        Args:
            agent_class_name: Name of the agent class (ignored in single-model mode)
            
        Returns:
            ModelTier.UNIFIED
        """
        return ModelTier.UNIFIED
    
    async def ensure_model_loaded(self, model_tier: ModelTier = ModelTier.UNIFIED) -> bool:
        """Ensure Llama 3.1 8B is loaded and ready.
        
        Args:
            model_tier: Ignored in single-model mode
            
        Returns:
            True if model is ready, False if loading failed
        """
        # Start model on first call
        if not self.model_started:
            logger.info("🚀 Starting Llama 3.1 8B (single-model mode)...")
            if not await self._start_model():
                return False
            self.model_started = True
            return True
        
        # Check if model is healthy
        if await self._check_model_health():
            logger.debug(f"✅ {self.MODEL_CONFIG['name']} is healthy")
            return True
        else:
            logger.warning(f"⚠️ {self.MODEL_CONFIG['name']} unhealthy, restarting...")
            await self.shutdown()
            return await self._start_model()
    
    async def _start_model(self) -> bool:
        """Start the Llama 3.1 8B server.
        
        Returns:
            True if server started successfully
        """
        config = self.MODEL_CONFIG
        script_path = Path(config["script"])
        
        if not script_path.exists():
            logger.error(f"❌ Startup script not found: {script_path}")
            return False
        
        logger.info(f"   Model: {config['name']}")
        logger.info(f"   Context: {config['context']}")
        logger.info(f"   Quality: {config['quality']}")
        logger.info(f"   Port: {config['port']}")
        logger.info(f"   Use Case: {config['use_case']}")
        
        try:
            # Run startup script in background
            subprocess.Popen(
                [str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            # Wait for model to be ready
            logger.info(f"⏳ Waiting for {config['name']} to be ready...")
            for i in range(self.startup_timeout // self.health_check_interval):
                if await self._check_model_health():
                    logger.info(f"✅ {config['name']} is ready!")
                    self.model_started = True
                    return True
                await asyncio.sleep(self.health_check_interval)
            
            logger.error(f"❌ {config['name']} failed to become healthy within {self.startup_timeout}s")
            return False
            
        except Exception as exc:
            logger.error(f"❌ Failed to start {config['name']}: {exc}")
            return False
    
    async def _check_model_health(self) -> bool:
        """Check if the model is healthy and responding.
        
        Returns:
            True if model is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.MODEL_CONFIG['base_url']}/v1/models")
                return response.status_code == 200
        except Exception:
            return False
    
    def get_current_model_info(self) -> Optional[dict]:
        """Get information about the current model.
        
        Returns:
            Dict with model info, or None if not started
        """
        if not self.model_started:
            return None
        
        config = self.MODEL_CONFIG
        return {
            "tier": "unified",
            "name": config["name"],
            "model_path": config["model_path"],
            "context": config["context"],
            "quality": config["quality"],
            "use_case": config["use_case"],
            "port": config["port"],
        }
    
    async def shutdown(self) -> None:
        """Shutdown the model manager and vLLM server."""
        logger.info("🛑 Shutting down ModelManager...")
        
        # Kill vLLM process
        try:
            subprocess.run(
                ["pkill", "-f", "vllm.entrypoints"],
                capture_output=True,
                timeout=10,
            )
            await asyncio.sleep(2)
            logger.info("✅ vLLM server stopped")
        except Exception as exc:
            logger.error(f"⚠️ Error during shutdown: {exc}")
        
        self.model_started = False
        logger.info("🛑 ModelManager shutdown complete")


# Convenience function
def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager.get_instance()
