"""
Base Agent class for the Enlitens Multi-Agent System.

This provides the foundation for all specialized agents in the system.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agents in the Enlitens multi-agent system.
    """

    def __init__(self, name: str, role: str, model: str = "qwen3:32b"):
        self.name = name
        self.role = role
        self.model = model
        self.created_at = datetime.now()
        self.is_initialized = False
        logger.info(f"Initializing agent: {name} ({role})")

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent with necessary resources."""
        pass

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return results."""
        pass

    @abstractmethod
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the output quality."""
        pass

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main function with error handling.
        """
        try:
            if not self.is_initialized:
                success = await self.initialize()
                if not success:
                    raise RuntimeError(f"Failed to initialize agent {self.name}")

            logger.info(f"Agent {self.name} starting processing")
            result = await self.process(context)

            if await self.validate_output(result):
                logger.info(f"Agent {self.name} completed successfully")
                return result
            else:
                logger.warning(f"Agent {self.name} output validation failed")
                return {}

        except Exception as e:
            logger.error(f"Agent {self.name} execution failed: {e}")
            return {}

    async def cleanup(self):
        """Clean up agent resources."""
        logger.info(f"Cleaning up agent: {self.name}")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "initialized": self.is_initialized,
            "created_at": self.created_at.isoformat()
        }
