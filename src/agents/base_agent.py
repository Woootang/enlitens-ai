"""
Base Agent class for the Enlitens Multi-Agent System.

This provides the foundation for all specialized agents in the system.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from src.utils.settings import get_settings
from src.utils.chain_of_thought import (
    get_cot_prefix,
    get_data_agent_cot_prompt,
    get_research_agent_cot_prompt,
    get_writer_agent_cot_prompt,
    get_qa_agent_cot_prompt,
)

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agents in the Enlitens multi-agent system.
    """

    def __init__(self, name: str, role: str, model: Optional[str] = None, enable_cot: bool = True):
        settings = get_settings()
        agent_key = self.__class__.__name__
        resolved_model = model or settings.model_for_agent(agent_key)
        if not resolved_model:
            raise ValueError(f"No model configured for agent '{agent_key}'")

        self.name = name
        self.role = role
        self.model = resolved_model
        self.created_at = datetime.now()
        self.is_initialized = False
        self.settings = settings
        self.llm_provider = settings.llm.provider
        self.connection_info = {
            "base_url": settings.llm.endpoint_for(agent_key),
            "provider": settings.llm.provider,
        }
        self.enable_cot = enable_cot  # Chain-of-thought reasoning enabled by default
        logger.info(f"Initializing agent: {name} ({role}) [CoT: {enable_cot}]")

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
            # Work with a copy so downstream modifications don't leak across nodes
            context = dict(context)
            context.setdefault("cache_prefix", self.name)
            context.setdefault("cache_chunk_id", context.get("document_id", "global"))

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

    def _cache_kwargs(self, context: Dict[str, Any], suffix: Optional[str] = None) -> Dict[str, str]:
        """Helper to build cache arguments for prompt-based agents."""
        prefix = context.get("cache_prefix", self.name)
        if suffix:
            prefix = f"{prefix}:{suffix}"
        chunk_id = context.get("cache_chunk_id") or context.get("document_id", "global")
        return {
            "cache_prefix": prefix,
            "cache_chunk_id": chunk_id,
        }
    
    def add_cot_to_prompt(
        self,
        base_prompt: str,
        *,
        task_description: Optional[str] = None,
        context_description: Optional[str] = None,
        output_format: Optional[str] = None,
        emphasis: str = "relationships",
    ) -> str:
        """Add chain-of-thought reasoning to a prompt.
        
        Args:
            base_prompt: The base prompt to enhance
            task_description: Optional task description (defaults to agent role)
            context_description: Description of available context
            output_format: Expected output format
            emphasis: Reasoning emphasis ("relationships", "synthesis", "accuracy", "creativity")
        
        Returns:
            Enhanced prompt with CoT reasoning instructions
        """
        if not self.enable_cot:
            return base_prompt
        
        task = task_description or f"{self.role} task"
        cot_prefix = get_cot_prefix(
            task_description=task,
            context_description=context_description,
            output_format=output_format,
            emphasis=emphasis,
        )
        
        return f"{cot_prefix}\n\n---\n\n{base_prompt}"
