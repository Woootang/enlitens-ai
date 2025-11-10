"""Supervisor/orchestrator that coordinates persona agents end-to-end."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from .config import ProfilePipelineConfig
from .data_ingestion import IngestionBundle
from .deep_research import DeepResearchAgent, ResearchCache
from .foundation_builder import FoundationBuilderAgent, PersonaFoundation
from .knowledge_keeper import KnowledgeGraphContext, KnowledgeKeeperAgent
from .profile_builder import ClientProfileBuilder
from .schema import ClientProfileDocument

logger = logging.getLogger(__name__)


class PersonaOrchestrator:
    """Run the knowledge keeper → foundation → research → writer chain."""

    def __init__(self, config: ProfilePipelineConfig) -> None:
        self.config = config
        self.knowledge_keeper = KnowledgeKeeperAgent(config)
        self.foundation_builder = FoundationBuilderAgent()
        self.research_agent = DeepResearchAgent(config)
        self.profile_builder = ClientProfileBuilder(config)
        self._knowledge_context: Optional[KnowledgeGraphContext] = None

    def prepare_context(self, bundle: IngestionBundle) -> KnowledgeGraphContext:
        if self._knowledge_context is None:
            self._knowledge_context = self.knowledge_keeper.build_graph(bundle)
        return self._knowledge_context

    def assemble_persona(
        self,
        bundle: IngestionBundle,
    ) -> Tuple[
        ClientProfileDocument,
        KnowledgeGraphContext,
        PersonaFoundation,
        ResearchCache,
    ]:
        knowledge_context = self.prepare_context(bundle)
        foundation = self.foundation_builder.build(bundle, knowledge_context)
        research = self.research_agent.run(foundation)

        if not research.results:
            logger.warning("Deep research returned no results; persona generation halted for this iteration.")
            return None, knowledge_context, foundation, research

        document = self.profile_builder.generate_profile(
            bundle,
            foundation=foundation,
            research=research,
        )
        return document, knowledge_context, foundation, research


