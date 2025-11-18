#!/usr/bin/env python
"""Run web intelligence agents outside of the main document pipeline."""

import asyncio
from typing import Dict

from src.agents.community_impact_mapper import CommunityImpactMapper
from src.agents.event_finder_agent import EventFinderAgent
from src.agents.live_local_news_agent import LiveLocalNewsAgent
from src.agents.myth_scraper_agent import MythScraperAgent
from src.agents.policy_monitor_agent import PolicyMonitorAgent
from src.agents.research_update_agent import ResearchUpdateAgent
from src.agents.resource_intake_agent import ResourceIntakeAgent
from src.agents.symptom_trend_tracker_agent import SymptomTrendTrackerAgent


AGENTS = [
    LiveLocalNewsAgent(),
    PolicyMonitorAgent(),
    ResourceIntakeAgent(),
    EventFinderAgent(),
    ResearchUpdateAgent(),
    MythScraperAgent(),
    CommunityImpactMapper(),
    SymptomTrendTrackerAgent(),
]


async def run_agent(agent, context: Dict):
    if not agent.is_initialized:
        await agent.initialize()
    result = await agent.process(context)
    await agent.validate_output(result)
    return agent.name, result


async def main() -> None:
    context = {
        "document_id": "web-snapshot",
        "document_text": "",
        "client_insights": {},
        "st_louis_context": {},
    }
    tasks = [run_agent(agent, context) for agent in AGENTS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item in results:
        if isinstance(item, Exception):
            print(f"Error: {item}")
        else:
            name, payload = item
            print(f"{name}: keys={list(payload.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
