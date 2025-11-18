# 🏗️ Enlitens Multi-Agent Architecture V2
## Specialized Data Agents + Hierarchical Orchestration

**Problem Solved**: Current architecture passes 280k+ tokens to EVERY agent (impossible with 40k context window)

**Solution**: Specialized data agents hold large datasets, return ONLY requested summaries to writer agents

---

## 🎯 Core Principle: Query/Response Pattern

Instead of:
```
❌ EVERY AGENT GETS EVERYTHING (280k tokens)
```

We do:
```
✅ ORCHESTRATOR → queries → DATA AGENT → returns 500-token summary → WRITER AGENT
```

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  HEAD ORCHESTRATOR                           │
│  (Coordinates all teams, minimal context)                    │
└────┬────────────────────────────────────────────────────┬───┘
     │                                                      │
     ├─ Team 1: DATA SOURCES                              │
     ├─ Team 2: RESEARCH EXTRACTION                        │
     ├─ Team 3: CONTENT GENERATION                         │
     └─ Team 4: QUALITY ASSURANCE                          │
```

---

## 🏢 TEAM 1: DATA SOURCE AGENTS
**Supervisor**: `DataSourceSupervisor`  
**Role**: Hold large datasets, respond to targeted queries

### Agent 1.1: `LizVoiceDataAgent`
- **Context**: Full Liz transcripts (454k chars = ~110k tokens)
- **Input Query Examples**:
  - "What's Liz's stance on ADHD medication?"
  - "How does Liz explain executive function to parents?"
  - "Find Liz's phrases about sensory processing"
- **Output**: 300-500 token style guide snippet
- **Model**: Qwen3-14B @ 60k context

### Agent 1.2: `StLouisIntelligenceAgent`
- **Context**: Full St. Louis ethnographic report (682k chars = ~170k tokens)
- **Input Query Examples**:
  - "What are the social stressors in Tower Grove South?"
  - "Explain the 'High School Question' mechanism"
  - "Religious trauma patterns in St. Charles County"
- **Output**: 400-600 token contextual brief
- **Model**: Qwen3-14B @ 60k context (will need 3 sequential loads for full report)

### Agent 1.3: `PersonaDataAgent`
- **Context**: ALL 57 client personas (50-70k tokens)
- **Input Query Examples**:
  - "Top 5 personas for inflammation research"
  - "Personas with ADHD + trauma history"
  - "Age range and diagnoses for selected personas"
- **Output**: 5 persona IDs + 200-token justification
- **Model**: Qwen3-14B @ 60k context

### Agent 1.4: `WebsiteKnowledgeAgent`
- **Context**: Crawled enlitens.com content (310 URLs)
- **Input Query Examples**:
  - "Find pages about sensory processing"
  - "What does Enlitens say about ADOS-2?"
  - "Existing content on executive function"
- **Output**: Relevant URLs + 300-token summary
- **Model**: Qwen3-14B @ 60k context

### Agent 1.5: `AnalyticsAgent`
- **Context**: Google Analytics + Search Console data
- **Input Query Examples**:
  - "Top 10 search queries for ADHD content"
  - "Most visited pages in last 90 days"
  - "Demographics of site visitors"
- **Output**: Top 10 items + traffic stats (200 tokens)
- **Model**: Qwen3-14B @ 40k context (API calls, not large dataset)

---

## 🔬 TEAM 2: RESEARCH EXTRACTION AGENTS
**Supervisor**: `ResearchExtractionSupervisor`  
**Role**: Extract structured data from research paper

### Agent 2.1: `ScienceExtractionAgent`
- **Context**: Research paper (6-15k tokens) + NER entities (2k tokens)
- **Output**: `ResearchContent` schema fields
  - findings, statistics, methodologies, limitations, implications
- **Model**: Qwen3-14B @ 40k context

### Agent 2.2: `ClinicalSynthesisAgent`
- **Context**: Research paper + 5 selected personas (brief) + health context (brief)
- **Output**: `ClinicalContent` schema fields
  - interventions, assessments, outcomes, protocols
- **Model**: Qwen3-14B @ 40k context

### Agent 2.3: `EducationalContentAgent`
- **Context**: Research paper + Liz voice guide (from Agent 1.1)
- **Output**: `EducationalContent` schema fields
  - explanations, examples, analogies, definitions
- **Model**: Qwen3-14B @ 40k context

### Agent 2.4: `RebellionFrameworkAgent`
- **Context**: Research paper + Liz voice guide + persona summaries
- **Output**: `RebellionFramework` schema fields
  - narrative_deconstruction, sensory_profiling, executive_function, etc.
- **Model**: Qwen3-14B @ 40k context

---

## ✍️ TEAM 3: CONTENT GENERATION AGENTS
**Supervisor**: `ContentGenerationSupervisor`  
**Role**: Generate marketing/SEO/blog content

### Agent 3.1: `MarketingSEOAgent`
- **Context**: Research paper + analytics data (from Agent 1.5) + website knowledge (from Agent 1.4)
- **Output**: `MarketingContent` + `SEOContent` schema fields
  - headlines, taglines, keywords, meta descriptions
- **Model**: Qwen3-14B @ 40k context

### Agent 3.2: `BlogContentAgent`
- **Context**: Research paper + Liz voice guide + St. Louis context (if relevant)
- **Output**: `BlogContent` schema fields
  - article_ideas, outlines, talking_points, statistics (with citations)
- **Model**: Qwen3-14B @ 40k context

### Agent 3.3: `SocialMediaAgent`
- **Context**: Research paper + Liz voice guide + analytics trends
- **Output**: `SocialMediaContent` schema fields
  - post_ideas, captions, hashtags, story_ideas
- **Model**: Qwen3-14B @ 40k context

### Agent 3.4: `WebsiteCopyAgent`
- **Context**: Research paper + existing website content (from Agent 1.4)
- **Output**: `WebsiteCopy` schema fields
  - about_sections, feature_descriptions, service_descriptions
- **Model**: Qwen3-14B @ 40k context

---

## ✅ TEAM 4: QUALITY ASSURANCE AGENTS
**Supervisor**: `QualitySupervisor`  
**Role**: Validate outputs, check citations, ensure Liz voice

### Agent 4.1: `OutputVerifierAgent` (EXISTING)
- **Context**: Final JSON output + research paper
- **Output**: Quality scores, validation status
- **Model**: Qwen3-14B @ 40k context

### Agent 4.2: `CitationVerifierAgent` (NEW)
- **Context**: All statistics + full research paper text
- **Output**: Citation validation report
- **Model**: Qwen3-14B @ 40k context

### Agent 4.3: `VoiceConsistencyAgent` (NEW)
- **Context**: All generated content + Liz voice guide
- **Output**: Voice consistency score + flagged phrases
- **Model**: Qwen3-14B @ 40k context

---

## 🔄 Execution Flow

### Phase 1: Data Preparation (Parallel)
```
1. Load research paper → extract text + NER entities
2. Initialize ALL data agents (load their contexts once)
3. Build audience language profile
4. Build topic alignment profile
```

### Phase 2: Data Agent Queries (Sequential, as needed)
```
HEAD ORCHESTRATOR asks:
├─ PersonaDataAgent: "Select top 5 personas for this paper"
├─ LizVoiceDataAgent: "Generate voice guide for these personas"
├─ StLouisIntelligenceAgent: "Relevant context for these personas"
├─ WebsiteKnowledgeAgent: "Existing content on these topics"
└─ AnalyticsAgent: "Top search queries for these topics"

Each returns 200-600 token summary → stored in orchestrator state
```

### Phase 3: Research Extraction (Parallel)
```
ResearchExtractionSupervisor coordinates:
├─ ScienceExtractionAgent (paper + entities)
├─ ClinicalSynthesisAgent (paper + persona summaries + health brief)
├─ EducationalContentAgent (paper + voice guide)
└─ RebellionFrameworkAgent (paper + voice guide + persona summaries)

Each writes to its schema section
```

### Phase 4: Content Generation (Parallel)
```
ContentGenerationSupervisor coordinates:
├─ MarketingSEOAgent (paper + analytics + website knowledge)
├─ BlogContentAgent (paper + voice guide + St. Louis context)
├─ SocialMediaAgent (paper + voice guide + analytics)
└─ WebsiteCopyAgent (paper + existing website content)

Each writes to its schema section
```

### Phase 5: Quality Assurance (Sequential)
```
QualitySupervisor coordinates:
1. CitationVerifierAgent validates all statistics
2. VoiceConsistencyAgent checks Liz voice alignment
3. OutputVerifierAgent final quality check

If any fail → feedback to relevant team for revision
```

### Phase 6: Assembly
```
HEAD ORCHESTRATOR:
1. Collects all schema sections
2. Assembles final EnlitensKnowledgeEntry JSON
3. Saves to knowledge base
```

---

## 📏 Context Window Usage

| Agent Type | Context Size | Model Config |
|------------|--------------|--------------|
| **Data Agents** | 60-170k tokens | Qwen3-14B @ 60k (CPU offload) |
| **Research Agents** | 20-30k tokens | Qwen3-14B @ 40k |
| **Content Agents** | 18-25k tokens | Qwen3-14B @ 40k |
| **QA Agents** | 25-40k tokens | Qwen3-14B @ 40k |

**Total Context Across All Agents**: ~400k tokens  
**But only ONE agent runs at a time**: Max 170k tokens in memory

---

## 🎯 Benefits

1. **Fits in Hardware**: Each agent uses ≤60k context (fits in 24GB GPU + 64GB RAM)
2. **Scalable**: Add new data sources without bloating writer agents
3. **Maintainable**: Each agent has ONE job, ONE dataset
4. **Quality**: Specialized agents = better outputs
5. **Debuggable**: Can test each agent independently
6. **Efficient**: Data agents load once, serve many queries

---

## 🚀 Implementation Priority

### Phase 1: Core Data Agents (CRITICAL)
1. `PersonaDataAgent` - Already partially exists in `ProfileMatcherAgent`
2. `LizVoiceDataAgent` - Extract from `VoiceGuideGeneratorAgent`
3. `StLouisIntelligenceAgent` - NEW, uses health report

### Phase 2: External Data Agents
4. `WebsiteKnowledgeAgent` - NEW, crawl enlitens.com
5. `AnalyticsAgent` - NEW, GA4 + Search Console API

### Phase 3: Refactor Existing Agents
6. Update `ScienceExtractionAgent` to use data agent queries
7. Update `ClinicalSynthesisAgent` to use data agent queries
8. Update other content agents

### Phase 4: New QA Agents
9. `CitationVerifierAgent` - NEW
10. `VoiceConsistencyAgent` - NEW

---

## 📝 Next Steps

1. **Start vLLM with 60k context** (Option 1+2 from earlier)
2. **Create `DataSourceSupervisor` class**
3. **Implement `PersonaDataAgent` (refactor existing code)**
4. **Implement `LizVoiceDataAgent` (refactor existing code)**
5. **Implement `StLouisIntelligenceAgent` (NEW)**
6. **Test single-document flow with new architecture**
7. **Implement remaining agents**

---

## 🔧 Technical Notes

### Data Agent Pattern
```python
class BaseDataAgent(BaseAgent):
    def __init__(self, data_path: str, context_window: int):
        self.data = self._load_data(data_path)
        self.context_window = context_window
        
    async def query(self, query: str, max_tokens: int = 500) -> str:
        """
        Query the data agent's internal knowledge.
        Returns a concise summary (max_tokens).
        """
        prompt = f"""
        You are a specialized data agent with access to:
        {self.data[:self.context_window]}
        
        User query: {query}
        
        Provide a concise, factual response in {max_tokens} tokens or less.
        """
        return await self.llm_client.generate_response(prompt, max_tokens=max_tokens)
```

### Orchestrator Pattern
```python
class HeadOrchestrator:
    def __init__(self):
        self.data_agents = {
            'liz_voice': LizVoiceDataAgent(),
            'stl_intelligence': StLouisIntelligenceAgent(),
            'personas': PersonaDataAgent(),
            'website': WebsiteKnowledgeAgent(),
            'analytics': AnalyticsAgent(),
        }
        self.research_supervisor = ResearchExtractionSupervisor()
        self.content_supervisor = ContentGenerationSupervisor()
        self.qa_supervisor = QualitySupervisor()
        
    async def process_document(self, paper_text: str) -> EnlitensKnowledgeEntry:
        # Phase 1: Gather data summaries
        persona_ids = await self.data_agents['personas'].query(
            f"Select top 5 personas for this paper: {paper_text[:2000]}"
        )
        voice_guide = await self.data_agents['liz_voice'].query(
            f"Generate voice guide for personas: {persona_ids}"
        )
        stl_context = await self.data_agents['stl_intelligence'].query(
            f"Relevant St. Louis context for personas: {persona_ids}"
        )
        
        # Phase 2: Research extraction
        research_results = await self.research_supervisor.process({
            'paper_text': paper_text,
            'persona_summaries': persona_ids,
            'voice_guide': voice_guide,
            'stl_context': stl_context,
        })
        
        # Phase 3: Content generation
        content_results = await self.content_supervisor.process({
            'paper_text': paper_text,
            'voice_guide': voice_guide,
            'research_results': research_results,
        })
        
        # Phase 4: QA
        qa_results = await self.qa_supervisor.validate({
            'research_results': research_results,
            'content_results': content_results,
        })
        
        # Phase 5: Assemble
        return self._assemble_knowledge_entry(
            research_results, content_results, qa_results
        )
```

---

**Status**: DESIGN COMPLETE - Ready for implementation  
**Author**: AI Assistant  
**Date**: 2025-11-12  
**Version**: 2.0

