# Persona Integration Plan
**Goal:** Integrate 57 client personas into the PDF processing pipeline and knowledge base

---

## Phase 1: Export Personas for Knowledge Base

### Create Persona Export Script
**File:** `enlitens_client_profiles/export_personas_for_kb.py`

**What it does:**
1. Load all 57 personas
2. Extract key data for knowledge base:
   - Target audience segments
   - Pain points and struggles
   - Therapy goals and barriers
   - SEO keywords and local entities
   - Website copy snippets
   - Quotes and narrative voice
3. Create aggregated persona insights:
   - Top 20 pain points across all personas
   - Top 30 SEO keywords
   - Top 20 St. Louis local entities
   - Common therapy goals
   - Age/demographic distribution
4. Export as `persona_insights.json`

**Output format:**
```json
{
  "metadata": {
    "total_personas": 57,
    "generated_date": "2025-11-08",
    "source": "cluster-based generation from 224 intakes"
  },
  "aggregated_insights": {
    "top_pain_points": ["burnout", "executive function struggles", ...],
    "top_therapy_goals": ["improve time management", "reduce anxiety", ...],
    "top_keywords": ["adhd therapy st louis", "autism affirming therapy", ...],
    "top_local_entities": ["Webster Groves", "Kirkwood", "Clayton", ...],
    "age_distribution": {"25-35": 7, "35-45": 10, ...},
    "common_struggles": ["late diagnosis", "masking exhaustion", ...]
  },
  "persona_segments": [
    {
      "segment_id": "cluster-002",
      "persona_name": "The Overwhelmed ADHD Intern",
      "target_audience": "PsyD students, psychology interns, 23-28",
      "pain_points": ["time management", "anxiety", "burnout"],
      "therapy_goals": ["improve organization", "reduce stress"],
      "seo_keywords": ["adhd therapy for students", "psychology intern burnout"],
      "local_entities": ["Central West End", "Washington University"],
      "website_copy_snippet": "Struggling to balance internship and coursework?",
      "quote_struggle": "I'm really struggling with my own ADHD.",
      "quote_hope": "I'm looking for a therapist that can help me manage my adhd."
    },
    ...
  ]
}
```

---

## Phase 2: Modify PDF Processing to Include Personas

### Update `process_multi_agent_corpus.py`

**Changes needed:**

1. **Add persona loading** (line ~150):
```python
# Load persona insights
persona_insights_file = Path("enlitens_client_profiles/persona_insights.json")
if persona_insights_file.exists():
    with open(persona_insights_file) as f:
        persona_insights = json.load(f)
    logger.info(f"✅ Loaded {persona_insights['metadata']['total_personas']} persona insights")
else:
    persona_insights = None
    logger.warning("⚠️  No persona insights found - proceeding without persona data")
```

2. **Pass persona data to agents** (line ~400+):
```python
# When creating extraction team or supervisor
extraction_team = ExtractionTeam(
    ...,
    persona_insights=persona_insights  # NEW
)
```

3. **Merge persona insights into knowledge base** (line ~800+):
```python
# After processing all PDFs, add persona section
if persona_insights:
    knowledge_base["persona_insights"] = persona_insights["aggregated_insights"]
    knowledge_base["persona_segments"] = persona_insights["persona_segments"]
    logger.info("✅ Added persona insights to knowledge base")
```

---

## Phase 3: Use Personas in Content Generation

### Marketing Agent Enhancement
**File:** `src/agents/marketing_agent.py`

**Use personas to:**
- Generate target audience descriptions
- Create pain point messaging
- Write website copy that resonates with specific segments
- Generate SEO keywords based on persona search intent

**Example:**
```python
def generate_marketing_content(self, research_content, persona_insights):
    """Generate marketing content informed by personas."""
    
    # Extract relevant persona segments for this research topic
    relevant_segments = self._match_personas_to_research(research_content, persona_insights)
    
    # Generate pain point messaging
    pain_points = [p for segment in relevant_segments for p in segment['pain_points']]
    
    # Generate website copy using persona language
    copy_snippets = [segment['website_copy_snippet'] for segment in relevant_segments]
    
    # Return marketing content with persona-informed messaging
    ...
```

### SEO Agent Enhancement
**File:** `src/agents/seo_agent.py`

**Use personas to:**
- Generate local SEO keywords (St. Louis neighborhoods)
- Create long-tail keywords based on persona struggles
- Identify content topics that resonate with target audiences

**Example:**
```python
def generate_seo_keywords(self, research_content, persona_insights):
    """Generate SEO keywords informed by personas."""
    
    # Base keywords from research
    research_keywords = self._extract_research_keywords(research_content)
    
    # Persona-informed keywords
    persona_keywords = persona_insights['aggregated_insights']['top_keywords']
    
    # Local entities for local SEO
    local_entities = persona_insights['aggregated_insights']['top_local_entities']
    
    # Combine and rank
    all_keywords = research_keywords + persona_keywords
    local_keywords = [f"{kw} {entity}" for kw in all_keywords[:10] for entity in local_entities[:5]]
    
    return {
        "primary_keywords": all_keywords[:10],
        "local_keywords": local_keywords[:20],
        "long_tail_keywords": self._generate_long_tail(all_keywords, persona_insights)
    }
```

---

## Phase 4: Fine-Tuning Training Pairs

### Create Training Pair Generator
**File:** `enlitens_client_profiles/generate_training_pairs.py`

**Purpose:** Generate training pairs for fine-tuning a model to talk like Liz Wooten

**Data sources:**
1. **Personas** - Client struggles, quotes, therapy goals
2. **Transcripts** - Liz's actual responses to clients
3. **Knowledge Base** - Research-backed content
4. **Website Copy** - Liz's writing style

**Training pair format:**
```json
{
  "instruction": "A client says: 'I'm really struggling with my own ADHD. I can't manage my time and I'm constantly overwhelmed.'",
  "input": "Context: 23-28 year old psychology intern, late-diagnosed ADHD, struggling with internship demands",
  "output": "I hear you. That feeling of being overwhelmed is so common when you're navigating ADHD in a demanding environment like an internship. Many of my clients describe this exact struggle - the time blindness, the executive function challenges, the constant sense of falling behind. What you're experiencing isn't a personal failing; it's your nervous system responding to demands that weren't designed with neurodivergent brains in mind. Let's work together to build systems that actually work for your brain, not against it."
}
```

**Training pair types:**
1. **Client struggle → Liz response** (from personas + transcripts)
2. **Research finding → Liz explanation** (from knowledge base + Liz's style)
3. **Client question → Liz answer** (from intakes + transcripts)
4. **Marketing copy → Liz voice** (from personas + website)

**Generation strategy:**
1. **Extract Liz's voice patterns** from transcripts:
   - Common phrases ("I hear you", "That makes sense", "Let's work together")
   - Tone markers (compassionate, direct, neurodivergent-affirming)
   - Clinical language style (plain language, no jargon)
2. **Match persona struggles** to transcript responses
3. **Generate synthetic pairs** using Gemini/GPT-4 with Liz's voice examples
4. **Validate pairs** for authenticity and tone

**Output:** `training_pairs.jsonl` (1000-5000 pairs)

---

## Implementation Steps

### Step 1: Export Personas (15 min)
```bash
cd /home/antons-gs/enlitens-ai
python -m enlitens_client_profiles.export_personas_for_kb
# Output: enlitens_client_profiles/persona_insights.json
```

### Step 2: Modify PDF Processing (30 min)
- Update `process_multi_agent_corpus.py` to load and merge persona insights
- Test with a small PDF to verify integration

### Step 3: Reprocess Knowledge Base (60-90 min)
```bash
cd /home/antons-gs/enlitens-ai
./start_processing.sh
# Output: enlitens_knowledge_base_YYYYMMDD_HHMMSS.json (with persona insights)
```

### Step 4: Generate Training Pairs (30-60 min)
```bash
python -m enlitens_client_profiles.generate_training_pairs
# Output: training_pairs.jsonl
```

### Step 5: Validate & Review (30 min)
- Review sample training pairs
- Check knowledge base has persona insights
- Verify persona data quality

---

## Expected Outcomes

### Enhanced Knowledge Base
- ✅ Research content (from PDFs)
- ✅ Persona insights (from 57 personas)
- ✅ Target audience data
- ✅ Pain points and therapy goals
- ✅ SEO keywords (research + persona-informed)
- ✅ Local entities (St. Louis neighborhoods)
- ✅ Website copy snippets (persona-informed)

### Training Pairs for Fine-Tuning
- ✅ 1000-5000 high-quality training pairs
- ✅ Liz Wooten's authentic voice
- ✅ Neurodivergent-affirming tone
- ✅ Client-centered language
- ✅ Research-backed content
- ✅ St. Louis regional context

### Use Cases
1. **Fine-tune a model** to respond like Liz Wooten
2. **Generate website content** that resonates with target personas
3. **Create SEO-optimized blog posts** using persona keywords
4. **Write marketing copy** that addresses real client pain points
5. **Develop chatbot responses** in Liz's voice
6. **Create email sequences** for different persona segments

---

## Next Steps

**Immediate (Today):**
1. ✅ Confirm 57 personas are sufficient (YES)
2. ⏳ Create `export_personas_for_kb.py` script
3. ⏳ Run export to generate `persona_insights.json`

**Short-term (This Week):**
4. ⏳ Modify `process_multi_agent_corpus.py` to include personas
5. ⏳ Reprocess knowledge base with persona insights
6. ⏳ Create `generate_training_pairs.py` script
7. ⏳ Generate initial training pairs

**Medium-term (Next Week):**
8. ⏳ Fine-tune a model on training pairs
9. ⏳ Test model responses for Liz's voice
10. ⏳ Iterate on training pairs based on quality

---

## Questions for You

1. **Do you want me to start implementing now?** (I can create the export script and modify the PDF processing)

2. **How many training pairs do you want?** (1000? 5000? More?)

3. **What model do you want to fine-tune?** (Qwen2.5-14B local? Gemini? GPT-4o?)

4. **Do you have more transcript data** I should use for training pairs? (The more Liz voice examples, the better)

5. **Should I prioritize knowledge base integration or training pairs first?**

