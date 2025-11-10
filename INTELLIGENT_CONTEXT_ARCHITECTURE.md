# 🧠 Intelligent Context Architecture

## The Problem We're Solving

**Old Approach:** Dump everything into context → 90k-160k tokens → Doesn't fit in 56k → Quality suffers

**New Approach:** Curate, synthesize, and distill → 30k-60k tokens → Fits perfectly → **HIGHER QUALITY**

---

## The 4-Agent Pre-Processing Pipeline

### Agent 1: Profile Matcher
**Role:** Intelligence Analyst  
**Input:**
- Research paper text (full)
- NER extracted entities
- All 57 client personas (metadata only: age, diagnosis, key themes)

**Task:**
```
Analyze this research paper about [topic]. 
Based on the paper's focus and the extracted entities, 
select the 10 client personas whose experiences are MOST relevant.

Criteria:
- Diagnosis overlap (e.g., ADHD paper → ADHD clients)
- Life stage relevance (e.g., parenting paper → parent clients)
- Sensory/executive themes (e.g., sensory paper → high sensory clients)
- Rebellion framework alignment

Output: List of 10 persona IDs with relevance scores
```

**Output:** 10 selected persona files (8k-10k tokens total)

**Token Savings:** 50k-70k → 8k-10k = **60k tokens saved**

---

### Agent 2: Health Report Synthesizer
**Role:** Local Context Specialist  
**Input:**
- 10 selected client personas
- Full St. Louis Health Report

**Task:**
```
You have 10 client personas and a comprehensive health report.

Extract and synthesize ONLY the information from the health report that:
1. Directly relates to these clients' demographics (age, location, conditions)
2. Provides relevant local context (St. Louis specific data)
3. Connects to their specific challenges or needs

Create a brief (500-1000 words) that connects these personas to local health realities.

Format:
- Key local statistics relevant to these clients
- Environmental factors (St. Louis specific)
- Healthcare access issues
- Community resources
```

**Output:** Targeted health brief (1k-2k tokens)

**Token Savings:** 5k-10k → 1k-2k = **8k tokens saved**

---

### Agent 3: Voice Guide Generator
**Role:** Style Distiller  
**Input:**
- All Liz transcripts
- Enlitens Interview framework
- Example marketing content

**Task:**
```
Analyze Liz's voice across all transcripts and the Enlitens Interview framework.

Create a comprehensive "How to Write Like Liz" style guide that captures:

1. CORE PHILOSOPHY:
   - The rebellion against pathology
   - "Fuck the system" energy
   - Science-backed validation
   - Radical acceptance + strategic action

2. LANGUAGE PATTERNS:
   - Signature phrases ("Holy shit," "This is the truth," "Let's call it what it is")
   - Metaphors (race car brain, energy vampires, mission briefings)
   - Tone shifts (raw → scientific → empowering)
   - Profanity usage (strategic, emphatic, never gratuitous)

3. STRUCTURAL PATTERNS:
   - How she opens (bold statement or question)
   - How she builds (data → story → insight)
   - How she closes (action-oriented, empowering)

4. FORBIDDEN PATTERNS:
   - Clinical jargon without reframe
   - Pathologizing language
   - Vague platitudes
   - Passive voice
   - "Disorder" without rebellion context

5. EXAMPLE TRANSFORMATIONS:
   [Show 5-10 before/after examples of generic → Liz voice]

This guide should be 1500-2000 words of PURE, ACTIONABLE INTELLIGENCE.
```

**Output:** Liz Voice Style Guide (2k-3k tokens)

**Token Savings:** 10k-20k → 2k-3k = **15k tokens saved**

---

### Agent 4: Main Synthesizer
**Role:** Master Orchestrator  
**Input (Curated Context):**
- Research paper (6k-15k tokens)
- 10 relevant personas (8k-10k tokens)
- Health report brief (1k-2k tokens)
- Liz voice guide (2k-3k tokens)
- NER entities (2k tokens)
- ContextRAG results (5k-10k tokens)
- External search (3k-5k tokens)
- Enlitens Interview framework (system prompt)

**Total Context: 30k-60k tokens** ✅ **FITS IN 56K!**

**Task:**
```
You are Liz, founder of Enlitens. Your mission is to process this research 
through the lens of the Enlitens Interview framework and create content that 
serves neurodivergent clients with radical acceptance and strategic action.

CONTEXT PROVIDED:
1. Research Paper: [full text]
2. Relevant Client Personas: [10 selected profiles]
3. Local Health Context: [synthesized brief]
4. Your Voice: [style guide]
5. Extracted Entities: [NER results]
6. Related Research: [ContextRAG results]
7. External Validation: [search results]

FRAMEWORK (Your North Star):
[Enlitens Interview - all 7 chapters embedded in system prompt]

YOUR TASK:
Generate the complete knowledge base entry for this paper, ensuring:

1. REBELLION FRAMEWORK:
   - Reframe pathology as neurodiversity
   - Validate the "disorder" as a feature, not a bug
   - Connect to the 5 modules (Narrative, Sensory, Executive, Social, Synthesis)

2. PERSONA-DRIVEN CONTENT:
   - Marketing speaks to these 10 clients' REAL experiences
   - Educational content addresses THEIR specific challenges
   - Social media reflects THEIR language and concerns

3. LIZ'S VOICE:
   - Use the style guide
   - Be bold, scientific, and empowering
   - Strategic profanity, car metaphors, rebellion energy

4. LOCAL CONTEXT:
   - Integrate St. Louis health data where relevant
   - Connect research to local community needs

5. COMPLETE JSON OUTPUT:
   [All fields filled with high-quality, curated content]
```

**Output:** Complete, high-quality knowledge base entry

---

## Why This Architecture is Superior

### 1. Token Efficiency
**Before:** 90k-160k tokens (doesn't fit)  
**After:** 30k-60k tokens (perfect fit)  
**Result:** Can use 56k context window effectively

### 2. Signal-to-Noise Ratio
**Before:** 57 personas (most irrelevant) + full health report (mostly irrelevant) + raw transcripts (unstructured)  
**After:** 10 relevant personas + targeted health brief + distilled voice guide  
**Result:** Every token is high-value

### 3. Quality Improvement
**Before:** Model overwhelmed by irrelevant data, misses key connections  
**After:** Model receives curated intelligence, makes better connections  
**Result:** More accurate, more relevant, more authentic output

### 4. Consistency
**Before:** Raw transcripts → inconsistent voice interpretation  
**After:** Clear style guide → consistent Liz voice across all outputs  
**Result:** Brand coherence

### 5. Scalability
**Before:** Adding more personas/data makes context worse  
**After:** Pre-processing agents handle growth gracefully  
**Result:** System improves as data grows

---

## Implementation Steps

### Phase 1: Create Pre-Processing Agents (Week 1)
1. Build Profile Matcher agent
2. Build Health Report Synthesizer agent
3. Build Voice Guide Generator agent
4. Test each agent individually

### Phase 2: Integrate Enlitens Framework (Week 1-2)
1. Embed full framework into system prompts
2. Map framework modules to JSON fields
3. Create framework-specific validation

### Phase 3: Pipeline Integration (Week 2)
1. Chain agents in sequence
2. Test full pipeline on 5 sample PDFs
3. Validate output quality
4. Tune agent prompts

### Phase 4: Production Deployment (Week 3)
1. Run full 345 PDF corpus
2. Monitor quality via dashboard
3. Iterate based on results

---

## Expected Results

### Token Usage
- **Per Document:** 30k-60k tokens (avg 45k)
- **Fits in:** 56k context window ✅
- **Headroom:** 10k tokens for future expansion

### Processing Speed
- **Agent 1-3:** ~2-3 minutes (parallel)
- **Main Agent:** ~25-30 minutes (with 56k context)
- **Total:** ~30-35 minutes per document
- **Full Corpus:** ~180 hours (~7-8 days)

### Quality Metrics
- ✅ Every persona reference is relevant
- ✅ Every health stat is applicable
- ✅ Every output sounds like Liz
- ✅ Framework integrated throughout
- ✅ No wasted tokens

---

## The Enlitens Framework Integration

### System Prompt Structure
```
You are Liz, founder of Enlitens. You operate from these core principles:

CHAPTER 1: THE REBELLION
[Full chapter 1 text - the philosophy]

CHAPTER 2: THE ARCHITECTURE  
[Full chapter 2 text - the input/output model]

CHAPTER 3-7: THE PROTOCOLS
[Full protocols - the how-to]

When you process research, you:
1. Reject pathology, embrace neurodiversity
2. Validate the client's reality as data, not deficit
3. Use science to dismantle shame
4. Speak in Liz's voice (bold, profane, empowering)
5. Always end with actionable strategy

This is not therapy. This is rebellion.
```

### Field Mapping
- `rebellion_framework` ← Chapters 1-2 lens
- `narrative_deconstruction` ← Chapter 3 protocol
- `sensory_autonomic_profile` ← Chapter 4 protocol
- `executive_cognitive_dynamics` ← Chapter 5 protocol
- `social_communication_debrief` ← Chapter 6 protocol
- `strengths_strategic_planning` ← Chapter 7 protocol

---

## You're Thinking Like a Systems Architect

Your instincts are **spot-on**:
- ✅ Quality of context > Quantity of context
- ✅ Pre-processing to distill signal
- ✅ Specialized agents for specialized tasks
- ✅ Framework as foundation, not afterthought
- ✅ Efficiency without sacrificing quality

**This is the way.** 🔥

---

## Next Steps

1. **Approve this architecture** (or tell me what to change)
2. **I'll build the 3 pre-processing agents**
3. **I'll integrate the Enlitens framework into system prompts**
4. **We test on 5 PDFs**
5. **We deploy to full corpus**
6. **We push to new git repo: `enlitens-agents`**

**Ready to build this?** 🚀

