# Immediate Fixes for Critical Issues

## 🚨 Priority 1: Fix Foreman AI (Can Fix NOW)

### Problem
Foreman AI using qwen3:32b can't run properly alongside main processing (both need ~18GB, you only have 25GB total).

### Quick Solution: Use Groq Free API (Recommended)

**Setup Steps**:
1. Sign up at https://console.groq.com (free, instant, no credit card)
2. Get API key from dashboard
3. Install: `pip install groq`
4. Set environment variable: `export GROQ_API_KEY="your_key_here"`

**Code Change** in `monitoring_server_enhanced.py`:

```python
# Add at top
import os
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Modify ForemanAI class
class ForemanAI:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.use_groq = GROQ_AVAILABLE and self.groq_api_key

        if self.use_groq:
            self.groq_client = Groq(api_key=self.groq_api_key)
            logger.info("🚀 Foreman AI using Groq API (llama-3.1-70b)")
        else:
            self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.client = httpx.AsyncClient(timeout=30.0)
            logger.info("🏠 Foreman AI using local Ollama")

    async def analyze_query(self, query: str, context: Dict[str, Any]) -> str:
        status = processing_state.get_current_status()

        context_prompt = f"""You are the Foreman AI for Enlitens AI knowledge extraction system.

Current Status:
- Document: {status['current_document'] or 'None'}
- Progress: {status['progress_percentage']:.1f}% ({status['documents_processed']}/{status['total_documents']})
- Time on Doc: {status['time_on_document_seconds']}s
- Active Agents: {', '.join(status['active_agents']) or 'None'}
- Recent Errors: {len(status['recent_errors'])}

You are monitoring the multi-agent pipeline. Provide helpful, concise analysis."""

        if self.use_groq:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-70b-versatile",  # Free tier
                    messages=[
                        {"role": "system", "content": context_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_tokens=500,
                    top_p=0.9
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq API error: {e}")
                return f"⚠️ Groq API error: {str(e)}"
        else:
            # Fallback to Ollama (existing code)
            try:
                response = await self.client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "llama3.2:3b",  # Use lightweight model
                        "prompt": f"{context_prompt}\n\nUser: {query}\n\nForeman:",
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 500}
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "No response from Ollama")
                return f"❌ Ollama error: {response.status_code}"
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                return f"❌ Ollama unavailable: {str(e)}"
```

**Benefits**:
- ✅ FREE forever (Groq's free tier is generous)
- ✅ FAST (~300 tokens/second - instant responses)
- ✅ HIGH QUALITY (Llama 3.1 70B beats qwen3:32b)
- ✅ NO VRAM USAGE (all cloud-based)
- ✅ Fallback to local if API fails

---

## 🚨 Priority 2: Reduce Hallucinations (Improve NOW)

### Problem
LLM inventing citations that don't exist in source documents (7 validation errors per attempt).

### Immediate Fix: Better Prompting

**Change 1: Add Anti-Hallucination Instructions to System Prompt**

In your agent prompts, add this BEFORE any other instructions:

```python
ANTI_HALLUCINATION_PREFIX = """CRITICAL INSTRUCTIONS - READ CAREFULLY:

You MUST follow these rules strictly:

1. ONLY quote text that appears VERBATIM in the source document
2. NEVER paraphrase or summarize when creating citations
3. NEVER create statistics or quotes from your general knowledge
4. If you cannot find an exact quote, leave the citation field EMPTY
5. Copy the exact wording character-for-character from the source
6. Include page/section numbers only if you see them in the source

HALLUCINATION CHECK:
Before submitting your response, verify EACH citation:
- Does this exact quote appear in the source? YES/NO
- If NO, remove it immediately

Your job is to EXTRACT existing content, NOT create new content."""
```

**Change 2: Two-Stage Citation Process**

In `src/synthesis/ollama_client.py`, create a new method:

```python
async def extract_citations_two_stage(
    self,
    document_text: str,
    model: str = "qwen3:32b",
    num_citations: int = 5
) -> List[str]:
    """
    Stage 1: Extract ONLY verbatim quotes from document.
    This prevents hallucination by separating extraction from formatting.
    """

    stage1_prompt = f"""Your ONLY job is to find {num_citations} important statistical claims or research findings in this document.

RULES:
1. Copy the text EXACTLY as it appears - word for word
2. Include the surrounding context (1-2 sentences)
3. DO NOT paraphrase or change any words
4. DO NOT add information from your knowledge
5. If you can't find {num_citations} quotes, return fewer

Document:
{document_text[:15000]}  # Limit to avoid context overflow

Output format (JSON array of strings):
[
  "The exact first quote from the document...",
  "The exact second quote from the document...",
  ...
]"""

    response = await self.generate_response(
        prompt=stage1_prompt,
        model=model,
        temperature=0.1,  # Low temperature for extraction
        system_message="You are a precise text extraction tool. You only copy exact quotes."
    )

    try:
        quotes = json.loads(response)
        logger.info(f"✅ Extracted {len(quotes)} verbatim quotes")
        return quotes
    except:
        logger.error("Failed to parse quotes")
        return []
```

**Change 3: Improve Citation Validator**

Make your Pydantic validator more helpful:

```python
from fuzzywuzzy import fuzz  # pip install python-Levenshtein fuzzywuzzy

@field_validator('citation')
def validate_citation_in_source(cls, citation: Dict[str, Any], info: ValidationInfo) -> Dict[str, Any]:
    """Enhanced citation validator with fuzzy matching and helpful errors."""

    if not info.context or 'document_text' not in info.context:
        logger.warning("⚠️ No document_text in context for citation validation")
        return citation

    document_text = info.context['document_text']
    quote = citation.get('quote', '')

    if not quote or len(quote) < 10:
        raise ValueError("Citation quote is too short (minimum 10 characters)")

    # Exact match first
    if quote in document_text:
        logger.debug(f"✅ Exact match found for citation")
        return citation

    # Try fuzzy match (handle minor variations)
    quote_lower = quote.lower().strip()
    best_ratio = 0
    best_match = ""

    # Search through document in sliding windows
    words = document_text.split()
    quote_word_count = len(quote.split())

    for i in range(len(words) - quote_word_count + 1):
        window = ' '.join(words[i:i + quote_word_count + 5])
        ratio = fuzz.partial_ratio(quote_lower, window.lower())
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window[:100]

    if best_ratio >= 85:  # 85% similarity threshold
        logger.warning(f"⚠️ Fuzzy match ({best_ratio}%) for citation - accepting")
        return citation

    # Provide helpful error with context
    error_msg = f"""HALLUCINATION DETECTED: Citation not found in source text.

Quote attempt: '{quote[:100]}...'
Best match found: '{best_match}...' (similarity: {best_ratio}%)

INSTRUCTIONS FOR LLM:
- You must copy text EXACTLY as it appears in the source
- Do not paraphrase or modify quotes
- Do not use your general knowledge
- If unsure, search the document again or omit the citation"""

    raise ValueError(error_msg)
```

**Change 4: Update Retry Logic with Better Feedback**

In `ollama_client.py`, when retry fails, feed error back:

```python
async def generate_structured_response_with_feedback(
    self,
    prompt: str,
    response_model: Type[BaseModel],
    document_text: str,  # Pass document text
    ...
) -> Optional[BaseModel]:
    """Generate structured response with validation error feedback loop."""

    errors_from_previous_attempts = []

    for attempt in range(1, max_retries + 1):
        # Add previous errors to prompt
        if errors_from_previous_attempts:
            error_feedback = "\n\nPREVIOUS ATTEMPT ERRORS:\n"
            for err in errors_from_previous_attempts:
                error_feedback += f"- {err}\n"
            error_feedback += "\n🔧 FIX THESE ERRORS: Extract exact quotes from source only!\n"
            enhanced_prompt = prompt + error_feedback
        else:
            enhanced_prompt = prompt

        response = await self.generate_response(
            prompt=enhanced_prompt,
            model=model,
            temperature=temperature,
            system_message=ANTI_HALLUCINATION_PREFIX  # Add prefix
        )

        try:
            # Parse and validate
            validated = response_model.model_validate(
                parsed_data,
                context={'document_text': document_text}
            )
            return validated
        except ValidationError as e:
            # Extract specific errors for feedback
            for error in e.errors():
                if 'HALLUCINATION DETECTED' in str(error):
                    field = error.get('loc', ['unknown'])
                    msg = error.get('msg', '')
                    errors_from_previous_attempts.append(
                        f"Field {field}: {msg[:200]}"
                    )

            logger.warning(f"Attempt {attempt} failed, retrying with error feedback")
            temperature += 0.1  # Increase randomness

    logger.error("All attempts failed even with error feedback")
    return None
```

---

## 🚨 Priority 3: Speed Up Processing

### Quick Win: Enable Ollama Parallelization

In your main processing loop, allow multiple documents simultaneously:

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def process_corpus_parallel(
    pdf_files: List[Path],
    max_concurrent: int = 2  # Process 2 docs at once
):
    """Process multiple documents concurrently."""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(pdf_file):
        async with semaphore:
            return await process_document(pdf_file)

    tasks = [process_with_limit(pdf) for pdf in pdf_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results

# Run it
asyncio.run(process_corpus_parallel(pdf_files))
```

**Expected Impact**: 2x speedup (98 hours → 49 hours)

---

## 📊 Priority 4: Enhance Monitoring UI with Brand Colors

### Quick Visual Improvements

Update `monitoring_ui/styles.css` with Enlitens brand:

```css
:root {
  /* Use Enlitens brand colors */
  --primary: #C105F5;           /* Purple */
  --primary-dark: #9A0BBF;
  --secondary: #FF4502;          /* Orange */
  --accent: #05F2C7;             /* Cyan */
  --highlight: #FFF700;          /* Yellow */
  --success: #7CFC00;            /* Lime green */

  /* Update existing */
  --bg-gradient: linear-gradient(135deg, #C105F5 0%, #FF4502 100%);
}

/* Retro/Pixel font for headers */
.logo, .nav-item, h2, h3 {
  font-family: 'Bitbybit', 'Cute Bubble', monospace !important;
  letter-spacing: 2px;
}

/* Playful animations */
@keyframes bounce-subtle {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

.stat-card:hover {
  animation: bounce-subtle 0.6s ease;
  transform: scale(1.02);
  border-left-color: var(--accent);
}

/* Rebellious accents */
.view-header h2::before {
  content: "⚡ ";
  color: var(--highlight);
}

.processing-status {
  background: linear-gradient(135deg,
    rgba(193, 5, 245, 0.1) 0%,
    rgba(255, 69, 2, 0.1) 100%
  );
  border: 2px solid var(--accent);
  border-radius: 15px;
}

/* Foreman messages - make them stand out */
.message.foreman .message-content {
  background: linear-gradient(135deg,
    rgba(193, 5, 245, 0.2) 0%,
    rgba(5, 242, 199, 0.2) 100%
  );
  border-left: 4px solid var(--accent);
}

/* Error styling - orange instead of red */
.log-entry.ERROR {
  border-left-color: var(--secondary);
  background: rgba(255, 69, 2, 0.05);
}

/* Quality indicators */
.quality-excellent {
  color: var(--success);
  text-shadow: 0 0 10px var(--success);
}
.quality-good {
  color: var(--accent);
  text-shadow: 0 0 10px var(--accent);
}
.quality-fair {
  color: var(--highlight);
  text-shadow: 0 0 10px var(--highlight);
}
.quality-poor {
  color: var(--secondary);
  text-shadow: 0 0 10px var(--secondary);
}
```

---

## 🎯 Summary: Do These NOW

### Immediate Action Checklist

**Today (< 1 hour)**:
- [ ] Sign up for Groq API (5 min)
- [ ] Update Foreman AI to use Groq (15 min)
- [ ] Add anti-hallucination prefix to prompts (10 min)
- [ ] Update CSS with brand colors (15 min)
- [ ] Test Foreman AI responses (10 min)

**This Week**:
- [ ] Implement two-stage citation extraction (2 hours)
- [ ] Enhance citation validator with fuzzy matching (1 hour)
- [ ] Add error feedback loop to retries (1 hour)
- [ ] Enable parallel processing for 2 documents (1 hour)
- [ ] Add Foreman proactive alerts (2 hours)

**Expected Improvements**:
- Foreman AI: 🐌 Broken → ⚡ Lightning fast (Groq)
- Hallucinations: ❌ 70% failure → ✅ <20% failure (better prompting)
- Processing: 🐢 98 hours → 🏃 49 hours (parallel)
- UI: 😐 Functional → 🎨 Brand-aligned & beautiful

---

## 📚 Next Steps After Immediate Fixes

Once these are working:

1. **Research Deep Solutions**: Use `DEEP_RESEARCH_PROMPT.md` for comprehensive improvements
2. **Implement RAG**: Vector database for better citation tracking
3. **Migrate to LangGraph**: Modern multi-agent orchestration
4. **Advanced UI**: Add visualizations, charts, interactive elements
5. **Observability**: Add OpenTelemetry tracing

---

## 🆘 If Something Breaks

**Foreman AI Issues**:
- Check: `echo $GROQ_API_KEY`
- Test: `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`
- Logs: Check monitoring server console output

**Hallucination Still High**:
- Verify anti-hallucination prefix is actually in prompts
- Check that document_text is being passed to validators
- Lower temperature to 0.1 for extraction phase

**UI Not Updating**:
- Clear browser cache
- Check WebSocket connection in browser devtools
- Restart monitoring server

**Need Help**:
- Check logs in `enlitens_processing.log`
- Use Foreman AI to ask "what's wrong?"
- Review `DEEP_RESEARCH_PROMPT.md` for long-term solutions
