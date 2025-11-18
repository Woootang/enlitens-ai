"""Chain-of-thought prompting utilities for deep reasoning across all agents.

This module provides standardized chain-of-thought (CoT) prompts that enable
LLMs to perform step-by-step reasoning, connect concepts, and synthesize
complex information before generating final outputs.
"""

from typing import Optional


def get_cot_prefix(
    task_description: str,
    *,
    context_description: Optional[str] = None,
    output_format: Optional[str] = None,
    emphasis: str = "relationships",
) -> str:
    """Generate a chain-of-thought prompt prefix for any agent task.
    
    Args:
        task_description: Brief description of what the agent needs to do
        context_description: Optional description of the context/data available
        output_format: Optional description of expected output format
        emphasis: What to emphasize in reasoning ("relationships", "synthesis", 
                 "accuracy", "creativity")
    
    Returns:
        Formatted chain-of-thought prompt prefix
    """
    
    # Base reasoning instructions
    base_instructions = f"""You are tasked with: {task_description}

Before providing your final answer, think through this step-by-step using the following reasoning process:"""
    
    # Emphasis-specific reasoning steps
    emphasis_steps = {
        "relationships": """
1. **Identify Key Concepts**: What are the main concepts, entities, or themes present?
2. **Map Relationships**: How do these concepts relate to each other? What connections exist?
3. **Trace Mechanisms**: What are the underlying mechanisms or causal chains?
4. **Synthesize Patterns**: What patterns or themes emerge from these relationships?
5. **Apply Context**: How do these patterns apply to the specific context or query?""",
        
        "synthesis": """
1. **Gather Information**: What information is available across all sources?
2. **Identify Themes**: What common themes or threads connect this information?
3. **Resolve Conflicts**: Are there any contradictions? How can they be reconciled?
4. **Build Framework**: What is the best way to organize this information?
5. **Create Synthesis**: How can all of this be combined into a coherent whole?""",
        
        "accuracy": """
1. **Verify Facts**: What are the verifiable facts and data points?
2. **Check Sources**: Where does each piece of information come from?
3. **Assess Confidence**: How confident can we be in each claim?
4. **Identify Gaps**: What information is missing or uncertain?
5. **Ensure Precision**: How can we state this most accurately and precisely?""",
        
        "creativity": """
1. **Understand Constraints**: What are the requirements and boundaries?
2. **Explore Possibilities**: What different approaches or angles are possible?
3. **Connect Unexpectedly**: What unusual or novel connections can be made?
4. **Evaluate Options**: Which approaches best serve the goal?
5. **Craft Solution**: How can we combine the best elements creatively?""",
    }
    
    reasoning_steps = emphasis_steps.get(emphasis, emphasis_steps["relationships"])
    
    # Add context description if provided
    context_section = ""
    if context_description:
        context_section = f"\n\n**Available Context**: {context_description}"
    
    # Add output format if provided
    output_section = ""
    if output_format:
        output_section = f"\n\n**Expected Output Format**: {output_format}"
    
    # Combine all sections
    full_prompt = f"""{base_instructions}
{reasoning_steps}
{context_section}
{output_section}

**Instructions**:
- Show your reasoning process explicitly for each step
- Connect concepts and identify relationships (don't just list facts)
- Synthesize information across sources
- Be thorough but concise in your reasoning
- After completing your reasoning, provide your final answer

Begin your response with your step-by-step reasoning, then provide your final answer.
"""
    
    return full_prompt.strip()


def get_data_agent_cot_prompt(
    query: str,
    data_description: str,
    output_requirements: str,
) -> str:
    """Specialized CoT prompt for data source agents.
    
    Data agents need to understand the RELATIONSHIP between queries and data,
    not just perform keyword matching.
    
    Args:
        query: The query from the orchestrator
        data_description: Description of the data this agent holds
        output_requirements: What the agent should return
    
    Returns:
        Formatted CoT prompt for data agents
    """
    return get_cot_prefix(
        task_description=f"Extract and synthesize information from {data_description} to answer this query: {query}",
        context_description=data_description,
        output_format=output_requirements,
        emphasis="relationships",
    )


def get_research_agent_cot_prompt(
    task: str,
    available_context: str,
    output_schema: str,
) -> str:
    """Specialized CoT prompt for research agents.
    
    Research agents need deep synthesis and mechanism understanding.
    
    Args:
        task: The research task to perform
        available_context: Description of available context
        output_schema: Expected output structure
    
    Returns:
        Formatted CoT prompt for research agents
    """
    return get_cot_prefix(
        task_description=task,
        context_description=available_context,
        output_format=output_schema,
        emphasis="synthesis",
    )


def get_writer_agent_cot_prompt(
    writing_task: str,
    content_requirements: str,
    voice_guidelines: str,
) -> str:
    """Specialized CoT prompt for final writer agents.
    
    Writer agents need accuracy, creativity, and voice consistency.
    
    Args:
        writing_task: The writing task to perform
        content_requirements: What content must be included
        voice_guidelines: Voice and style requirements
    
    Returns:
        Formatted CoT prompt for writer agents
    """
    base_cot = get_cot_prefix(
        task_description=writing_task,
        context_description=content_requirements,
        output_format="Final polished content",
        emphasis="creativity",
    )
    
    # Add voice-specific reasoning
    voice_reasoning = f"""

**Voice & Style Reasoning**:
Before writing, also consider:
1. **Voice Alignment**: How does this content align with the voice guidelines?
2. **Tone Calibration**: What tone is appropriate for this audience and topic?
3. **Language Patterns**: What specific language patterns should be used or avoided?
4. **Authenticity Check**: Does this sound authentic and natural?

**Voice Guidelines**: {voice_guidelines}
"""
    
    return base_cot + voice_reasoning


def get_qa_agent_cot_prompt(
    content_to_verify: str,
    verification_criteria: str,
) -> str:
    """Specialized CoT prompt for QA/verification agents.
    
    QA agents need extreme accuracy and attention to detail.
    
    Args:
        content_to_verify: The content being verified
        verification_criteria: What to check for
    
    Returns:
        Formatted CoT prompt for QA agents
    """
    return get_cot_prefix(
        task_description=f"Verify this content against these criteria: {verification_criteria}",
        context_description=f"Content to verify: {content_to_verify[:500]}...",
        output_format="Verification result with specific issues identified",
        emphasis="accuracy",
    )


# Example usage strings for documentation
EXAMPLE_DATA_AGENT_USAGE = """
# Example: PersonaDataAgent
query = "Find personas with ADHD and sensory processing challenges"
data_desc = "57 client personas with health conditions, demographics, and goals"
output_req = "JSON with selected persona IDs and justifications"

prompt = get_data_agent_cot_prompt(query, data_desc, output_req)
"""

EXAMPLE_RESEARCH_AGENT_USAGE = """
# Example: ClinicalSynthesisAgent
task = "Synthesize clinical interventions from research paper for selected personas"
context = "Research paper (15k tokens), 5 personas, health context, voice guide"
schema = "JSON with interventions, mechanisms, and persona mappings"

prompt = get_research_agent_cot_prompt(task, context, schema)
"""

EXAMPLE_WRITER_AGENT_USAGE = """
# Example: BlogContentAgent
task = "Write a 2000-word blog post on ADHD and sensory processing"
requirements = "Include research findings, clinical interventions, personal stories"
voice = "Liz's authentic, science-backed, non-cliché therapeutic voice"

prompt = get_writer_agent_cot_prompt(task, requirements, voice)
"""

EXAMPLE_QA_AGENT_USAGE = """
# Example: ValidationAgent
content = "Draft blog post (3000 tokens)"
criteria = "Clinical accuracy, voice consistency, SEO optimization, no therapy clichés"

prompt = get_qa_agent_cot_prompt(content, criteria)
"""

