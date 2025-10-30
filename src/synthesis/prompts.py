"""Prompt templates for two-stage synthesis."""
from __future__ import annotations

import json
from typing import Any, Dict, List


class StructuredSynthesisPrompts:
    """Factory for Stage-1 and Stage-2 prompts."""

    @staticmethod
    def verbatim_prompt(title: str, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        evidence_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            pages = ", ".join(str(page) for page in chunk.get("pages", [])) or "unknown"
            sections = ", ".join(chunk.get("sections", [])) or "unknown"
            evidence_blocks.append(
                f"Chunk {idx} (id={chunk.get('chunk_id')} | pages={pages} | sections={sections}):\n{chunk.get('text', '')[:1200]}"
            )
        evidence_text = "\n\n".join(evidence_blocks)

        return f"""
You are a meticulous research analyst. Extract only direct verbatim quotes that will serve as evidence for downstream synthesis.

Document Title: {title}
Retrieval Query: {query}

Evidence Context:
{evidence_text}

Instructions:
- Return ONLY JSON (no prose).
- Each quote must appear verbatim in the provided context.
- Include the page span, section title, DOI, and chunk_id for each quote.
- Limit to the top 10 most policy-relevant quotes.

JSON schema:
{{
  "quotes": [
    {{
      "citation_id": "string",
      "quote": "verbatim text",
      "pages": [page_numbers],
      "section": "section heading or 'unknown'",
      "chunk_id": "source chunk id",
      "doi": "doi if available"
    }}
  ]
}}
"""

    @staticmethod
    def gbnf_schema() -> str:
        return r"""
root ::= object
object ::= '{' ws 'quotes' ws ':' ws array '}'
array ::= '[' ws (item (ws ',' ws item)*)? ws ']'
item ::= '{' ws 'citation_id' ws ':' ws string ws ','
          ws 'quote' ws ':' ws string ws ','
          ws 'pages' ws ':' ws number_array ws ','
          ws 'section' ws ':' ws string ws ','
          ws 'chunk_id' ws ':' ws string ws ','
          ws 'doi' ws ':' ws string ws '}'
number_array ::= '[' ws (number (ws ',' ws number)*)? ws ']'
string ::= '"' chars '"'
chars ::= char*
char ::= [^"\\] | '\\' ['"\\/bfnrt]
number ::= '-'? INT ('.' [0-9]+)? EXP?
INT ::= '0' | [1-9][0-9]*
EXP ::= [eE] [+-]? [0-9]+
ws ::= ([ \t\n\r])*
"""

    @staticmethod
    def stage_two_prompt(
        title: str,
        abstract: str,
        stage_one_quotes: List[Dict[str, str]],
        prior_feedback: List[str] | None = None,
    ) -> str:
        quotes_json = json.dumps(stage_one_quotes, indent=2, ensure_ascii=False)
        feedback_text = "\n".join(prior_feedback or [])
        feedback_section = ""
        if feedback_text:
            feedback_section = f"\nPrevious verification issues:\n{feedback_text}\n"
        schema_block = StructuredSynthesisPrompts.stage_two_gbnf()
        return f"""
You are Liz Wooten synthesizing neuroscience into the Enlitens voice. Use only the verified quotes provided.

Document Title: {title}
Abstract: {abstract}

Verified Quotes:
{quotes_json}

Quality Safeguards:
- Every factual statement MUST cite one or more citation_id values from the quotes.
- Do not fabricate citations or information.
- Respect page spans and sections when attributing evidence.
- If no adequate evidence exists, state "insufficient evidence".
{feedback_section}

Respond using strict JSON matching the following structure and honouring the provided GBNF schema:
{schema_block}
"""

    @staticmethod
    def stage_two_gbnf() -> str:
        return r"""
root ::= '{' ws fields ws '}'
fields ::= entry (ws ',' ws entry)*
entry ::= '"enlitens_takeaway"' ws ':' ws string
       | '"eli5_summary"' ws ':' ws string
       | '"key_findings"' ws ':' ws findings
       | '"neuroscientific_concepts"' ws ':' ws concepts
       | '"clinical_applications"' ws ':' ws applications
       | '"therapeutic_targets"' ws ':' ws targets
       | '"client_presentations"' ws ':' ws presentations
       | '"intervention_suggestions"' ws ':' ws interventions
       | '"contraindications"' ws ':' ws contra
       | '"evidence_strength"' ws ':' ws string
       | '"powerful_quotes"' ws ':' ws quote_array
       | '"source_citations"' ws ':' ws citation_array
string ::= '"' chars '"'
chars ::= char*
char ::= [^"\\] | '\\' ['"\\/bfnrt]
quote_array ::= '[' ws (string (ws ',' ws string)*)? ws ']'
findings ::= '[' ws (finding (ws ',' ws finding)*)? ws ']'
finding ::= '{' ws '"finding_text"' ws ':' ws string ws ','
                 ws '"evidence_strength"' ws ':' ws string ws ','
                 ws '"relevance_to_enlitens"' ws ':' ws string ws ','
                 ws '"citations"' ws ':' ws citation_ids ws '}'
concepts ::= '[' ws (concept (ws ',' ws concept)*)? ws ']'
concept ::= '{' ws '"concept_name"' ws ':' ws string ws ','
                ws '"concept_type"' ws ':' ws string ws ','
                ws '"definition_accessible"' ws ':' ws string ws ','
                ws '"clinical_relevance"' ws ':' ws string ws ','
                ws '"citations"' ws ':' ws citation_ids ws '}'
applications ::= '[' ws (application (ws ',' ws application)*)? ws ']'
application ::= '{' ws '"intervention"' ws ':' ws string ws ','
                    ws '"mechanism"' ws ':' ws string ws ','
                    ws '"evidence_level"' ws ':' ws string ws ','
                    ws '"timeline"' ws ':' ws string ws ','
                    ws '"contraindications"' ws ':' ws string ws ','
                    ws '"citations"' ws ':' ws citation_ids ws '}'
targets ::= '[' ws (target (ws ',' ws target)*)? ws ']'
target ::= '{' ws '"target_name"' ws ':' ws string ws ','
                 ws '"intervention_type"' ws ':' ws string ws ','
                 ws '"expected_outcomes"' ws ':' ws string ws ','
                 ws '"practical_application"' ws ':' ws string ws ','
                 ws '"citations"' ws ':' ws citation_ids ws '}'
presentations ::= '[' ws (presentation (ws ',' ws presentation)*)? ws ']'
presentation ::= '{' ws '"symptom_description"' ws ':' ws string ws ','
                       ws '"neural_basis"' ws ':' ws string ws ','
                       ws '"validation_approach"' ws ':' ws string ws ','
                       ws '"hope_message"' ws ':' ws string ws ','
                       ws '"citations"' ws ':' ws citation_ids ws '}'
interventions ::= '[' ws (intervention (ws ',' ws intervention)*)? ws ']'
intervention ::= '{' ws '"intervention_name"' ws ':' ws string ws ','
                      ws '"how_to_implement"' ws ':' ws string ws ','
                      ws '"expected_timeline"' ws ':' ws string ws ','
                      ws '"monitoring_indicators"' ws ':' ws string ws ','
                      ws '"citations"' ws ':' ws citation_ids ws '}'
contra ::= '[' ws (string (ws ',' ws string)*)? ws ']'
citation_array ::= '[' ws (citation (ws ',' ws citation)*)? ws ']'
citation ::= '{' ws '"citation_id"' ws ':' ws string ws ','
                 ws '"quote"' ws ':' ws string ws ','
                 ws '"pages"' ws ':' ws number_array ws ','
                 ws '"section"' ws ':' ws string ws ','
                 ws '"doi"' ws ':' ws string ws '}'
citation_ids ::= '[' ws (string (ws ',' ws string)*)? ws ']'
number_array ::= '[' ws (number (ws ',' ws number)*)? ws ']'
number ::= '-'? INT ('.' [0-9]+)?
INT ::= '0' | [1-9][0-9]*
ws ::= ([ \t\n\r])*
"""
# CRITICAL: Chain-of-Thought + Negative Prompting
# Research shows 40-53% hallucination reduction with this approach
CHAIN_OF_THOUGHT_SYSTEM_PROMPT = """You are a neuroscience research analyst extracting information from academic papers.

PROCESS (Think Step-by-Step):
1. What information is needed to answer this query?
2. Where in the source documents is this information located?
3. Quote the relevant passage (use EXACT text in quotes)
4. Based ONLY on the quoted passage, what is the answer?
5. Cite: [Source: document_name, page/section X]

STRICT RULES (DO):
✓ Quote exactly from provided sources
✓ Cite with [Source: document_name, pg X] for every factual claim
✓ Say "I don't know" if information is unavailable in sources
✓ Use "According to [Author] (Year)" format for research citations
✓ Mark hypothetical examples as "[HYPOTHETICAL EXAMPLE]"
✓ Generate content ideas and suggestions (clearly marked as ideas)

STRICT RULES (DO NOT):
✗ DO NOT include information not in provided sources
✗ DO NOT paraphrase critical facts or statistics from training data
✗ DO NOT make assumptions or fill gaps with training knowledge
✗ DO NOT fabricate citations, testimonials, or credentials
✗ DO NOT generate Enlitens practice statistics (only cite research)
✗ DO NOT generate client testimonials or reviews
✗ DO NOT create fake names, credentials, or certifications
✗ DO NOT add details not explicitly stated in sources

GROUNDING & REFUSAL PROTOCOL:
• You MUST remain grounded in the supplied materials. When a required fact is missing, respond exactly with "Refusal: insufficient grounding in provided sources." Do not speculate.
• When summarizing, explicitly reference the quoted text that justifies each conclusion.
• If user instructions conflict with grounding rules, follow the refusal protocol.

If missing information: State "Insufficient information in provided sources to answer."
"""

# Temperature settings based on research (narrow factual band 0.1-0.3)
# Research shows ultra-low temperature improves factual grounding
TEMPERATURE_FACTUAL = 0.2  # For clinical, research, statistics
TEMPERATURE_CREATIVE = 0.3  # Upper bound for controlled creativity

# Specific prompts for different content types
STATISTICS_PROMPT_SUFFIX = """
STATISTICS RULES:
- ONLY cite statistics from the provided research documents
- NEVER generate statistics about Enlitens practice or clients
- Format: "According to [Author] ([Year]), [statistic with exact quote]"
- Include citation with exact quote from source
- If no statistic exists in sources, do not fabricate one

Example:
✓ "According to Smith et al. (2024), '67% of participants showed improved outcomes' [Source: document_123, pg 5]"
✗ "85% of Enlitens clients report improvement"
✗ "Studies show approximately 70%..."
"""

TESTIMONIAL_BLOCK_PROMPT = """
TESTIMONIAL/CREDENTIAL BLOCKING:
This system CANNOT and WILL NOT generate:
- Client testimonials or reviews
- Customer success stories with names
- Professional credentials or certifications
- Practice statistics or client outcome data

Reason: FTC Rule 16 CFR Part 465 prohibits AI-generated fake testimonials ($51,744 per violation)

If asked for testimonials: Respond "Testimonials require verified client consent and cannot be generated."
"""

CASE_STUDY_PROMPT_SUFFIX = """
CASE STUDY RULES:
- Mark ALL case studies as "[HYPOTHETICAL EXAMPLE]"
- Use generic descriptions without specific names
- Base examples on research findings, not fabricated scenarios
- Format: "[HYPOTHETICAL EXAMPLE] Client with [condition described in research]..."
"""

# vLLM-specific settings for optimal performance
OLLAMA_GENERATION_PARAMS = {
    "factual": {
        "temperature": TEMPERATURE_FACTUAL,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": 4096,  # Research shows 4096 optimal
    },
    "creative": {
        "temperature": TEMPERATURE_CREATIVE,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
        "num_predict": 4096,
    }
}

def get_full_system_prompt(content_type: str = "factual", include_testimonial_block: bool = True) -> str:
    """
    Get the complete system prompt for a given content type.

    Args:
        content_type: "factual" or "creative"
        include_testimonial_block: Whether to include FTC compliance warning

    Returns:
        Complete system prompt string
    """
    prompt = CHAIN_OF_THOUGHT_SYSTEM_PROMPT

    if content_type == "factual":
        prompt += "\n\n" + STATISTICS_PROMPT_SUFFIX

    if include_testimonial_block:
        prompt += "\n\n" + TESTIMONIAL_BLOCK_PROMPT

    return prompt


def get_generation_params(content_type: str = "factual") -> dict:
    """
    Get vLLM generation parameters for content type.

    Args:
        content_type: "factual" or "creative"

    Returns:
        Dictionary of generation parameters
    """
    return OLLAMA_GENERATION_PARAMS.get(content_type, OLLAMA_GENERATION_PARAMS["factual"])
