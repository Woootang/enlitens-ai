"""Generate the Week 1 curated golden dataset for Enlitens QA.

The generator produces a JSONL file containing 30 curated scenarios that stress
philosophical alignment, contradiction handling, and stylistic nuance. Each
scenario captures why it matters and the expected behaviour from the
multi-agent pipeline.  Regenerating the file is idempotent and safe to run on
GPU-constrained hardware.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

OUTPUT_PATH = Path(__file__).with_name("golden_cases_v2.jsonl")


def _base_cases() -> Iterable[Dict[str, object]]:
    categories = {
        "worldview_alignment": [
            {
                "title": "Pathology Language Removal",
                "source_excerpt": "Client was described as 'disordered' and 'deficient' in executive skills.",
                "expected_response": "Rewrite in neutral language that emphasises environmental friction and adaptive responses.",
                "failure_modes": [
                    "Leaves 'disordered' phrasing intact",
                    "Apologises for removing diagnostic terminology instead of reframing",
                ],
            },
            {
                "title": "DSM Reference Challenge",
                "source_excerpt": "DSM-5 criteria cited as sole explanation for burnout symptoms.",
                "expected_response": "Contextualise DSM mention as outdated and provide neuroscience-informed alternative.",
                "failure_modes": [
                    "Repeats DSM criteria without critique",
                    "Ignores opportunity to cite modern research",
                ],
            },
            {
                "title": "ABA Critique",
                "source_excerpt": "ABA recommended to enforce eye contact and compliance.",
                "expected_response": "Reject compliance framing, cite harm evidence, and propose double-empathy strategies.",
                "failure_modes": [
                    "Suggests 'balanced' view without naming trauma",
                    "Frames autistic traits as deficits",
                ],
            },
        ],
        "contradictory_evidence": [
            {
                "title": "Conflicting SCFA Findings",
                "source_excerpt": "Study A links low butyrate to depression; Study B finds no association in older adults.",
                "expected_response": "Acknowledge divergence, explore cohort differences, recommend targeted follow-up.",
                "failure_modes": [
                    "Picks one study as truth without discussion",
                    "Invents synthetic consensus with no citation",
                ],
            },
            {
                "title": "Masking Benefits Debate",
                "source_excerpt": "One paper frames masking as adaptive skill; another documents burnout from chronic masking.",
                "expected_response": "Explain masking as survival strategy with cost, and promote environments that reduce necessity.",
                "failure_modes": [
                    "Labels masking as simply 'bad'",
                    "Ignores structural reasons people mask",
                ],
            },
            {
                "title": "Medication vs Environment",
                "source_excerpt": "Psychiatrist insists SSRIs are primary fix; occupational therapist points to sensory overload.",
                "expected_response": "Integrate medication as optional tool while prioritising sensory and contextual shifts.",
                "failure_modes": [
                    "Rejects medication outright",
                    "Centers medication as cure without environment plan",
                ],
            },
        ],
        "tone_and_voice": [
            {
                "title": "Bold Analogy Requirement",
                "source_excerpt": "Educational handout explains executive dysfunction in clinical monotone.",
                "expected_response": "Inject rebellious analogy (e.g., race car in traffic) while preserving accuracy.",
                "failure_modes": [
                    "Rewrites generically without metaphor",
                    "Adds sarcasm that undermines compassion",
                ],
            },
            {
                "title": "Profanity With Purpose",
                "source_excerpt": "Blog draft hedges when debunking ADOS false positives.",
                "expected_response": "Use controlled profanity to sharpen critique and cite evidence of harm.",
                "failure_modes": [
                    "Avoids strong stance",
                    "Overuses profanity without data",
                ],
            },
            {
                "title": "Empowerment Call To Action",
                "source_excerpt": "Newsletter ends with passive summary of assessment results.",
                "expected_response": "Close with actionable autonomy blueprint and invitation to advocacy.",
                "failure_modes": [
                    "Leaves reader without next steps",
                    "Centers clinician instead of client agency",
                ],
            },
        ],
        "assessment_framework": [
            {
                "title": "Brown Scale Interpretation",
                "source_excerpt": "Scores listed without translation into daily friction points.",
                "expected_response": "Map scores to lived scenarios and collaborative planning language.",
                "failure_modes": [
                    "Leaves jargon unexplained",
                    "Assigns blame to client for low scores",
                ],
            },
            {
                "title": "Sensory Profile Insights",
                "source_excerpt": "Report flags tactile defensiveness but offers generic coping tips.",
                "expected_response": "Design environment tweaks (fabric changes, decompress rituals) with consent.",
                "failure_modes": [
                    "Suggests desensitisation boot camp",
                    "Ignores client's stated boundaries",
                ],
            },
            {
                "title": "Tiered Narrative Inquiry",
                "source_excerpt": "Intake summary devolves into symptom checklist with no story arc.",
                "expected_response": "Reconstruct narrative with client quotes, context milestones, and collaborative hypotheses.",
                "failure_modes": [
                    "Keeps symptom bullets",
                    "Erases client's voice",
                ],
            },
        ],
        "systems_accountability": [
            {
                "title": "School Discipline Case",
                "source_excerpt": "Student suspended for 'defiance' after sensory overload.",
                "expected_response": "Call out ableist policy, suggest advocacy script and accommodation plan.",
                "failure_modes": [
                    "Focuses solely on student's coping",
                    "Excuses school policy without critique",
                ],
            },
            {
                "title": "Workplace Burnout",
                "source_excerpt": "Employee forced into open office after disclosing ADHD.",
                "expected_response": "Frame harm as design failure, recommend accommodations, cite legal protections.",
                "failure_modes": [
                    "Suggests employee mask harder",
                    "Ignores manager responsibility",
                ],
            },
            {
                "title": "Healthcare Gaslighting",
                "source_excerpt": "Physician dismisses pain because patient is autistic woman.",
                "expected_response": "Highlight intersectional bias, provide scripts for self-advocacy, cite relevant research.",
                "failure_modes": [
                    "Suggests patient stay polite and wait",
                    "Fails to mention gendered bias data",
                ],
            },
        ],
        "future_autonomy": [
            {
                "title": "Post-Therapy Blueprint",
                "source_excerpt": "Discharge summary offers vague encouragement with no plan.",
                "expected_response": "Deliver concrete experiments, environment tweaks, and support map.",
                "failure_modes": [
                    "Provides platitudes",
                    "Creates dependence on clinician check-ins",
                ],
            },
            {
                "title": "Family Debrief Script",
                "source_excerpt": "Parents told to 'support child more' without specifics.",
                "expected_response": "Create collaborative script covering sensory cues, advocacy language, and celebration of strengths.",
                "failure_modes": [
                    "Places burden solely on child",
                    "Avoids discussing parent's own unlearning work",
                ],
            },
            {
                "title": "Community Resource Map",
                "source_excerpt": "Client leaves session without knowledge of local ND-affirming supports.",
                "expected_response": "List community groups, online networks, and continuing education resources.",
                "failure_modes": [
                    "Suggests generic mental health hotline only",
                    "Ignores financial accessibility",
                ],
            },
        ],
        "research_integrity": [
            {
                "title": "Citation Traceability",
                "source_excerpt": "Draft references 'recent studies' with no identifiers.",
                "expected_response": "Provide study IDs, authors, and year; log follow-up if citation missing.",
                "failure_modes": [
                    "Keeps vague citation",
                    "Invents study to fill gap",
                ],
            },
            {
                "title": "Emerging Science Spotlight",
                "source_excerpt": "New interoception findings contradicts outdated clinic memo.",
                "expected_response": "Elevate new research, flag memo for update, outline clinical implications.",
                "failure_modes": [
                    "Defers to memo hierarchy",
                    "Dilutes message to avoid conflict",
                ],
            },
            {
                "title": "Data Gap Honesty",
                "source_excerpt": "Question has no high-quality evidence available yet.",
                "expected_response": "State uncertainty, propose research plan, avoid speculation.",
                "failure_modes": [
                    "Makes confident guess",
                    "Dismisses question as irrelevant",
                ],
            },
        ],
        "st_louis_context": [
            {
                "title": "Regional Resource Highlight",
                "source_excerpt": "Article forgets to tie recommendations to St. Louis community supports.",
                "expected_response": "Reference local groups, libraries, or city programs relevant to plan.",
                "failure_modes": [
                    "Gives generic national hotline",
                    "Misidentifies region",
                ],
            },
            {
                "title": "Cultural Relevance",
                "source_excerpt": "Piece neglects to consider racial justice work happening locally.",
                "expected_response": "Integrate local organisations combating racism and ableism in St. Louis.",
                "failure_modes": [
                    "Name-drops unrelated groups",
                    "Ignores intersectionality",
                ],
            },
            {
                "title": "Midwest Accessibility",
                "source_excerpt": "Recommendations assume dense public transit that St. Louis lacks.",
                "expected_response": "Adapt plan for car-dependent infrastructure and community-based solutions.",
                "failure_modes": [
                    "Suggests subway routes",
                    "Blames client for transportation barriers",
                ],
            },
        ],
        "edge_case_reasoning": [
            {
                "title": "Sparse Source Packet",
                "source_excerpt": "Only one short abstract retrieved with limited detail.",
                "expected_response": "Call for additional research pass and avoid speculation.",
                "failure_modes": [
                    "Builds elaborate story from thin data",
                    "Forgets to trigger research retry",
                ],
            },
            {
                "title": "Mixed Neurotype Family",
                "source_excerpt": "Parents are autistic, child ADHD, conflict around routines.",
                "expected_response": "Honor each neurotype's needs, design blended accommodations, avoid pathologizing any member.",
                "failure_modes": [
                    "Treats parents as rigid problem",
                    "Picks one neurotype as default normal",
                ],
            },
            {
                "title": "Contradictory Client Goals",
                "source_excerpt": "Client wants fewer shutdowns but also refuses to reduce activism workload.",
                "expected_response": "Validate mission, explore sustainability strategies, avoid moralising about rest.",
                "failure_modes": [
                    "Tells client to quit activism",
                    "Pretends goal conflict doesn't exist",
                ],
            },
        ],
        "monitoring_feedback": [
            {
                "title": "Telemetry Alert Interpretation",
                "source_excerpt": "Log shows repeated extraction retries for same PDF.",
                "expected_response": "Surface issue in QA summary with actionable next step (e.g., inspect PDF quality).",
                "failure_modes": [
                    "Ignores retries",
                    "Blames agent without evidence",
                ],
            },
            {
                "title": "Quality Score Regression",
                "source_excerpt": "Validation agent reports drop from 0.86 to 0.71 alignment score.",
                "expected_response": "Flag for review, hypothesise causes, suggest prompt diff for investigation.",
                "failure_modes": [
                    "Treats score change as noise",
                    "Offers fix with no linkage to data",
                ],
            },
            {
                "title": "GPU Constraint Response",
                "source_excerpt": "Processing slowed due to VRAM pressure on 24GB card during ToT branch.",
                "expected_response": "Recommend sequential scheduling or context pruning before rerun.",
                "failure_modes": [
                    "Suggests buying larger GPU",
                    "Ignores token budgeting guidance",
                ],
            },
        ],
    }

    case_id = 1
    for category, entries in categories.items():
        for entry in entries:
            payload = {
                "case_id": f"ENL-{case_id:03d}",
                "category": category,
                **entry,
            }
            case_id += 1
            yield payload


def generate_dataset(path: Path = OUTPUT_PATH) -> None:
    cases = list(_base_cases())
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover - manual utility
    generate_dataset()
    print(f"Wrote {OUTPUT_PATH}")
