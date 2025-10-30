"""Few-shot example library with similarity-based retrieval.

This module centralizes curated golden examples for each agent task and
provides lightweight semantic retrieval so prompts can include the most
relevant exemplars. It also exposes prompt quality criteria that power the
testing harness.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptCriterion:
    """Quality criterion that a system prompt must satisfy."""

    name: str
    keywords: List[str]
    description: str

    def matches(self, text: str) -> bool:
        """Return True when all keywords appear in the prompt text."""

        lowered = text.lower()
        return all(keyword.lower() in lowered for keyword in self.keywords)


@dataclass(frozen=True)
class FewShotExample:
    """Golden example used for prompt few-shot augmentation."""

    task: str
    description: str
    input_context: str
    output: Dict[str, Any]
    rationale: str = ""
    criteria: List[PromptCriterion] = field(default_factory=list)

    def render(self) -> str:
        """Render the example as a formatted block for prompt injection."""

        pretty_json = json.dumps(self.output, indent=2, ensure_ascii=False)
        return (
            f"Summary: {self.description}\n"
            f"Input focus: {self.input_context.strip()}\n"
            f"Expected JSON:\n{pretty_json}"
        )


class FewShotLibrary:
    """Few-shot retrieval that selects examples per task via cosine similarity."""

    def __init__(self, examples: Iterable[FewShotExample]):
        self.examples_by_task: Dict[str, List[FewShotExample]] = {}
        self._token_counters: Dict[str, List[Counter[str]]] = {}
        self._document_frequency: Dict[str, Counter[str]] = {}
        self._vector_norms: Dict[str, List[float]] = {}

        for example in examples:
            self.examples_by_task.setdefault(example.task, []).append(example)

        for task, task_examples in self.examples_by_task.items():
            documents = [self._compose_document(example) for example in task_examples]
            if not documents:
                logger.warning("No documents available for task '%s'", task)
                continue

            token_lists = [self._tokenize(document) for document in documents]
            counters = [Counter(tokens) for tokens in token_lists]
            document_frequency: Counter[str] = Counter()
            for counter in counters:
                for token in counter:
                    document_frequency[token] += 1

            self._token_counters[task] = counters
            self._document_frequency[task] = document_frequency
            self._vector_norms[task] = [
                self._vector_norm(counter, document_frequency, len(counters))
                for counter in counters
            ]

    def _compose_document(self, example: FewShotExample) -> str:
        segments: List[str] = [example.description, example.input_context, example.rationale]
        for value in example.output.values():
            if isinstance(value, list):
                segments.append(" ".join(str(item) for item in value if item))
            elif isinstance(value, str):
                segments.append(value)

        return " \n".join(segment for segment in segments if isinstance(segment, str) and segment)

    def get_examples(self, task: str, query: Optional[str], k: int = 2) -> List[FewShotExample]:
        """Return the k most similar examples for a task."""

        task_examples = self.examples_by_task.get(task, [])
        if not task_examples:
            logger.debug("No few-shot examples registered for task '%s'", task)
            return []

        if not query or task not in self._token_counters:
            return task_examples[:k]

        counters = self._token_counters[task]
        document_frequency = self._document_frequency[task]
        norms = self._vector_norms[task]
        total_docs = len(counters)

        query_counter = Counter(self._tokenize(query))
        query_norm = self._vector_norm(query_counter, document_frequency, total_docs)

        scores = []
        for idx, counter in enumerate(counters):
            dot = self._dot_product(query_counter, counter, document_frequency, total_docs)
            denom = query_norm * norms[idx]
            score = dot / denom if denom else 0.0
            scores.append((score, task_examples[idx]))

        ranked = sorted(scores, key=lambda item: item[0], reverse=True)
        return [example for _, example in ranked[:k]]

    def render_for_prompt(self, task: str, query: Optional[str], k: int = 2) -> str:
        """Render the few-shot exemplars for injection into a prompt."""

        examples = self.get_examples(task, query, k)
        if not examples:
            return ""

        blocks = []
        for index, example in enumerate(examples, 1):
            blocks.append(
                f"### Few-shot example {index}\n{example.render()}"
            )

        return "\n\n".join(blocks)

    def aggregate_criteria(self, task: str) -> List[PromptCriterion]:
        """Return the unique prompt criteria for a task."""

        seen = {}
        for example in self.examples_by_task.get(task, []):
            for criterion in example.criteria:
                seen.setdefault(criterion.name, criterion)
        return list(seen.values())

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text or "") if token]

    def _idf(self, token: str, total_docs: int, document_frequency: Counter[str]) -> float:
        df = document_frequency.get(token, 0)
        return math.log((total_docs + 1) / (df + 1)) + 1.0

    def _vector_norm(self, counter: Counter[str], document_frequency: Counter[str], total_docs: int) -> float:
        norm = 0.0
        for token, freq in counter.items():
            idf = self._idf(token, total_docs, document_frequency)
            norm += (freq * idf) ** 2
        return math.sqrt(norm)

    def _dot_product(
        self,
        counter_a: Counter[str],
        counter_b: Counter[str],
        document_frequency: Counter[str],
        total_docs: int,
    ) -> float:
        dot = 0.0
        for token, freq_a in counter_a.items():
            if token not in counter_b:
                continue
            freq_b = counter_b[token]
            idf = self._idf(token, total_docs, document_frequency)
            dot += (freq_a * idf) * (freq_b * idf)
        return dot


GOLDEN_FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    FewShotExample(
        task="science_extraction",
        description="Adult ADHD mindfulness trial with dopamine transporter outcomes",
        input_context=(
            "Randomized controlled trial measuring dopamine transporter availability "
            "after an eight-week mindfulness-based intervention for adults with ADHD."
        ),
        output={
            "findings": [
                "Mindfulness training produced a 21% reduction in dorsal striatal dopamine transporter availability compared to controls.",
                "Intervention participants showed significant improvements in executive function scales (p = 0.018).",
                "Follow-up assessments indicated sustained attentional gains at 12 weeks post-treatment."
            ],
            "statistics": [
                "According to Rivera et al. (2024), \"dopamine transporter binding decreased by 21% in the mindfulness group\" [Source: rivera_adhd_mindfulness, pg 7]",
                "According to Rivera et al. (2024), \"executive function composite scores improved by 0.6 SD\" [Source: rivera_adhd_mindfulness, pg 9]",
                "According to Rivera et al. (2024), \"12-week maintenance retained 78% of attentional gains\" [Source: rivera_adhd_mindfulness, pg 11]"
            ],
            "methodologies": [
                "Randomized controlled design with mindfulness versus waitlist control arms.",
                "DAT-SPECT imaging at baseline and week eight to quantify transporter availability.",
                "Blinded clinical assessors administering executive function batteries."
            ],
            "limitations": [
                "Sample skewed toward college-educated adults which limits generalizability.",
                "Short follow-up interval restricts interpretation of long-term neurochemical change.",
                "Mindfulness fidelity relied on self-reported home practice logs."
            ],
            "future_directions": [
                "Replicate findings in adolescent ADHD populations.",
                "Compare mindfulness against stimulant titration with imaging endpoints.",
                "Evaluate booster sessions for sustaining dopamine transporter modulation."
            ],
            "implications": [
                "Mindfulness protocols can target dopaminergic regulation alongside pharmacology.",
                "Executive coaching elements may enhance cognitive outcomes in ADHD therapy.",
                "Maintenance plans should monitor neurochemical markers beyond the acute phase."
            ],
            "citations": [
                "Rivera et al. (2024)",
                "Zhao & Malik (2022)"
            ],
            "references": [
                "Rivera, L. et al. (2024). Mindfulness modulation of dopamine transporters. Journal of Cognitive Neuroscience.",
                "Zhao, P. & Malik, J. (2022). Imaging biomarkers for ADHD interventions. Neurotherapeutics."
            ]
        },
        rationale=(
            "Demonstrates precise statistical quoting with citation tags and highlights how to"
            " document methodologies and limitations in grounded language."
        ),
        criteria=[
            PromptCriterion(
                name="Grounding",
                keywords=["ONLY", "provided", "document"],
                description="Model must restrict outputs to supplied document content."
            ),
            PromptCriterion(
                name="Refusal",
                keywords=["Refusal", "insufficient grounding"],
                description="Model must refuse when sources do not contain the answer."
            ),
            PromptCriterion(
                name="Citation format",
                keywords=["[Source:"],
                description="Statistics must include bracketed source citations."
            ),
        ],
    ),
    FewShotExample(
        task="clinical_synthesis",
        description="Transcranial direct current stimulation for trauma-related rumination",
        input_context=(
            "Integrated review summarizing frontopolar tDCS paired with exposure therapy for"
            " trauma survivors experiencing intrusive rumination."
        ),
        output={
            "interventions": [
                "Pair frontopolar tDCS (2 mA, 20 minutes) with imaginal exposure sessions to dampen intrusive rumination.",
                "Introduce metacognitive restructuring that links neuroplastic changes to daily grounding rituals.",
                "Layer heart-rate variability biofeedback before trauma narratives to stabilize autonomic arousal."
            ],
            "assessments": [
                "Use the Impact of Event Scale weekly to monitor rumination frequency.",
                "Track dorsolateral prefrontal activation with functional near-infrared spectroscopy during sessions.",
                "Administer distress tolerance logs to capture between-session regulation attempts."
            ],
            "outcomes": [
                "Reduced nightly intrusive images reported over a four-week block.",
                "Improved executive control scores on the Cognitive Flexibility Scale.",
                "Increased heart-rate variability coherence during exposure."
            ],
            "protocols": [
                "Deliver ten-session protocol blending 15 minutes of regulation rehearsal with 30 minutes of trauma processing.",
                "Assign daily neuroplasticity journaling to reinforce new cognitive pathways.",
                "Schedule booster tDCS sessions at one-month intervals for maintenance."
            ],
            "guidelines": [
                "Obtain neurologist clearance prior to neuromodulation for clients with seizure history.",
                "Prime sessions with psychoeducation linking brain stimulation to symptom relief.",
                "Document informed consent highlighting experimental aspects of combined protocol."
            ],
            "contraindications": [
                "Avoid tDCS when clients have implanted medical devices or unstable mood disorders.",
                "Postpone exposure sessions during acute dissociative episodes.",
                "Use caution with clients reporting severe migraines after neuromodulation."
            ],
            "side_effects": [
                "Transient scalp tingling and mild fatigue after stimulation.",
                "Occasional headache lasting under 60 minutes.",
                "Short-lived emotional flooding requiring grounding."
            ],
            "monitoring": [
                "Collect session-by-session rumination ratings using a 0-10 visual analog scale.",
                "Review sleep quality trackers to detect rebound symptoms.",
                "Log homework completion to correlate practice with symptom shifts."
            ],
        },
        rationale="Illustrates how to translate research implications into concrete clinical playbooks without inventing new science.",
        criteria=[
            PromptCriterion(
                name="Clinical grounding",
                keywords=["Base all interventions", "research findings"],
                description="Ensure interventions explicitly reference research context."
            ),
            PromptCriterion(
                name="Hypothetical marking",
                keywords=["Potential", "application"],
                description="Require speculative content to be marked as potential."
            ),
        ],
    ),
    FewShotExample(
        task="educational_content",
        description="Explaining sensory gating research to clients",
        input_context=(
            "Findings on sensory gating deficits in autistic adults and strategies that improve"
            " thalamocortical filtering through occupational therapy routines."
        ),
        output={
            "explanations": [
                "Sensory gating is the brain's volume knob—when it sticks, every sound and texture rushes in at full blast.",
                "Thalamocortical circuits decide what input gets promoted to conscious awareness, which is why crowd noise can feel unbearable.",
                "Therapeutic sensory diets retrain the brain to predict input, calming the alarm response."
            ],
            "examples": [
                "Imagine trying to read while every light in the room flashes; that's how commuting home can feel after work.",
                "When the elevator door opens, the rush of sound hits like a wave before your brain has time to brace.",
                "Wearing noise-dampening headphones is like dimming the lights so the nervous system can recalibrate."
            ],
            "analogies": [
                "Your brain's filter is like a club bouncer—when it's overwhelmed, everyone storms inside.",
                "Think of the thalamus as the office manager deciding which emails hit your inbox.",
                "Calming routines act like pre-check security, letting safe signals glide through."  
            ],
            "definitions": [
                "Sensory gating: the process that blocks redundant information before it overloads the cortex.",
                "Predictive coding: the brain's habit of guessing what comes next to save energy.",
                "Habituation: getting used to repeated input so it stops triggering alarms."
            ],
            "processes": [
                "Step 1: Sensory input hits the thalamus; Step 2: The thalamus filters; Step 3: The cortex acts on what remains.",
                "Before therapy, every input gets flagged as urgent; after practice, the brain recognizes familiar signals and relaxes.",
                "Over time, repeated regulation drills teach the nervous system which sensations are safe."
            ],
            "comparisons": [
                "Unlike distraction, sensory regulation teaches the brain to reinterpret the noise, not just ignore it.",
                "Traditional coping avoids triggers, while sensory integration builds tolerance.",
                "Noise-cancelling headphones reduce volume, but occupational therapy changes the mixing board."
            ],
            "visual_aids": [
                "Flowchart showing sensory input travelling through the thalamus before reaching the cortex.",
                "Before/after graph depicting reduced startle responses after regulation practice.",
                "Illustration of a sensory diet schedule broken into morning, midday, and evening anchors."
            ],
            "learning_objectives": [
                "Understand why certain environments cause sensory overload.",
                "Identify at least three regulation tools to deploy during peak overwhelm.",
                "Explain how consistent routines retrain sensory gating pathways."
            ],
        },
        rationale="Shows the tone, readability, and concreteness expected for educational handouts.",
        criteria=[
            PromptCriterion(
                name="Accessibility",
                keywords=["8th grade", "client"],
                description="Ensure prompts demand client-accessible language."
            ),
            PromptCriterion(
                name="Hypothetical flag",
                keywords=["[HYPOTHETICAL EXAMPLE]"],
                description="Examples must be clearly labelled when hypothetical."
            ),
        ],
    ),
    FewShotExample(
        task="marketing_content",
        description="Positioning neuroscience therapy in a local market",
        input_context=(
            "Synthesised talking points connecting neuroscience-backed therapy outcomes to client"
            " pain points for St. Louis marketing campaigns."
        ),
        output={
            "headlines": [
                "Your brain isn't broken—it's ready for a reboot.",
                "Science-forward therapy for the moments your coping stops coping.",
                "Rewrite burnout with neuroscience that sticks."
            ],
            "taglines": [
                "Less shame, more synapses.",
                "Therapy with lab-grade receipts.",
                "Grounded in research, delivered with rebellion."
            ],
            "value_propositions": [
                "Personalized care plans map brain circuits to daily relief.",
                "We translate clinical trials into Tuesday night coping rituals.",
                "Every session comes with a why behind the what."
            ],
            "benefits": [
                "Sleep without replaying the entire day.",
                "Handle sensory storms without cancelling life.",
                "Know what your brain is doing—and how to redirect it."
            ],
            "pain_points": [
                "You're exhausted from coping plans that ignore your neurology.",
                "Your therapist keeps telling you to breathe when your brain needs a manual.",
                "You want science that respects your rebellious streak."
            ],
        },
        rationale="Demonstrates rebellious but responsible positioning that avoids testimonials or guarantees.",
        criteria=[
            PromptCriterion(
                name="Compliance",
                keywords=["No testimonials", "No guarantees"],
                description="Ensure marketing prompts ban testimonials and promises."
            ),
            PromptCriterion(
                name="Local focus",
                keywords=["St. Louis"],
                description="Marketing prompts should anchor to the local market."
            ),
        ],
    ),
]


FEW_SHOT_LIBRARY = FewShotLibrary(GOLDEN_FEW_SHOT_EXAMPLES)

