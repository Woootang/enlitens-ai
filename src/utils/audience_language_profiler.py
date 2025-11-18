"""
Audience Language Profiler
--------------------------

Builds a lightweight, deterministic view of the phrases, tone, and vocabulary
our audience actually uses.  The profiler fuses historic intake transcripts
with live GA4/Search Console analytics (when credentials are available) so the
agents can ground their language choices in real data instead of generic
clinical wording.

The output is intentionally simple: a structured dictionary that downstream
agents can consume plus a pre-formatted prompt block for injecting the guidance
into LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from enlitens_client_profiles.analytics import build_analytics_snapshot
from enlitens_client_profiles.config import ProfilePipelineConfig

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")

# A deliberately tiny stop-word list so we keep emotionally charged terms.
_STOPWORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "have",
    "from",
    "will",
    "just",
    "like",
    "they",
    "them",
    "their",
    "when",
    "your",
    "hear",
    "really",
    "about",
    "been",
    "into",
    "some",
    "very",
    "what",
    "want",
    "need",
    "dont",
    "doesnt",
    "cant",
    "should",
    "shouldnt",
    "could",
    "couldnt",
    "ourselves",
}

_DEFAULT_BANNED_TERMS = {
    "mindfulness",
    "mindful",
    "mindfully",
    "journey",
    "pathway",
    "pathways",
    "toxic positivity",
    "manifest",
    "manifestation",
    "gratitude practice",
    "breathe through it",
}


def _tokenise(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for idx in range(len(tokens) - n + 1):
        yield tuple(tokens[idx : idx + n])


def _format_ngram(ngram: Tuple[str, ...]) -> str:
    return " ".join(ngram)


def _top_items(counter: Counter, limit: int) -> List[str]:
    return [item for item, _ in counter.most_common(limit)]


@dataclass(slots=True)
class AudienceLanguageProfile:
    """Container for the compiled language insights."""

    top_terms: List[str]
    top_phrases: List[str]
    quotes: List[str]
    analytic_queries: List[str]
    analytic_pages: List[str]
    banned_terms: List[str]
    tone_descriptors: List[str]
    prompt_block: str
    raw_stats: Dict[str, Any]


class AudienceLanguageProfiler:
    """
    Build a reusable language profile grounded in Enlitens audience data.
    """

    def __init__(
        self,
        *,
        config: Optional[ProfilePipelineConfig] = None,
        banned_terms: Optional[Iterable[str]] = None,
    ) -> None:
        self.config = config or ProfilePipelineConfig()
        custom_terms = {term.lower().strip() for term in (banned_terms or []) if term}
        self.banned_terms = sorted((_DEFAULT_BANNED_TERMS | custom_terms))
        self._cache: Optional[AudienceLanguageProfile] = None

    def build_profile(self, *, force_refresh: bool = False) -> AudienceLanguageProfile:
        """
        Return an audience language profile (cached by default).
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        intakes_profile = self._profile_intakes(self.config.intakes_path)
        analytics_profile = self._profile_analytics()

        tone_descriptors = self._derive_tone_descriptors(intakes_profile)
        prompt_block = self._compose_prompt_block(intakes_profile, analytics_profile, tone_descriptors)

        profile = AudienceLanguageProfile(
            top_terms=intakes_profile.get("top_terms", []),
            top_phrases=intakes_profile.get("top_phrases", []),
            quotes=intakes_profile.get("quotes", []),
            analytic_queries=analytics_profile.get("queries", []),
            analytic_pages=analytics_profile.get("pages", []),
            banned_terms=self.banned_terms,
            tone_descriptors=tone_descriptors,
            prompt_block=prompt_block,
            raw_stats={
                "intakes": intakes_profile,
                "analytics": analytics_profile,
            },
        )
        self._cache = profile
        return profile

    # --------------------------------------------------------------------- #
    # Profiling helpers
    # --------------------------------------------------------------------- #

    def _profile_intakes(self, intakes_path: Path) -> Dict[str, Any]:
        """
        Pull top vocabulary and representative quotes from the intake corpus.
        """
        if not intakes_path.exists():
            logger.warning("AudienceLanguageProfiler: intakes file missing at %s", intakes_path)
            return {}

        try:
            text = intakes_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - IO failure
            logger.warning("AudienceLanguageProfiler: failed to read intakes: %s", exc)
            return {}

        tokens = _tokenise(text)
        filtered_tokens = [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 2]

        word_counter = Counter(filtered_tokens)
        bigram_counter = Counter(_ngrams(filtered_tokens, 2))
        trigram_counter = Counter(_ngrams(filtered_tokens, 3))

        phrase_counter: Counter[str] = Counter()
        for grams_counter in (bigram_counter, trigram_counter):
            for grams, count in grams_counter.items():
                phrase_counter[_format_ngram(grams)] += count

        quotes = self._extract_quotes(text)

        return {
            "top_terms": _top_items(word_counter, 40),
            "top_phrases": _top_items(phrase_counter, 25),
            "quotes": quotes,
            "token_count": len(tokens),
            "unique_terms": len(word_counter),
        }

    def _profile_analytics(self) -> Dict[str, Any]:
        """
        Pull recent GA4 + Search Console queries when credentials are available.
        """
        snapshot = None
        try:
            snapshot = build_analytics_snapshot(
                credentials_path=self.config.google_credentials_path,
                ga_property_id=self.config.ga_property_id,
                gsc_site_url=self.config.gsc_site_url,
                lookback_days=self.config.analytics_lookback_days,
            )
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("AudienceLanguageProfiler: analytics snapshot failed: %s", exc)

        if snapshot is None:
            return {}

        queries = [
            query.query.strip()
            for query in snapshot.gsc_queries
            if query.query and len(query.query.strip().split()) <= 8
        ]
        pages = [
            json.dumps(page.extra, ensure_ascii=False)
            if page.extra
            else page.metric
            for page in snapshot.ga_top_pages[:10]
        ]

        return {
            "queries": queries[:25],
            "pages": pages[:10],
            "generated_at": snapshot.generated_at.isoformat(),
        }

    def _extract_quotes(self, text: str) -> List[str]:
        """
        Pull a handful of representative, plain-language quotes.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        quotes: List[str] = []

        for line in lines:
            if len(quotes) >= 8:
                break
            # Bias towards lines that look like client statements.
            if any(line.lower().startswith(prefix) for prefix in ("i ", "we ", "my ", "our ")):
                quotes.append(line[:220])

        return quotes

    def _derive_tone_descriptors(self, intakes_profile: Dict[str, Any]) -> List[str]:
        """
        Provide quick descriptors about how clients talk (e.g., blunt, exhausted).
        """
        descriptors: List[str] = []
        top_terms = intakes_profile.get("top_terms", [])
        phrases = intakes_profile.get("top_phrases", [])
        joined = " ".join(top_terms + phrases)

        if not joined:
            return descriptors

        lower_joined = joined.lower()
        if any(word in lower_joined for word in ("burnout", "burned", "fried", "exhausted")):
            descriptors.append("exhausted")
        if any(word in lower_joined for word in ("overwhelm", "overloaded", "drowning")):
            descriptors.append("overwhelmed")
        if any(word in lower_joined for word in ("angry", "pissed", "fed up", "bullshit")):
            descriptors.append("fed up with the system")
        if any(word in lower_joined for word in ("neurodivergent", "adhd", "autistic", "sensory")):
            descriptors.append("neurodivergent foregrounded")

        return descriptors

    # --------------------------------------------------------------------- #
    # Prompt shaping
    # --------------------------------------------------------------------- #

    def _compose_prompt_block(
        self,
        intakes_profile: Dict[str, Any],
        analytics_profile: Dict[str, Any],
        tone_descriptors: List[str],
    ) -> str:
        """
        Convert raw stats into a compact, human-readable prompt injection.
        """
        lines: List[str] = ["CLIENT LANGUAGE SNAPSHOT:"]

        top_terms = intakes_profile.get("top_terms", [])[:15]
        if top_terms:
            lines.append(f"- Frequent words clients use: {', '.join(top_terms)}")

        top_phrases = intakes_profile.get("top_phrases", [])[:8]
        if top_phrases:
            lines.append(f"- Common phrases: {', '.join(top_phrases)}")

        quotes = intakes_profile.get("quotes", [])[:3]
        if quotes:
            lines.append("- Representative client quotes:")
            for quote in quotes:
                lines.append(f'  • "{quote}"')

        queries = analytics_profile.get("queries", [])[:5]
        if queries:
            lines.append(f"- Search queries bringing traffic: {', '.join(queries)}")

        if tone_descriptors:
            lines.append(f"- Tone descriptors: {', '.join(tone_descriptors)}")

        if self.banned_terms:
            lines.append(f"- Avoid these words unless the source explicitly uses them: {', '.join(self.banned_terms)}")

        lines.append("- Speak plainly. No therapy clichés, no toxic positivity, no metaphor about journeys/pathways.")

        return "\n".join(lines)


__all__ = ["AudienceLanguageProfiler", "AudienceLanguageProfile"]


