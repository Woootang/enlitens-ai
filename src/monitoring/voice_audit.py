"""Voice drift auditing utilities for Enlitens content generation.

Week 1 of the roadmap requires a voice drift audit that compares generated
content to canonical Enlitens prose. This module provides a reusable
implementation that can run locally on the GPU workstation without additional
infrastructure.

The auditor supports two embedding backends:

* ``sentence-transformers`` when available (leveraging GPU if possible).
* A pure-Python TF–IDF fallback so audits still run when transformer models or
  heavy numerical libraries are unavailable on the box.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback
    SentenceTransformer = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceAuditResult:
    document_id: str
    similarity: float
    verdict: str


class EmbeddingBackend:
    """Abstract base class for text embedding backends."""

    def fit_reference(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def transform(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover - interface
        raise NotImplementedError


class TransformerBackend(EmbeddingBackend):
    """SentenceTransformer powered embeddings."""

    def __init__(self, model_name: str = "all-mpnet-base-v2", device: Optional[str] = None):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is unavailable")
        if device is None:
            device = "cuda" if torch_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def fit_reference(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - heavy call
        return self._encode(texts)

    def transform(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - heavy call
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
        return [[float(value) for value in row] for row in embeddings]


class LightweightTfidfBackend(EmbeddingBackend):
    """Pure-Python TF–IDF style encoder suitable for constrained environments."""

    TOKEN_RE = re.compile(r"[\w']+")

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []

    def _tokenise(self, text: str) -> List[str]:
        return [match.group(0).lower() for match in self.TOKEN_RE.finditer(text)]

    def fit_reference(self, texts: Sequence[str]) -> List[List[float]]:
        documents = [self._tokenise(text) for text in texts]
        vocab: Dict[str, int] = {}
        for tokens in documents:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocabulary = vocab
        doc_freq = [0] * len(vocab)
        for tokens in documents:
            seen = set()
            for token in tokens:
                idx = vocab[token]
                if idx not in seen:
                    doc_freq[idx] += 1
                    seen.add(idx)
        total_docs = max(len(documents), 1)
        self.idf = [math.log((1 + total_docs) / (1 + df)) + 1.0 for df in doc_freq]
        return [self._vectorise(tokens) for tokens in documents]

    def transform(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.vocabulary:
            raise RuntimeError("TF–IDF backend must be fitted before use")
        return [self._vectorise(self._tokenise(text)) for text in texts]

    def _vectorise(self, tokens: List[str]) -> List[float]:
        vector = [0.0] * len(self.vocabulary)
        if not tokens:
            return vector
        counts: Dict[int, int] = {}
        for token in tokens:
            idx = self.vocabulary.get(token)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        total_terms = sum(counts.values()) or 1
        for idx, count in counts.items():
            tf = count / total_terms
            vector[idx] = tf * self.idf[idx]
        return vector


def cosine_matrix(vectors_a: Sequence[Sequence[float]], vectors_b: Sequence[Sequence[float]]) -> List[List[float]]:
    """Compute cosine similarity matrix without requiring numpy."""

    result: List[List[float]] = []
    for vector_a in vectors_a:
        row: List[float] = []
        for vector_b in vectors_b:
            row.append(_cosine(vector_a, vector_b))
        result.append(row)
    return result


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ax, bx in zip(a, b):
        dot += ax * bx
        norm_a += ax * ax
        norm_b += bx * bx
    for ax in a[len(b):]:
        norm_a += ax * ax
    for bx in b[len(a):]:
        norm_b += bx * bx
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def torch_available() -> bool:  # pragma: no cover - runtime detection
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


class VoiceAuditor:
    """Scores generated passages against canonical Enlitens samples."""

    def __init__(self, reference_texts: Sequence[str], backend: Optional[EmbeddingBackend] = None):
        if not reference_texts:
            raise ValueError("At least one reference text is required")
        self.reference_texts = list(reference_texts)
        self.backend = backend or self._default_backend()
        self.reference_embeddings = self._to_matrix(
            self.backend.fit_reference(self.reference_texts)
        )

    def _default_backend(self) -> EmbeddingBackend:
        if SentenceTransformer is not None:
            try:
                return TransformerBackend()
            except Exception as exc:  # pragma: no cover - fallback path
                LOGGER.warning("Falling back to TF–IDF backend: %s", exc)
        return LightweightTfidfBackend()

    def score(self, document_id: str, text: str, threshold: float = 0.72) -> VoiceAuditResult:
        candidate_embedding = self._to_matrix(self.backend.transform([text]))[0]
        similarities = cosine_matrix([candidate_embedding], self.reference_embeddings)[0]
        similarity = max(similarities) if similarities else 0.0
        verdict = "pass" if similarity >= threshold else "drift"
        return VoiceAuditResult(document_id=document_id, similarity=similarity, verdict=verdict)

    def batch_score(self, samples: Sequence[dict], threshold: float = 0.72) -> List[VoiceAuditResult]:
        results: List[VoiceAuditResult] = []
        if not samples:
            return results
        texts = [sample["text"] for sample in samples]
        embeddings = self._to_matrix(self.backend.transform(texts))
        similarities = cosine_matrix(embeddings, self.reference_embeddings)
        for idx, sample in enumerate(samples):
            row = similarities[idx] if idx < len(similarities) else []
            similarity = max(row) if row else 0.0
            verdict = "pass" if similarity >= threshold else "drift"
            results.append(
                VoiceAuditResult(
                    document_id=sample.get("document_id", f"sample-{idx}"),
                    similarity=similarity,
                    verdict=verdict,
                )
            )
        return results

    def _to_matrix(self, embeddings: Sequence[Sequence[float]]) -> List[List[float]]:
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()  # type: ignore[assignment]
        matrix: List[List[float]] = []
        for row in embeddings:
            matrix.append([float(value) for value in row])
        return matrix


def load_reference_texts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    content = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    return blocks


def load_samples(path: Path) -> List[dict]:
    samples: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Enlitens voice alignment")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs") / "enlitens_voice_reference.md",
        help="Path to markdown file with canonical Enlitens voice samples",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Path to JSONL file with {\"document_id\", \"text\"} entries",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.72,
        help="Similarity threshold for pass/drift verdict",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON results")
    args = parser.parse_args(list(argv) if argv is not None else None)

    reference_texts = load_reference_texts(args.reference)
    auditor = VoiceAuditor(reference_texts)
    samples = load_samples(args.samples)
    results = auditor.batch_score(samples, threshold=args.threshold)

    if args.json:
        payload = [result.__dict__ for result in results]
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            print(f"{result.document_id}: similarity={result.similarity:.3f} verdict={result.verdict}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
