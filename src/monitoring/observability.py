"""Centralized observability configuration for the Enlitens pipeline."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import requests

try:  # Optional dependency: OpenTelemetry
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
except Exception:  # pragma: no cover - library might not be installed in tests
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore

try:  # Optional dependency: Sentry
    import sentry_sdk
except Exception:  # pragma: no cover - optional dependency
    sentry_sdk = None  # type: ignore

try:  # Optional dependency: Phoenix
    from phoenix.trace.exporter import PhoenixSpanExporter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PhoenixSpanExporter = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ObservabilityConfig:
    """Runtime configuration for observability features."""

    service_name: str = "enlitens-pipeline"
    service_version: str = "1.0.0"
    environment: str = os.getenv("ENLITENS_ENV", "development")
    langfuse_endpoint: Optional[str] = os.getenv("LANGFUSE_OTLP_ENDPOINT")
    langfuse_public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    phoenix_endpoint: Optional[str] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    phoenix_dataset: str = os.getenv("PHOENIX_DATASET", "enlitens-rag")
    monitoring_broadcast_url: Optional[str] = os.getenv(
        "MONITORING_BROADCAST_URL", "http://localhost:8765/api/broadcast"
    )
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    sentry_sample_rate: float = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))


class _NullSpan:
    """Fallback span that mimics the OpenTelemetry span API."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    # OpenTelemetry span compatibility methods
    def set_attribute(self, key: str, value: Any) -> None:  # pragma: no cover - noop
        return None

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        return None

    def record_exception(self, exc: Exception) -> None:  # pragma: no cover - noop
        return None

    @property
    def duration(self) -> float:
        return time.perf_counter() - self._start


class ObservabilityManager:
    """Configures telemetry, tracing exports, and alerting hooks."""

    def __init__(self, config: Optional[ObservabilityConfig] = None) -> None:
        self.config = config or ObservabilityConfig()
        self._tracer_provider = None
        self._monitoring_session = requests.Session()
        self._langfuse_enabled = False
        self._phoenix_enabled = False
        self._sentry_enabled = False
        self._configure_sentry()
        self._configure_tracing()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _configure_tracing(self) -> None:
        if trace is None or TracerProvider is None:
            logger.warning("OpenTelemetry SDK not available; tracing disabled")
            return

        resource = Resource.create(
            {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
            }
        )

        provider = TracerProvider(resource=resource)
        self._tracer_provider = provider
        trace.set_tracer_provider(provider)

        # Console exporter for local debugging
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(console_exporter))

        # Langfuse exporter via OTLP
        if self.config.langfuse_endpoint and self.config.langfuse_secret_key:
            headers = {
                "x-langfuse-public-key": self.config.langfuse_public_key or "",
                "x-langfuse-secret-key": self.config.langfuse_secret_key,
            }
            try:
                langfuse_exporter = OTLPSpanExporter(
                    endpoint=self.config.langfuse_endpoint,
                    headers=headers,
                )
                provider.add_span_processor(BatchSpanProcessor(langfuse_exporter))
                self._langfuse_enabled = True
                logger.info("Langfuse OTLP exporter configured")
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning("Failed to initialize Langfuse exporter: %s", exc)

        # Phoenix exporter for RAG metrics
        if PhoenixSpanExporter and self.config.phoenix_endpoint:
            try:
                phoenix_exporter = PhoenixSpanExporter(
                    endpoint=self.config.phoenix_endpoint,
                    project_name=self.config.phoenix_dataset,
                )
                provider.add_span_processor(BatchSpanProcessor(phoenix_exporter))
                self._phoenix_enabled = True
                logger.info("Phoenix exporter configured")
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning("Failed to initialize Phoenix exporter: %s", exc)
        elif not self.config.phoenix_endpoint:
            logger.info("Phoenix exporter not configured (PHOENIX_COLLECTOR_ENDPOINT missing)")

    def _configure_sentry(self) -> None:
        if not self.config.sentry_dsn:
            logger.info("Sentry DSN not provided; error aggregation disabled")
            return
        if sentry_sdk is None:
            logger.warning("sentry-sdk not installed; cannot initialize Sentry")
            return

        sentry_sdk.init(
            dsn=self.config.sentry_dsn,
            traces_sample_rate=self.config.sentry_sample_rate,
            environment=self.config.environment,
            release=self.config.service_version,
        )
        self._sentry_enabled = True
        logger.info("Sentry initialized for structured error aggregation")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_tracer(self, name: str):  # type: ignore[override]
        if trace is None:
            return None
        return trace.get_tracer(name)

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        tracer = trace.get_tracer(__name__) if trace is not None else None
        if tracer is None:
            span = _NullSpan()
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
            return

        with tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def capture_exception(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        logger.error("Pipeline error: %s", error, exc_info=True)
        if self._sentry_enabled and sentry_sdk:
            with sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_extra(key, value)
                sentry_sdk.capture_exception(error)
        self._broadcast(
            {
                "type": "alert",
                "severity": "error",
                "title": "Pipeline failure",
                "details": {
                    "message": str(error),
                    "context": context or {},
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def emit_alert(
        self,
        title: str,
        severity: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "type": "alert",
            "severity": severity,
            "title": title,
            "details": {"description": description, **(context or {})},
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.warning("%s: %s", title, description)
        if self._sentry_enabled and sentry_sdk:
            sentry_sdk.capture_message(
                f"{title}: {description}",
                level=severity if severity in {"info", "warning", "error", "fatal"} else "info",
            )
        self._broadcast(payload)

    def record_stage_timing(
        self,
        document_id: str,
        stage_name: str,
        agent_name: str,
        start_time: datetime,
        end_time: datetime,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "type": "timeline",
            "document_id": document_id,
            "stage": stage_name,
            "agent": agent_name,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "status": status,
            "metadata": metadata or {},
        }
        self._broadcast(payload)

    def record_quality_metrics(
        self,
        document_id: str,
        metrics: Dict[str, Any],
    ) -> None:
        payload = {
            "type": "quality_metrics",
            "document_id": document_id,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._broadcast(payload)

    def record_rag_metrics(
        self,
        document_id: str,
        prompt: str,
        context_chunks: Iterable[str],
        response_text: str,
        citations: Optional[Iterable[Dict[str, Any]]] = None,
        score: Optional[float] = None,
    ) -> None:
        payload = {
            "type": "rag_metrics",
            "document_id": document_id,
            "prompt": prompt,
            "context": list(context_chunks),
            "response": response_text,
            "citations": list(citations or []),
            "score": score,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._broadcast(payload)

        if self._phoenix_enabled and trace is not None:
            with self.start_span("phoenix.rag.metrics") as span:
                span.set_attribute("document.id", document_id)
                span.set_attribute("rag.context.count", len(payload["context"]))
                if score is not None:
                    span.set_attribute("rag.score", score)

    def record_citation_graph(
        self,
        document_id: str,
        nodes: Iterable[Dict[str, Any]],
        edges: Iterable[Dict[str, Any]],
    ) -> None:
        payload = {
            "type": "citation_graph",
            "document_id": document_id,
            "nodes": list(nodes),
            "edges": list(edges),
        }
        self._broadcast(payload)

    def record_system_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = {
            "type": "system_metrics",
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._broadcast(payload)

    def check_latency_anomaly(self, stage: str, duration_seconds: float, threshold: float = 300.0) -> None:
        if duration_seconds <= threshold:
            return
        self.emit_alert(
            title="Latency spike detected",
            severity="warning",
            description=f"Stage '{stage}' exceeded {threshold:.0f}s (took {duration_seconds:.1f}s)",
            context={"stage": stage, "duration_seconds": duration_seconds},
        )

    def check_quality_anomaly(self, document_id: str, quality_score: float, issues: Optional[Iterable[str]] = None) -> None:
        if quality_score >= 0.75 and not issues:
            return
        description = f"Quality score {quality_score:.2f} for document {document_id}"
        if issues:
            description += " | Issues: " + ", ".join(issues)
        self.emit_alert(
            title="Hallucination risk detected",
            severity="warning",
            description=description,
            context={"document_id": document_id, "quality_score": quality_score},
        )

    def check_cost_anomaly(self, document_id: str, estimated_tokens: int, threshold: int = 32000) -> None:
        if estimated_tokens <= threshold:
            return
        self.emit_alert(
            title="Cost anomaly detected",
            severity="info",
            description=(
                f"Document {document_id} estimated {estimated_tokens} tokens (threshold {threshold})"
            ),
            context={"document_id": document_id, "estimated_tokens": estimated_tokens},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self.config.monitoring_broadcast_url:
            return
        try:
            response = self._monitoring_session.post(
                self.config.monitoring_broadcast_url,
                json=payload,
                timeout=2,
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network errors
            logger.debug("Broadcast failed: %s", exc)


_observability_manager = ObservabilityManager()


def get_observability() -> ObservabilityManager:
    """Return the singleton observability manager."""

    return _observability_manager
