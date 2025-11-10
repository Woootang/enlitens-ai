"""Integrations for pulling Google Analytics 4 and Search Console data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GA_SCOPES = ("https://www.googleapis.com/auth/analytics.readonly",)
GSC_SCOPES = ("https://www.googleapis.com/auth/webmasters.readonly",)


@dataclass(slots=True)
class GoogleAnalyticsInsight:
    metric: str
    value: Any
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchConsoleQuery:
    query: str
    clicks: float
    impressions: float
    ctr: float
    position: float
    country: Optional[str] = None
    device: Optional[str] = None


@dataclass(slots=True)
class AnalyticsSnapshot:
    generated_at: datetime
    lookback_days: int
    ga_property_id: Optional[str]
    gsc_site_url: Optional[str]
    ga_top_pages: List[GoogleAnalyticsInsight] = field(default_factory=list)
    ga_locations: List[GoogleAnalyticsInsight] = field(default_factory=list)
    ga_events: List[GoogleAnalyticsInsight] = field(default_factory=list)
    gsc_queries: List[SearchConsoleQuery] = field(default_factory=list)
    gsc_pages: List[SearchConsoleQuery] = field(default_factory=list)

    def summary_block(self) -> str:
        """Return a human-readable summary suitable for prompting."""

        lines: List[str] = []
        if self.ga_top_pages:
            lines.append("Top GA4 landing pages (by sessions):")
            for insight in self.ga_top_pages[:10]:
                label = insight.extra.get("pagePath") or insight.metric
                lines.append(f"- {label}: {insight.value} sessions")
        if self.ga_locations:
            lines.append("\nWhere traffic is coming from:")
            for insight in self.ga_locations[:10]:
                city = insight.extra.get("city") or insight.metric
                state = insight.extra.get("region")
                lines.append(f"- {city}, {state or 'N/A'}: {insight.value} users")
        if self.gsc_queries:
            lines.append("\nTop Search Console queries:")
            for row in self.gsc_queries[:10]:
                lines.append(
                    f"- \"{row.query}\" | clicks: {row.clicks:.0f}, impressions: {row.impressions:.0f}, CTR: {row.ctr:.1%}, avg position: {row.position:.1f}"
                )
        if self.gsc_pages:
            lines.append("\nSearch-driven pages:")
            for row in self.gsc_pages[:10]:
                label = row.query or row.country or "page"
                lines.append(
                    f"- {label} | clicks: {row.clicks:.0f}, impressions: {row.impressions:.0f}, avg position: {row.position:.1f}"
                )
        return "\n".join(lines).strip()


def _load_credentials(path: Path, scopes: List[str]):
    from google.oauth2 import service_account  # type: ignore[import]

    return service_account.Credentials.from_service_account_file(str(path), scopes=scopes)


def _date_range(lookback_days: int) -> Dict[str, str]:
    end = date.today()
    start = end - timedelta(days=max(lookback_days, 1))
    return {"start": start.isoformat(), "end": end.isoformat()}


def fetch_google_analytics(
    *,
    credentials_path: Path,
    property_id: str,
    lookback_days: int = 60,
    page_limit: int = 15,
) -> Dict[str, List[GoogleAnalyticsInsight]]:
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient  # type: ignore[import]
        from google.analytics.data_v1beta.types import (  # type: ignore[import]
            DateRange,
            Dimension,
            Metric,
            OrderBy,
            RunReportRequest,
        )
    except Exception as exc:  # pragma: no cover - dependency missing
        logger.warning("google-analytics-data not available: %s", exc)
        return {}

    credentials = _load_credentials(credentials_path, list(GA_SCOPES))
    client = BetaAnalyticsDataClient(credentials=credentials)

    daterange = DateRange(start_date=_date_range(lookback_days)["start"], end_date=_date_range(lookback_days)["end"])
    property_name = f"properties/{property_id}"

    def _run(request: RunReportRequest) -> List[GoogleAnalyticsInsight]:
        response = client.run_report(request)
        insights: List[GoogleAnalyticsInsight] = []
        for row in response.rows:
            dimensions = {dim.name: value for dim, value in zip(request.dimensions, row.dimension_values)}
            metrics = {met.name: value for met, value in zip(request.metrics, row.metric_values)}
            metric_name, metric_value = next(iter(metrics.items()))
            insights.append(
                GoogleAnalyticsInsight(
                    metric=metric_name,
                    value=float(metric_value.value or 0),
                    extra=dimensions,
                )
            )
        return insights

    top_pages = _run(
        RunReportRequest(
            property=property_name,
            date_ranges=[daterange],
            dimensions=[Dimension(name="pageTitle"), Dimension(name="pagePath")],
            metrics=[Metric(name="sessions")],
            limit=page_limit,
            order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="sessions"), desc=True)],
        )
    )

    top_locations = _run(
        RunReportRequest(
            property=property_name,
            date_ranges=[daterange],
            dimensions=[Dimension(name="city"), Dimension(name="region"), Dimension(name="country")],
            metrics=[Metric(name="totalUsers")],
            limit=page_limit,
            order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="totalUsers"), desc=True)],
        )
    )

    top_events = _run(
        RunReportRequest(
            property=property_name,
            date_ranges=[daterange],
            dimensions=[Dimension(name="eventName")],
            metrics=[Metric(name="eventCount")],
            limit=page_limit,
            order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="eventCount"), desc=True)],
        )
    )

    return {
        "pages": top_pages,
        "locations": top_locations,
        "events": top_events,
    }


def fetch_search_console(
    *,
    credentials_path: Path,
    site_url: str,
    lookback_days: int = 60,
    row_limit: int = 25,
) -> Dict[str, List[SearchConsoleQuery]]:
    try:
        from googleapiclient.discovery import build  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - dependency missing
        logger.warning("googleapiclient not available: %s", exc)
        return {}

    credentials = _load_credentials(credentials_path, list(GSC_SCOPES))
    service = build("searchconsole", "v1", credentials=credentials)

    date_range = _date_range(lookback_days)

    def _query(dimensions: List[str]) -> List[SearchConsoleQuery]:
        try:
            response = (
                service.searchanalytics()
                .query(
                    siteUrl=site_url,
                    body={
                        "startDate": date_range["start"],
                        "endDate": date_range["end"],
                        "dimensions": dimensions,
                        "rowLimit": row_limit,
                        "dataState": "all",
                    },
                )
                .execute()
            )
        except Exception as exc:  # pragma: no cover - API failure
            logger.warning("Search Console query failed (dimensions=%s): %s", dimensions, exc)
            return []

        rows = response.get("rows", [])
        results: List[SearchConsoleQuery] = []
        for row in rows:
            keys = row.get("keys", [])
            query = keys[0] if keys else ""
            country = None
            device = None
            if len(keys) > 1:
                country = keys[1]
            if len(keys) > 2:
                device = keys[2]
            results.append(
                SearchConsoleQuery(
                    query=query,
                    clicks=row.get("clicks", 0.0),
                    impressions=row.get("impressions", 0.0),
                    ctr=row.get("ctr", 0.0),
                    position=row.get("position", 0.0),
                    country=country,
                    device=device,
                )
            )
        return results

    top_queries = _query(["query"])
    page_performance = _query(["page", "country"])

    return {
        "queries": top_queries,
        "pages": page_performance,
    }


def build_analytics_snapshot(
    *,
    credentials_path: Optional[Path],
    ga_property_id: Optional[str],
    gsc_site_url: Optional[str],
    lookback_days: int = 60,
) -> Optional[AnalyticsSnapshot]:
    if not credentials_path or not credentials_path.exists():
        logger.info("Google credentials path not provided or missing; skipping analytics snapshot.")
        return None

    lookup_path = credentials_path.expanduser()
    ga_data: Dict[str, List[GoogleAnalyticsInsight]] = {}
    gsc_data: Dict[str, List[SearchConsoleQuery]] = {}

    if ga_property_id:
        try:
            ga_data = fetch_google_analytics(
                credentials_path=lookup_path,
                property_id=ga_property_id,
                lookback_days=lookback_days,
            )
        except Exception as exc:  # pragma: no cover - API failure
            logger.warning("Failed to fetch GA4 analytics: %s", exc)
    else:
        logger.debug("GA property ID not configured; skipping GA4 pull.")

    if gsc_site_url:
        try:
            gsc_data = fetch_search_console(
                credentials_path=lookup_path,
                site_url=gsc_site_url,
                lookback_days=lookback_days,
            )
        except Exception as exc:  # pragma: no cover - API failure
            logger.warning("Failed to fetch Search Console analytics: %s", exc)
    else:
        logger.debug("Search Console site URL not configured; skipping Search Console pull.")

    if not ga_data and not gsc_data:
        return None

    return AnalyticsSnapshot(
        generated_at=datetime.utcnow(),
        lookback_days=lookback_days,
        ga_property_id=ga_property_id,
        gsc_site_url=gsc_site_url,
        ga_top_pages=ga_data.get("pages", []),
        ga_locations=ga_data.get("locations", []),
        ga_events=ga_data.get("events", []),
        gsc_queries=gsc_data.get("queries", []),
        gsc_pages=gsc_data.get("pages", []),
    )


