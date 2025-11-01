import { create } from 'zustand';
import {
  AgentPerformanceStat,
  DashboardInsight,
  DashboardSummary,
  PipelineAgent,
  PlanStep,
  PerformanceTrendPoint,
  QualityMetric,
  SeverityLevel,
  SnapshotPayload,
  TelemetryEvent,
} from '../types';
import { normalizeSnapshot, resolveSeverity } from '../services/normalize';
import type { SocketStatus } from '../services/socket';

const EVENT_LIMIT = 200;
const METRIC_HISTORY_LIMIT = 120;
const TREND_HISTORY_LIMIT = 180;
const LAYOUT_STORAGE_KEY = 'mission-control-layout';

type LayoutState = {
  showPlan: boolean;
  lastDocument?: string | null;
};

type HistoryPoint = { timestamp: string; value: number | null };

type PerformanceState = {
  stats: AgentPerformanceStat[];
  history: Record<string, HistoryPoint[]>;
  trend: PerformanceTrendPoint[];
};

type QualityPanelState = {
  metrics: QualityMetric[];
  layerFailures: string[];
  history: Record<string, HistoryPoint[]>;
};

const loadLayout = (): LayoutState => {
  if (typeof window === 'undefined') {
    return { showPlan: true };
  }
  try {
    const stored = window.localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!stored) {
      return { showPlan: true };
    }
    return JSON.parse(stored) as LayoutState;
  } catch {
    return { showPlan: true };
  }
};

const persistLayout = (layout: LayoutState) => {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout));
  } catch {
    // ignore persistence errors silently
  }
};

const defaultSummary: DashboardSummary = {
  currentDocument: null,
  documentsProcessed: 0,
  totalDocuments: 0,
  progressPercentage: 0,
  timeOnDocumentSeconds: null,
  lastLogSecondsAgo: null,
  severity: 'normal',
  alertMessages: [],
  errors: 0,
  warnings: 0,
};

const defaultPerformance: PerformanceState = {
  stats: [],
  history: {},
  trend: [],
};

const defaultQuality: QualityPanelState = {
  metrics: [],
  layerFailures: [],
  history: {},
};

const severityFromLevel = (level?: string): SeverityLevel => {
  if (!level) return 'normal';
  if (level.toUpperCase() === 'ERROR') return 'critical';
  if (level.toUpperCase() === 'WARNING') return 'warning';
  return 'normal';
};

const normalizeEvent = (raw: any): TelemetryEvent => {
  const level = (raw?.level ?? 'INFO').toUpperCase();
  const severity = severityFromLevel(level);
  return {
    id: raw?.id ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    timestamp: raw?.timestamp ?? new Date().toISOString(),
    message: raw?.message ?? JSON.stringify(raw),
    agentName: raw?.agent_name ?? raw?.agentName,
    level: level as TelemetryEvent['level'],
    severity,
  };
};

const median = (values: number[]): number | null => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
  return sorted[middle];
};

const computeBaseline = (history: HistoryPoint[]): number | null => {
  const numeric = history.map((point) => point.value).filter((value): value is number => typeof value === 'number');
  if (numeric.length < 3) {
    return numeric.length ? numeric[numeric.length - 1] : null;
  }
  return median(numeric);
};

const evaluateQualitySeverity = (
  metric: QualityMetric,
  history: HistoryPoint[],
): { severity: SeverityLevel; baseline: number | null } => {
  const baseline = computeBaseline(history);
  const value = metric.value;
  if (value === null || Number.isNaN(value)) {
    return { severity: 'normal', baseline };
  }

  const orientation = metric.orientation;
  if (baseline === null || Number.isNaN(baseline)) {
    if (orientation === 'lower_is_better') {
      if (value <= 0.1) return { severity: 'normal', baseline: null };
      if (value <= 0.18) return { severity: 'warning', baseline: null };
      return { severity: 'critical', baseline: null };
    }
    if (value >= 0.9) return { severity: 'normal', baseline: null };
    if (value >= 0.75) return { severity: 'warning', baseline: null };
    return { severity: 'critical', baseline: null };
  }

  if (orientation === 'lower_is_better') {
    const tolerance = Math.max(baseline * 0.35, 0.05);
    const diff = value - baseline;
    if (diff <= tolerance * 0.5) {
      return { severity: 'normal', baseline };
    }
    if (diff <= tolerance) {
      return { severity: 'warning', baseline };
    }
    return { severity: 'critical', baseline };
  }

  const tolerance = Math.max(baseline * 0.25, 0.08);
  const diff = baseline - value;
  if (diff <= tolerance * 0.5) {
    return { severity: 'normal', baseline };
  }
  if (diff <= tolerance) {
    return { severity: 'warning', baseline };
  }
  return { severity: 'critical', baseline };
};

const formatInsight = (message: string, severity: SeverityLevel | 'info'): DashboardInsight => ({
  id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
  message,
  severity,
  timestamp: new Date().toISOString(),
});

const buildPerformanceInsights = (
  stats: AgentPerformanceStat[],
  history: Record<string, HistoryPoint[]>,
): DashboardInsight[] => {
  const insights: DashboardInsight[] = [];
  stats.forEach((stat) => {
    const historyPoints = history[stat.id] ?? [];
    const previous = historyPoints.slice(0, -1).filter((point) => typeof point.value === 'number') as Array<{ value: number }>;
    if (!previous.length || stat.avgTimeSeconds === null || stat.avgTimeSeconds === undefined) {
      return;
    }
    const baseline = median(previous.map((point) => point.value));
    if (!baseline || baseline <= 0) {
      return;
    }
    const delta = stat.avgTimeSeconds - baseline;
    const ratio = delta / baseline;
    if (ratio > 0.3) {
      const severity = ratio > 0.6 ? 'critical' : 'warning';
      insights.push(
        formatInsight(
          `${stat.name} latency up ${Math.round(ratio * 100)}% vs. recent median (${stat.avgTimeSeconds.toFixed(2)}s)`,
          severity,
        ),
      );
    }
    if (stat.failureRate && stat.failureRate >= 0.2) {
      insights.push(
        formatInsight(
          `${stat.name} failure rate at ${(stat.failureRate * 100).toFixed(1)}% over ${stat.executions} runs`,
          stat.failureRate > 0.35 ? 'critical' : 'warning',
        ),
      );
    }
  });
  return insights;
};

const formatBaselineDisplay = (metric: QualityMetric): string | null => {
  if (metric.baseline === null || metric.baseline === undefined || Number.isNaN(metric.baseline)) {
    return null;
  }
  const value = metric.baseline;
  if (value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(2);
};

const buildQualityInsights = (metrics: QualityMetric[]): DashboardInsight[] =>
  metrics
    .filter((metric) => metric.severity !== 'normal')
    .map((metric) => {
      const baselineDisplay = formatBaselineDisplay(metric);
      const baselineText = baselineDisplay
        ? ` • baseline ${metric.orientation === 'lower_is_better' ? '≤' : '≥'} ${baselineDisplay}`
        : '';
      return formatInsight(`${metric.label} flagged (${metric.displayValue}${baselineText})`, metric.severity);
    });

const mergeInsights = (...groups: DashboardInsight[][]): DashboardInsight[] => {
  const seen = new Set<string>();
  const merged: DashboardInsight[] = [];
  groups.flat().forEach((insight) => {
    if (!insight.message || seen.has(insight.message)) {
      return;
    }
    seen.add(insight.message);
    merged.push(insight);
  });
  return merged.slice(0, 5);
};

export interface DashboardState {
  connection: SocketStatus;
  summary: DashboardSummary;
  agents: PipelineAgent[];
  plan: PlanStep[];
  events: TelemetryEvent[];
  performance: PerformanceState;
  quality: QualityPanelState;
  insights: DashboardInsight[];
  layout: LayoutState;
  highlightAgentId?: string;
  highlightPlanStepId?: string;
  lastUpdated?: string;
  actions: {
    setConnection(status: SocketStatus): void;
    ingestSnapshot(payload: SnapshotPayload): void;
    ingestEvent(event: TelemetryEvent): void;
    togglePlanVisibility(): void;
    setHighlightAgent(agentId?: string): void;
    setHighlightPlanStep(stepId?: string, relatedAgentId?: string): void;
    softReset(): void;
  };
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  connection: 'connecting',
  summary: defaultSummary,
  agents: [],
  plan: [],
  events: [],
  performance: defaultPerformance,
  quality: defaultQuality,
  insights: [],
  layout: loadLayout(),
  highlightAgentId: undefined,
  highlightPlanStepId: undefined,
  lastUpdated: undefined,
  actions: {
    setConnection(status) {
      set({ connection: status });
    },
    ingestSnapshot(payload) {
      const normalized = normalizeSnapshot(payload);
      const layout = get().layout;

      const summary = {
        ...normalized.summary,
        severity: resolveSeverity(
          normalized.summary.severity,
          layout.lastDocument && layout.lastDocument !== normalized.summary.currentDocument ? 'warning' : undefined,
        ),
      };

      const performanceHistory = { ...get().performance.history };
      const trendHistory = [...get().performance.trend];
      const qualityHistory = { ...get().quality.history };

      normalized.performance.forEach((stat) => {
        const history = performanceHistory[stat.id] ?? [];
        const updated = [...history, { timestamp: normalized.timestamp, value: stat.avgTimeSeconds ?? null }];
        performanceHistory[stat.id] = updated.slice(-METRIC_HISTORY_LIMIT);
      });

      const aggregateLatency = (() => {
        const valid = normalized.performance
          .map((stat) => stat.avgTimeSeconds)
          .filter((value): value is number => typeof value === 'number');
        if (!valid.length) return null;
        return valid.reduce((acc, value) => acc + value, 0) / valid.length;
      })();

      trendHistory.push({
        timestamp: normalized.timestamp,
        aggregateLatency,
        documentsProcessed: normalized.summary.documentsProcessed,
      });
      const trimmedTrend = trendHistory.slice(-TREND_HISTORY_LIMIT);

      const enrichedMetrics: QualityMetric[] = normalized.quality.metrics.map((metric) => {
        const history = qualityHistory[metric.id] ?? [];
        const { severity, baseline } = evaluateQualitySeverity(metric, history);
        const updatedHistory = [...history, { timestamp: normalized.timestamp, value: metric.value }].slice(
          -METRIC_HISTORY_LIMIT,
        );
        qualityHistory[metric.id] = updatedHistory;
        return { ...metric, severity, baseline };
      });

      const qualitySeverity = enrichedMetrics.reduce<SeverityLevel>(
        (acc, metric) => resolveSeverity(acc, metric.severity),
        'normal',
      );

      const mergedSummary: DashboardSummary = {
        ...summary,
        severity: resolveSeverity(summary.severity, qualitySeverity),
        currentDocument: normalized.summary.currentDocument,
      };

      const performanceInsights = buildPerformanceInsights(normalized.performance, performanceHistory);
      const qualityInsights = buildQualityInsights(enrichedMetrics);
      const insights = mergeInsights(performanceInsights, qualityInsights);

      const updatedLayout: LayoutState = {
        ...layout,
        lastDocument: normalized.summary.currentDocument ?? layout.lastDocument ?? null,
      };

      persistLayout(updatedLayout);

      set({
        summary: mergedSummary,
        agents: normalized.agents,
        plan: normalized.plan,
        performance: {
          stats: normalized.performance,
          history: performanceHistory,
          trend: trimmedTrend,
        },
        quality: {
          metrics: enrichedMetrics,
          layerFailures: normalized.quality.layerFailures,
          history: qualityHistory,
        },
        insights,
        layout: updatedLayout,
        lastUpdated: new Date().toISOString(),
      });
    },
    ingestEvent(raw) {
      const event = normalizeEvent(raw);
      const current = get().events;
      const events = [event, ...current].slice(0, EVENT_LIMIT);
      const severity = resolveSeverity(get().summary.severity, event.severity);
      set({
        events,
        summary: {
          ...get().summary,
          severity,
          alertMessages: event.severity !== 'normal' ? [event.message] : get().summary.alertMessages,
        },
      });
    },
    togglePlanVisibility() {
      const layout = get().layout;
      const updated = { ...layout, showPlan: !layout.showPlan };
      persistLayout(updated);
      set({ layout: updated });
    },
    setHighlightAgent(agentId) {
      set({ highlightAgentId: agentId });
    },
    setHighlightPlanStep(stepId, relatedAgentId) {
      set({
        highlightPlanStepId: stepId,
        highlightAgentId: relatedAgentId ?? (stepId ? get().highlightAgentId : undefined),
      });
    },
    softReset() {
      set({
        summary: defaultSummary,
        agents: [],
        plan: [],
        events: [],
        performance: defaultPerformance,
        quality: defaultQuality,
        insights: [],
      });
    },
  },
}));
