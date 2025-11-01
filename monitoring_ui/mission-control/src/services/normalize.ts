import {
  AgentNode,
  InsightMessage,
  PerformanceSnapshot,
  PlanStep,
  QualityMetric,
  QualitySnapshot,
  SummarySnapshot,
} from '../types';

const severityRanking = ['info', 'warning', 'danger'] as const;

type RawAgentPayload = {
  id: string;
  name: string;
  status: AgentNode['status'];
  started_at?: string;
  finished_at?: string;
  duration_ms?: number;
  metadata?: Record<string, unknown>;
};

type RawPlanNode = {
  id: string;
  label: string;
  status: PlanStep['status'];
  children?: RawPlanNode[];
};

type RawPerformanceStat = {
  id: string;
  name: string;
  avg_time_seconds?: number;
  successes?: number;
  failures?: number;
};

type RawQualityMetric = {
  id: string;
  label: string;
  value: number;
  target: number;
  direction: QualityMetric['direction'];
};

export const normalizeAgents = (payload: RawAgentPayload[]): AgentNode[] =>
  payload.map((agent) => ({
    id: agent.id,
    name: agent.name,
    status: agent.status,
    startedAt: agent.started_at,
    finishedAt: agent.finished_at,
    durationMs: agent.duration_ms,
    metadata: agent.metadata,
  }));

export const normalizePlan = (payload: RawPlanNode[]): PlanStep[] =>
  payload.map((step) => ({
    id: step.id,
    label: step.label,
    status: step.status,
    children: step.children ? normalizePlan(step.children) : undefined,
  }));

export const normalizeSummary = (payload: Partial<SummarySnapshot>): SummarySnapshot => ({
  activeDocumentId: payload.activeDocumentId ?? null,
  totalDocuments: payload.totalDocuments ?? 0,
  completedToday: payload.completedToday ?? 0,
  failedToday: payload.failedToday ?? 0,
  severity: payload.severity ?? 'info',
  alertMessages: payload.alertMessages ?? [],
});

export const normalizePerformance = (
  stats: RawPerformanceStat[],
  trend: PerformanceSnapshot['trend'],
): PerformanceSnapshot => ({
  stats: stats.map((stat) => ({
    id: stat.id,
    name: stat.name,
    avgTimeSeconds: stat.avg_time_seconds,
    successes: stat.successes,
    failures: stat.failures,
  })),
  trend,
});

export const normalizeQuality = (payload: {
  metrics?: RawQualityMetric[];
  layer_failures?: string[];
}): QualitySnapshot => ({
  metrics:
    payload.metrics?.map((metric) => ({
      id: metric.id,
      label: metric.label,
      value: metric.value,
      target: metric.target,
      direction: metric.direction,
    })) ?? [],
  layerFailures: payload.layer_failures ?? [],
});

export const computeInsightSeverity = (messages: InsightMessage[]): InsightMessage[] =>
  messages
    .map((message) => ({ ...message, severity: message.severity ?? 'info' }))
    .sort(
      (a, b) => severityRanking.indexOf(b.severity) - severityRanking.indexOf(a.severity),
    );
