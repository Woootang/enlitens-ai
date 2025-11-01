import {
  AgentPerformanceStat,
  DashboardSummary,
  NormalizedSnapshot,
  PipelineAgent,
  PlanStep,
  QualityMetric,
  QualityState,
  SnapshotPayload,
  SeverityLevel,
} from '../types';

const STATUS_PRIORITY: Record<SeverityLevel, number> = {
  normal: 0,
  warning: 1,
  critical: 2,
};

const AGENT_STATUS_ORDER: Record<string, SeverityLevel> = {
  failed: 'critical',
  running: 'warning',
  completed: 'normal',
  idle: 'normal',
};

const sanitizeId = (value: string) => value.toLowerCase().replace(/[^a-z0-9]+/g, '-');

const humanize = (value: string) =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .replace(/\s+/g, ' ')
    .trim();

const metricOrientation = (metricId: string) => {
  const lower = metricId.toLowerCase();
  if (lower.includes('rate') || lower.includes('error') || lower.includes('hallucination')) {
    return 'lower_is_better' as const;
  }
  return 'higher_is_better' as const;
};

const parseNumeric = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  if (typeof value === 'object' && value !== null && 'value' in value) {
    return parseNumeric((value as { value: unknown }).value);
  }
  return null;
};

const formatMetricValue = (metricId: string, raw: number | null): string => {
  if (raw === null || Number.isNaN(raw)) {
    return '—';
  }
  const normalizedId = metricId.toLowerCase();
  if (normalizedId.includes('rate') || normalizedId.includes('percentage') || raw <= 1) {
    return `${(raw * 100).toFixed(1)}%`;
  }
  if (normalizedId.includes('score') || normalizedId.includes('faithfulness')) {
    return raw.toFixed(2);
  }
  return raw.toFixed(2);
};

export const resolveSeverity = (
  base: SeverityLevel,
  candidate?: SeverityLevel,
): SeverityLevel => {
  if (!candidate) {
    return base;
  }
  const weight = STATUS_PRIORITY[candidate];
  return weight > STATUS_PRIORITY[base] ? candidate : base;
};

export function buildSummary(payload: SnapshotPayload): DashboardSummary {
  const recentErrors = payload.recent_errors ?? [];
  const recentWarnings = payload.recent_warnings ?? [];

  let severity: SeverityLevel = 'normal';
  if (recentErrors.length > 0) {
    severity = resolveSeverity(severity, 'critical');
  } else if (recentWarnings.length > 0) {
    severity = resolveSeverity(severity, 'warning');
  }

  const alertMessages: string[] = [];
  if (recentErrors.length) {
    alertMessages.push(recentErrors[recentErrors.length - 1].message);
  } else if (recentWarnings.length) {
    alertMessages.push(recentWarnings[recentWarnings.length - 1].message);
  }

  return {
    currentDocument: payload.current_document ?? null,
    documentsProcessed: payload.documents_processed ?? 0,
    totalDocuments: payload.total_documents ?? 0,
    progressPercentage: payload.progress_percentage ?? 0,
    timeOnDocumentSeconds: payload.time_on_document_seconds ?? null,
    lastLogSecondsAgo: payload.last_log_seconds_ago ?? null,
    severity,
    alertMessages,
    errors: recentErrors.length,
    warnings: recentWarnings.length,
  };
}

export function buildPipelineAgents(payload: SnapshotPayload): PipelineAgent[] {
  const pipeline = payload.agent_pipeline ?? [];
  const statusMap = payload.agent_status ?? {};
  const performance = payload.agent_performance ?? {};

  return pipeline.map((agentName, index) => {
    const normalizedId = sanitizeId(agentName || `agent-${index}`);
    const agentStatusRaw = statusMap[agentName] ?? 'unknown';
    const performanceData = performance[agentName] ?? {};

    return {
      id: normalizedId,
      name: agentName,
      status: (agentStatusRaw as PipelineAgent['status']) ?? 'unknown',
      executions: performanceData.executions ?? 0,
      avgTimeSeconds: performanceData.avg_time ?? undefined,
      successes: performanceData.successes ?? undefined,
      failures: performanceData.failures ?? undefined,
      isActive: agentStatusRaw === 'running',
    };
  });
}

export function buildPerformanceStats(payload: SnapshotPayload): AgentPerformanceStat[] {
  const performance = payload.agent_performance ?? {};
  const pipeline = payload.agent_pipeline ?? [];
  const knownAgents = new Set(pipeline);
  const orderedNames = [...pipeline, ...Object.keys(performance).filter((name) => !knownAgents.has(name))];

  return orderedNames.map((name, index) => {
    const normalizedId = sanitizeId(name || `performance-${index}`);
    const data = performance[name] ?? {};
    const executions = data.executions ?? 0;
    const successes = data.successes ?? 0;
    const failures = data.failures ?? 0;
    const avgTimeSeconds = typeof data.avg_time === 'number' ? data.avg_time : null;
    const successRate = executions > 0 ? successes / executions : null;
    const failureRate = executions > 0 ? failures / executions : null;

    return {
      id: normalizedId,
      name,
      avgTimeSeconds,
      executions,
      successes,
      failures,
      successRate,
      failureRate,
    };
  });
}

export function buildQualityState(payload: SnapshotPayload): QualityState {
  const raw = payload.quality_metrics ?? {};
  const metrics: QualityMetric[] = [];
  let layerFailures: string[] = [];

  Object.entries(raw).forEach(([key, value]) => {
    if (key.toLowerCase().includes('layer_failures') && Array.isArray(value)) {
      layerFailures = value.filter((entry): entry is string => typeof entry === 'string');
      return;
    }

    const numericValue = parseNumeric(value);
    const orientation = metricOrientation(key);
    metrics.push({
      id: sanitizeId(key),
      label: humanize(key),
      value: numericValue,
      displayValue: formatMetricValue(key, numericValue),
      orientation,
      severity: 'normal',
      baseline: null,
    });
  });

  metrics.sort((a, b) => a.label.localeCompare(b.label));

  return { metrics, layerFailures };
}

const deriveStepStatus = (index: number, total: number): PlanStep['status'] => {
  if (total === 0) return 'pending';
  if (index < total - 1) return 'completed';
  if (index === total - 1) return 'in_progress';
  return 'pending';
};

export function buildPlan(payload: SnapshotPayload, agents: PipelineAgent[]): PlanStep[] {
  const stack = payload.supervisor_stack ?? [];
  const total = stack.length;
  const agentIndexByName = new Map(agents.map((agent) => [agent.name.toLowerCase(), agent.id]));

  return stack.map((title, index) => {
    const relatedAgentId = agentIndexByName.get(title.toLowerCase());
    return {
      id: `stage-${index}`,
      title,
      status: deriveStepStatus(index, total),
      relatedAgentId,
    };
  });
}

export function reduceSeverityFromAgents(agents: PipelineAgent[], seed: SeverityLevel): SeverityLevel {
  return agents.reduce((acc, agent) => {
    const candidate = AGENT_STATUS_ORDER[agent.status] ?? 'normal';
    return resolveSeverity(acc, candidate);
  }, seed);
}

export function normalizeSnapshot(payload: SnapshotPayload): NormalizedSnapshot {
  const timestamp = new Date().toISOString();
  const summary = buildSummary(payload);
  const agents = buildPipelineAgents(payload);
  const performance = buildPerformanceStats(payload);
  const plan = buildPlan(payload, agents);
  const quality = buildQualityState(payload);
  const severityWithAgents = reduceSeverityFromAgents(agents, summary.severity);

  return {
    summary: { ...summary, severity: severityWithAgents },
    agents,
    plan,
    performance,
    quality,
    timestamp,
  };
}
