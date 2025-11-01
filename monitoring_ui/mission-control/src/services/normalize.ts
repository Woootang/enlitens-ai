import { DashboardSummary, PipelineAgent, PlanStep, SnapshotPayload, SeverityLevel } from '../types';

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

export function normalizeSnapshot(payload: SnapshotPayload): { summary: DashboardSummary; agents: PipelineAgent[]; plan: PlanStep[] } {
  const summary = buildSummary(payload);
  const agents = buildPipelineAgents(payload);
  const plan = buildPlan(payload, agents);
  const severity = reduceSeverityFromAgents(agents, summary.severity);

  return {
    summary: { ...summary, severity },
    agents,
    plan,
  };
}
