import type { DashboardSummary, PipelineAgent, QualityMetrics } from '../types';

export interface AssistantContextPayload {
  summary: DashboardSummary;
  agents: PipelineAgent[];
  quality: QualityMetrics;
  insights: string[];
}

export const composeAssistantContext = ({ summary, agents, quality, insights }: AssistantContextPayload): Record<string, unknown> => {
  const topAgents = agents.slice(0, 6).map((agent) => ({
    name: agent.name,
    status: agent.status,
    avgTimeSeconds: agent.avgTimeSeconds ?? null,
    executions: agent.executions ?? 0,
    failures: agent.failures ?? 0,
  }));

  return {
    document: summary.currentDocument,
    progress: summary.progressPercentage,
    errors: summary.errors,
    warnings: summary.warnings,
    quality: {
      faithfulness: quality.faithfulness ?? null,
      hallucinationRate: quality.hallucinationRate ?? null,
      precisionAt3: quality.precisionAt3 ?? null,
      recallAt3: quality.recallAt3 ?? null,
      validationFailures: quality.validationFailures ?? 0,
      layerFailures: quality.layerFailures,
    },
    agents: topAgents,
    insights,
  };
};

