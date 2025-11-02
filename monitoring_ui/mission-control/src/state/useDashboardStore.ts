import { create } from 'zustand';
import {
  AgentPerformanceSnapshot,
  DashboardSummary,
  PipelineAgent,
  PlanStep,
  QualityMetrics,
  SeverityLevel,
  SnapshotPayload,
  TelemetryEvent,
} from '../types';
import { normalizeSnapshot, resolveSeverity } from '../services/normalize';
import type { SocketStatus } from '../services/socket';

const EVENT_LIMIT = 200;
const LAYOUT_STORAGE_KEY = 'mission-control-layout';

type LayoutState = {
  showPlan: boolean;
  lastDocument?: string | null;
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
    // ignore
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

const defaultQuality: QualityMetrics = {
  precisionAt3: undefined,
  recallAt3: undefined,
  hallucinationRate: undefined,
  faithfulness: undefined,
  citationVerified: undefined,
  validationFailures: undefined,
  emptyFields: undefined,
  layerFailures: [],
  lastEvaluatedAt: null,
};

const HISTORY_LIMIT = 36;

const cloneHistory = (input: Record<string, AgentPerformanceSnapshot[]>): Record<string, AgentPerformanceSnapshot[]> => {
  return Object.fromEntries(Object.entries(input).map(([key, value]) => [key, value.slice()]));
};

const computeBaseline = (points: AgentPerformanceSnapshot[]): number | undefined => {
  if (points.length === 0) return undefined;
  const valid = points.map((p) => p.avgTimeSeconds ?? 0).filter((value) => value > 0);
  if (!valid.length) return undefined;
  const sum = valid.reduce((acc, value) => acc + value, 0);
  return sum / valid.length;
};

const computeInsights = (
  agents: PipelineAgent[],
  quality: QualityMetrics,
  history: Record<string, AgentPerformanceSnapshot[]>,
): string[] => {
  const insights: string[] = [];

  agents.forEach((agent) => {
    if (!agent.avgTimeSeconds) {
      return;
    }
    const points = history[agent.id] ?? [];
    const previous = points.slice(0, -1);
    const baseline = computeBaseline(previous);
    if (baseline && baseline > 0) {
      const delta = agent.avgTimeSeconds - baseline;
      const ratio = delta / baseline;
      if (ratio >= 0.3) {
        insights.push(`⚠️ ${agent.name} latency +${Math.round(ratio * 100)}% vs. baseline`);
      }
    }
    if ((agent.failures ?? 0) > 0) {
      insights.push(`❗ ${agent.name} reported ${agent.failures} failures on recent runs`);
    }
  });

  if ((quality.hallucinationRate ?? 0) > 0.1) {
    insights.push(`🧠 Hallucination rate elevated at ${(quality.hallucinationRate! * 100).toFixed(1)}%`);
  }
  if ((quality.validationFailures ?? 0) > 0) {
    insights.push(`🚫 ${quality.validationFailures} validation checks failed`);
  }
  if (quality.layerFailures.length) {
    quality.layerFailures.forEach((failure) => insights.push(`❌ ${failure}`));
  }

  return insights.slice(0, 5);
};

export interface DashboardState {
  connection: SocketStatus;
  summary: DashboardSummary;
  agents: PipelineAgent[];
  plan: PlanStep[];
  events: TelemetryEvent[];
  quality: QualityMetrics;
  performanceHistory: Record<string, AgentPerformanceSnapshot[]>;
  insights: string[];
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

export const useDashboardStore = create<DashboardState>((set, get) => ({
  connection: 'connecting',
  summary: defaultSummary,
  agents: [],
  plan: [],
  events: [],
  quality: defaultQuality,
  performanceHistory: {},
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
        severity: resolveSeverity(normalized.summary.severity, layout.lastDocument && layout.lastDocument !== normalized.summary.currentDocument ? 'warning' : undefined),
      };
      const mergedSummary = {
        ...summary,
        currentDocument: normalized.summary.currentDocument,
      };

      const updatedLayout: LayoutState = {
        ...layout,
        lastDocument: normalized.summary.currentDocument ?? layout.lastDocument ?? null,
      };

      persistLayout(updatedLayout);

      const now = new Date().toISOString();
      const existingHistory = cloneHistory(get().performanceHistory);
      normalized.agents.forEach((agent) => {
        const entry: AgentPerformanceSnapshot = {
          timestamp: now,
          avgTimeSeconds: agent.avgTimeSeconds,
          executions: agent.executions,
          successes: agent.successes,
          failures: agent.failures,
        };
        const queue = existingHistory[agent.id] ?? [];
        queue.push(entry);
        if (queue.length > HISTORY_LIMIT) {
          queue.splice(0, queue.length - HISTORY_LIMIT);
        }
        existingHistory[agent.id] = queue;
      });

      const insights = computeInsights(normalized.agents, normalized.quality, existingHistory);

      set({
        summary: mergedSummary,
        agents: normalized.agents,
        plan: normalized.plan,
        quality: normalized.quality,
        performanceHistory: existingHistory,
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
        summary: { ...get().summary, severity, alertMessages: event.severity !== 'normal' ? [event.message] : get().summary.alertMessages },
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
      set({ summary: defaultSummary, agents: [], plan: [], events: [], performanceHistory: {}, quality: defaultQuality, insights: [] });
    },
  },
}));
