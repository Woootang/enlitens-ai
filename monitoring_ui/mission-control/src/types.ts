export type Severity = 'info' | 'warning' | 'danger';

export interface AlertMessage {
  id: string;
  text: string;
  timestamp: string;
}

export interface SummarySnapshot {
  activeDocumentId: string | null;
  totalDocuments: number;
  completedToday: number;
  failedToday: number;
  severity: Severity;
  alertMessages: AlertMessage[];
}

export interface AgentNode {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped';
  startedAt?: string;
  finishedAt?: string;
  durationMs?: number;
  metadata?: Record<string, unknown>;
}

export interface PlanStep {
  id: string;
  label: string;
  status: AgentNode['status'];
  children?: PlanStep[];
}

export interface PerformanceTrendPoint {
  timestamp: string;
  aggregateLatency?: number;
}

export interface AgentPerformanceStat {
  id: string;
  name: string;
  avgTimeSeconds?: number;
  successes?: number;
  failures?: number;
}

export interface QualityMetric {
  id: string;
  label: string;
  value: number;
  target: number;
  direction: 'higher-is-better' | 'lower-is-better';
}

export interface QualitySnapshot {
  metrics: QualityMetric[];
  layerFailures: string[];
}

export interface InsightMessage {
  id: string;
  severity: Severity;
  message: string;
  timestamp: string;
}

export interface PerformanceSnapshot {
  stats: AgentPerformanceStat[];
  trend: PerformanceTrendPoint[];
}

export interface LayoutState {
  showPlan: boolean;
}

export interface ConnectionState {
  status: 'connecting' | 'online' | 'offline';
  lastHeartbeat?: string;
}

export interface DashboardState {
  summary: SummarySnapshot;
  agents: AgentNode[];
  plan: PlanStep[];
  performance: PerformanceSnapshot;
  quality: QualitySnapshot;
  insights: InsightMessage[];
  connection: ConnectionState;
  highlightAgentId: string | null;
  layout: LayoutState;
  actions: {
    upsertAgents: (agents: AgentNode[]) => void;
    updateSummary: (summary: Partial<SummarySnapshot>) => void;
    updatePlan: (plan: PlanStep[]) => void;
    updatePerformance: (performance: PerformanceSnapshot) => void;
    updateQuality: (quality: QualitySnapshot) => void;
    pushInsights: (insights: InsightMessage[]) => void;
    setConnection: (connection: ConnectionState) => void;
    setHighlightAgentId: (id: string | null) => void;
    togglePlanVisibility: (value?: boolean) => void;
  };
}
