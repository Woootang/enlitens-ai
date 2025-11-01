import { create } from 'zustand';
import {
  AgentNode,
  ConnectionState,
  DashboardState,
  InsightMessage,
  PerformanceSnapshot,
  PlanStep,
  QualitySnapshot,
  SummarySnapshot,
} from '../types';

const initialSummary: SummarySnapshot = {
  activeDocumentId: null,
  totalDocuments: 0,
  completedToday: 0,
  failedToday: 0,
  severity: 'info',
  alertMessages: [],
};

const initialQuality: QualitySnapshot = {
  metrics: [],
  layerFailures: [],
};

const initialPerformance: PerformanceSnapshot = {
  stats: [],
  trend: [],
};

const initialConnection: ConnectionState = {
  status: 'connecting',
};

export const useDashboardStore = create<DashboardState>((set) => ({
  summary: initialSummary,
  agents: [],
  plan: [],
  performance: initialPerformance,
  quality: initialQuality,
  insights: [],
  connection: initialConnection,
  highlightAgentId: null,
  layout: { showPlan: true },
  actions: {
    upsertAgents: (agents: AgentNode[]) => set({ agents }),
    updateSummary: (summary: Partial<SummarySnapshot>) =>
      set((state) => ({ summary: { ...state.summary, ...summary } })),
    updatePlan: (plan: PlanStep[]) => set({ plan }),
    updatePerformance: (performance: PerformanceSnapshot) => set({ performance }),
    updateQuality: (quality: QualitySnapshot) => set({ quality }),
    pushInsights: (insights: InsightMessage[]) =>
      set((state) => ({ insights: [...insights, ...state.insights].slice(-20) })),
    setConnection: (connection: ConnectionState) => set({ connection }),
    setHighlightAgentId: (id: string | null) => set({ highlightAgentId: id }),
    togglePlanVisibility: (value?: boolean) =>
      set((state) => ({ layout: { showPlan: value ?? !state.layout.showPlan } })),
  },
}));
