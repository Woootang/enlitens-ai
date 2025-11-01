export type AgentRunStatus = 'idle' | 'running' | 'completed' | 'failed' | 'unknown';

export interface PipelineAgent {
  id: string;
  name: string;
  status: AgentRunStatus;
  executions?: number;
  avgTimeSeconds?: number;
  successes?: number;
  failures?: number;
  isActive?: boolean;
}

export type PlanStepStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'skipped';

export interface PlanStep {
  id: string;
  title: string;
  status: PlanStepStatus;
  relatedAgentId?: string;
  description?: string;
}

export type SeverityLevel = 'normal' | 'warning' | 'critical';

export interface TelemetryEvent {
  id: string;
  timestamp?: string;
  message: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  agentName?: string;
  severity: SeverityLevel;
}

export interface DashboardSummary {
  currentDocument?: string | null;
  documentsProcessed: number;
  totalDocuments: number;
  progressPercentage: number;
  timeOnDocumentSeconds?: number | null;
  lastLogSecondsAgo?: number | null;
  severity: SeverityLevel;
  alertMessages: string[];
  errors: number;
  warnings: number;
}

export interface SnapshotPayload {
  current_document?: string | null;
  documents_processed?: number;
  total_documents?: number;
  progress_percentage?: number;
  time_on_document_seconds?: number | null;
  last_log_seconds_ago?: number | null;
  agent_status?: Record<string, string>;
  agent_pipeline?: string[];
  supervisor_stack?: string[];
  recent_errors?: Array<{ message: string }>;
  recent_warnings?: Array<{ message: string }>;
  quality_metrics?: Record<string, unknown>;
  agent_performance?: Record<string, {
    executions?: number;
    successes?: number;
    failures?: number;
    avg_time?: number;
  }>;
}

export interface PipelineSnapshot {
  agents: PipelineAgent[];
  summary: DashboardSummary;
  plan: PlanStep[];
}
