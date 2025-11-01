import { useEffect } from 'react';
import { useDashboardStore } from '../state/useDashboardStore';
import { getTelemetryUrl } from '../services/api';
import { computeInsightSeverity, normalizeAgents, normalizePlan, normalizePerformance, normalizeQuality, normalizeSummary } from '../services/normalize';
import { TelemetrySocket, TelemetryMessage } from '../services/socket';
import { InsightMessage, PerformanceTrendPoint } from '../types';

const performanceTrend: PerformanceTrendPoint[] = [];

export const useTelemetryConnection = () => {
  const {
    actions: {
      upsertAgents,
      updateSummary,
      updatePlan,
      updatePerformance,
      updateQuality,
      pushInsights,
      setConnection,
      setHighlightAgentId,
    },
  } = useDashboardStore();

  useEffect(() => {
    const socket = new TelemetrySocket({
      url: getTelemetryUrl(),
      onMessage: (message) => handleMessage(message),
      onStatusChange: (status) => setConnection({ status, lastHeartbeat: new Date().toISOString() }),
    });
    socket.connect();
    return () => socket.disconnect();
  }, [setConnection, upsertAgents, updateSummary, updatePlan, updatePerformance, updateQuality, pushInsights]);

  const handleMessage = (message: TelemetryMessage) => {
    switch (message.type) {
      case 'summary':
        updateSummary(normalizeSummary(message.payload as any));
        break;
      case 'agents': {
        const agents = normalizeAgents(message.payload as any);
        upsertAgents(agents);
        const active = agents.find((agent) => agent.status === 'running');
        setHighlightAgentId(active?.id ?? null);
        break;
      }
      case 'plan':
        updatePlan(normalizePlan(message.payload as any));
        break;
      case 'performance': {
        const payload = message.payload as { stats: any[]; aggregate_latency?: number; timestamp?: string };
        if (payload.timestamp) {
          performanceTrend.push({ timestamp: payload.timestamp, aggregateLatency: payload.aggregate_latency });
          if (performanceTrend.length > 360) {
            performanceTrend.shift();
          }
        }
        updatePerformance(normalizePerformance(payload.stats ?? [], [...performanceTrend]));
        break;
      }
      case 'quality':
        updateQuality(normalizeQuality(message.payload as any));
        break;
      case 'insight': {
        const insights = computeInsightSeverity(message.payload as InsightMessage[]);
        pushInsights(insights);
        break;
      }
      default:
        break;
    }
  };
};
