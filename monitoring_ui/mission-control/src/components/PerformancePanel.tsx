import { Card, CardContent, Stack, Typography, Box, Divider } from '@mui/material';
import { useMemo } from 'react';
import { useDashboardStore } from '../state/useDashboardStore';
import type { AgentPerformanceSnapshot, PipelineAgent } from '../types';

const MAX_HISTORY_POINTS = 16;

const Sparkline = ({ points }: { points: AgentPerformanceSnapshot[] }) => {
  const trimmed = points.slice(-MAX_HISTORY_POINTS);
  if (trimmed.length < 2) {
    return <Box height={32} />;
  }
  const values = trimmed.map((point) => point.avgTimeSeconds ?? 0);
  const fallback = values.length ? values[0] : 0;
  const max = Math.max(...values, fallback + 0.01);
  const min = Math.min(...values, fallback);
  const width = 120;
  const height = 32;
  const range = Math.max(max - min, 0.001);
  const scaleX = (index: number) => (index / (trimmed.length - 1)) * width;
  const scaleY = (value: number) => height - ((value - min) / range) * height;
  const path = trimmed
    .map((point, idx) => `${idx === 0 ? 'M' : 'L'}${scaleX(idx)},${scaleY(point.avgTimeSeconds ?? 0)}`)
    .join(' ');

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Latency trend sparkline">
      <path d={path} fill="none" stroke="rgba(99,179,237,0.8)" strokeWidth={2} strokeLinecap="round" />
    </svg>
  );
};

type AgentMetric = {
  agent: PipelineAgent;
  avgTime: number;
  successes: number;
  failures: number;
  history: AgentPerformanceSnapshot[];
};

const SuccessFailureBar = ({ success, failure, max }: { success: number; failure: number; max: number }) => {
  if (max <= 0) {
    return <Box height={10} borderRadius={999} bgcolor="rgba(255,255,255,0.08)" />;
  }
  const successWidth = (success / max) * 100;
  const failureWidth = (failure / max) * 100;
  return (
    <Box height={10} borderRadius={999} bgcolor="rgba(255,255,255,0.06)" overflow="hidden" display="flex">
      <Box flexBasis={`${successWidth}%`} bgcolor="rgba(72,187,120,0.6)" />
      <Box flexBasis={`${failureWidth}%`} bgcolor="rgba(245,101,101,0.6)" />
    </Box>
  );
};

export const PerformancePanel = () => {
  const agents = useDashboardStore((state) => state.agents);
  const history = useDashboardStore((state) => state.performanceHistory);

  const metrics = useMemo<AgentMetric[]>(() => {
    return agents.map((agent) => ({
      agent,
      avgTime: agent.avgTimeSeconds ?? 0,
      successes: agent.successes ?? 0,
      failures: agent.failures ?? 0,
      history: history[agent.id] ?? [],
    }));
  }, [agents, history]);

  const maxLatency = useMemo(() => Math.max(...metrics.map((metric) => metric.avgTime), 0), [metrics]);
  const maxRuns = useMemo(
    () => Math.max(...metrics.map((metric) => metric.successes + metric.failures), 0),
    [metrics],
  );

  if (!metrics.length) {
    return (
      <Card sx={{ height: '100%', border: '1px solid rgba(255,255,255,0.2)', backgroundColor: 'rgba(15,23,42,0.65)' }}>
        <CardContent sx={{ py: 8 }}>
          <Typography align="center" color="text.secondary">
            Waiting for agent performance telemetry…
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const sortedMetrics = [...metrics].sort((a, b) => (b.avgTime || 0) - (a.avgTime || 0));

  return (
    <Card sx={{ border: '1px solid rgba(255,255,255,0.2)', backgroundColor: 'rgba(15,23,42,0.65)', height: '100%' }}>
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="h6" color="text.primary">
            Agent Performance
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Avg latency & run outcomes (latest window)
          </Typography>
        </Stack>
        <Divider light sx={{ borderColor: 'rgba(255,255,255,0.08)' }} />
        <Stack spacing={2}>
          {sortedMetrics.map((metric, index) => {
            const latencyWidth = maxLatency > 0 ? Math.min((metric.avgTime / maxLatency) * 100, 100) : 0;
            const totalRuns = metric.successes + metric.failures;
            const maxRunsBase = maxRuns > 0 ? maxRuns : 1;
            return (
              <Stack key={metric.agent.id} spacing={1.5}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography fontWeight={600}>{metric.agent.name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {metric.avgTime ? `${metric.avgTime.toFixed(2)}s avg` : 'n/a'}
                  </Typography>
                </Stack>
                <Box height={10} borderRadius={999} bgcolor="rgba(255,255,255,0.08)" overflow="hidden">
                  <Box
                    height="100%"
                    width={`${latencyWidth}%`}
                    sx={{
                      background: 'linear-gradient(90deg, rgba(99,179,237,0.8), rgba(59,130,246,0.9))',
                      transition: 'width 200ms ease',
                    }}
                  />
                </Box>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="caption" color="text.secondary">
                    Success/Fail
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {metric.successes}/{metric.failures} ({totalRuns} runs)
                  </Typography>
                </Stack>
                <SuccessFailureBar success={metric.successes} failure={metric.failures} max={maxRunsBase} />
                <Sparkline points={metric.history} />
                {index !== sortedMetrics.length - 1 && <Divider light sx={{ borderColor: 'rgba(255,255,255,0.04)' }} />}
              </Stack>
            );
          })}
        </Stack>
      </CardContent>
    </Card>
  );
};

