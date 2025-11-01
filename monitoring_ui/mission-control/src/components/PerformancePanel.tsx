import { Paper, Stack, Typography } from '@mui/material';
import { useMemo, type ReactNode } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AgentPerformanceStat, PerformanceTrendPoint } from '../types';

interface PerformancePanelProps {
  stats: AgentPerformanceStat[];
  trend: PerformanceTrendPoint[];
}

const formatSeconds = (value: number | null | undefined) => {
  if (value === null || value === undefined) return '—';
  return `${value.toFixed(2)}s`;
};

export const PerformancePanel = ({ stats, trend }: PerformancePanelProps) => {
  const latencyData = useMemo(() => {
    const sorted = [...stats]
      .filter((stat) => typeof stat.avgTimeSeconds === 'number')
      .sort((a, b) => (b.avgTimeSeconds ?? 0) - (a.avgTimeSeconds ?? 0));
    return sorted.map((stat) => ({
      name: stat.name,
      avgTime: stat.avgTimeSeconds ?? 0,
    }));
  }, [stats]);

  const successFailureData = useMemo(
    () =>
      stats.map((stat) => ({
        name: stat.name,
        successes: stat.successes ?? 0,
        failures: stat.failures ?? 0,
      })),
    [stats],
  );

  const sparkData = useMemo(() => {
    return trend.slice(-60).map((point) => ({
      time: new Date(point.timestamp).toLocaleTimeString([], { minute: '2-digit', second: '2-digit' }),
      latency: point.aggregateLatency ?? 0,
    }));
  }, [trend]);

  const aggregateLatency = useMemo(() => {
    if (!trend.length) return null;
    return trend[trend.length - 1].aggregateLatency ?? null;
  }, [trend]);

  return (
    <Paper sx={{ p: 4, borderRadius: 3, border: '1px solid rgba(255,255,255,0.12)', backgroundColor: 'rgba(13,19,33,0.75)' }}>
      <Stack spacing={3}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <div>
            <Typography variant="overline" letterSpacing={2} color="primary.light">
              Agent Performance
            </Typography>
            <Typography variant="subtitle2" color="text.secondary">
              Execution velocity and reliability by agent
            </Typography>
          </div>
          <Typography variant="h6" color="text.primary">
            Aggregate latency: {formatSeconds(aggregateLatency)}
          </Typography>
        </Stack>

        <Stack spacing={1}>
          <Typography variant="caption" color="text.secondary">
            Latency trend (most recent)
          </Typography>
          <BoxWithChart minHeight={120}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sparkData} margin={{ left: 0, right: 0, top: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" hide tickLine={false} interval={sparkData.length > 8 ? 'preserveEnd' : 0} />
                <YAxis hide domain={['auto', 'auto']} />
                <Tooltip formatter={(value: number) => `${value.toFixed(2)}s`} />
                <Area type="monotone" dataKey="latency" stroke="#38bdf8" strokeWidth={2} fill="url(#latencyGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </BoxWithChart>
        </Stack>

        <Stack spacing={3} direction={{ xs: 'column', md: 'row' }}>
          <BoxWithChart flex={1} minHeight={260}>
            <Typography variant="subtitle2" color="text.secondary" mb={2}>
              Average execution time (s)
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData} layout="vertical" margin={{ left: 60, right: 16, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis type="number" tick={{ fill: '#9ca3af' }} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fill: '#e5e7eb' }} />
                <Tooltip formatter={(value: number) => `${value.toFixed(2)}s`} />
                <Bar dataKey="avgTime" fill="#60a5fa" radius={[4, 4, 4, 4]} />
              </BarChart>
            </ResponsiveContainer>
          </BoxWithChart>

          <BoxWithChart flex={1} minHeight={260}>
            <Typography variant="subtitle2" color="text.secondary" mb={2}>
              Success vs failure counts
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={successFailureData} margin={{ left: 16, right: 16, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="name" tick={{ fill: '#e5e7eb' }} />
                <YAxis tick={{ fill: '#9ca3af' }} allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="successes" name="Successes" fill="#34d399" radius={[4, 4, 0, 0]} />
                <Bar dataKey="failures" name="Failures" fill="#f87171" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </BoxWithChart>
        </Stack>
      </Stack>
    </Paper>
  );
};

type BoxWithChartProps = {
  children: ReactNode;
  minHeight?: number;
  flex?: number;
};

const BoxWithChart = ({ children, minHeight = 200, flex }: BoxWithChartProps) => (
  <div
    style={{
      position: 'relative',
      minHeight,
      flex: flex ?? 1,
    }}
  >
    {children}
  </div>
);
