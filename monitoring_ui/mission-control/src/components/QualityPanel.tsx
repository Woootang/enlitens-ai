import { Chip, LinearProgress, Paper, Stack, Typography } from '@mui/material';
import { QualityMetric } from '../types';

interface QualityPanelProps {
  metrics: QualityMetric[];
  layerFailures: string[];
}

const severityPalette: Record<QualityMetric['severity'], { bar: string; background: string; text: string }> = {
  normal: { bar: '#34d399', background: 'rgba(52,211,153,0.12)', text: '#a7f3d0' },
  warning: { bar: '#facc15', background: 'rgba(250,204,21,0.15)', text: '#fef3c7' },
  critical: { bar: '#f87171', background: 'rgba(248,113,113,0.18)', text: '#fee2e2' },
};

const formatBaseline = (metric: QualityMetric) => {
  if (metric.baseline === null || metric.baseline === undefined || Number.isNaN(metric.baseline)) {
    return null;
  }
  const value = metric.baseline;
  if (value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(2);
};

const normalizeValue = (metric: QualityMetric) => {
  if (metric.value === null || metric.value === undefined) {
    return 0;
  }
  if (metric.orientation === 'lower_is_better') {
    const value = Math.min(Math.max(metric.value * 100, 0), 100);
    return 100 - value;
  }
  const numeric = metric.value <= 1 ? metric.value * 100 : Math.min(metric.value, 100);
  return Math.max(Math.min(numeric, 100), 0);
};

export const QualityPanel = ({ metrics, layerFailures }: QualityPanelProps) => {
  return (
    <Paper sx={{ p: 4, borderRadius: 3, border: '1px solid rgba(255,255,255,0.12)', backgroundColor: 'rgba(17,24,39,0.78)' }}>
      <Stack spacing={3}>
        <Stack spacing={0.5}>
          <Typography variant="overline" letterSpacing={2} color="primary.light">
            Quality Metrics
          </Typography>
          <Typography variant="subtitle2" color="text.secondary">
            Faithfulness, alignment, and citation integrity checks
          </Typography>
        </Stack>

        <Stack spacing={2}>
          {metrics.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              Awaiting quality telemetry…
            </Typography>
          )}
          {metrics.map((metric) => {
            const palette = severityPalette[metric.severity];
            const baseline = formatBaseline(metric);
            return (
              <Stack key={metric.id} spacing={1.25}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Stack spacing={0.5}>
                    <Typography variant="subtitle2" color="text.primary">
                      {metric.label}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {metric.orientation === 'lower_is_better' ? 'Lower is better' : 'Higher is better'}
                    </Typography>
                  </Stack>
                  <Stack spacing={0.25} alignItems="flex-end">
                    <Typography variant="h6" color={palette.text}>
                      {metric.displayValue}
                    </Typography>
                    {baseline && (
                      <Typography variant="caption" color="text.secondary">
                        Baseline {metric.orientation === 'lower_is_better' ? '≤' : '≥'} {baseline}
                      </Typography>
                    )}
                  </Stack>
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={normalizeValue(metric)}
                  sx={{
                    height: 10,
                    borderRadius: 999,
                    backgroundColor: palette.background,
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 999,
                      backgroundColor: palette.bar,
                    },
                  }}
                />
              </Stack>
            );
          })}
        </Stack>

        {layerFailures.length > 0 && (
          <Stack spacing={1}>
            <Typography variant="subtitle2" color="text.secondary">
              Validation findings
            </Typography>
            <Stack direction="row" flexWrap="wrap" gap={1}>
              {layerFailures.map((failure, index) => (
                <Chip key={`${failure}-${index}`} label={failure} color="warning" variant="outlined" size="small" />
              ))}
            </Stack>
          </Stack>
        )}
      </Stack>
    </Paper>
  );
};
