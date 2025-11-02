import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Card,
  CardContent,
  LinearProgress,
  Stack,
  Typography,
  Chip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useMemo } from 'react';
import { useDashboardStore } from '../state/useDashboardStore';
import type { QualityMetrics } from '../types';

type GaugeColor = 'success' | 'warning' | 'error' | 'info';

const PROGRESS_PALETTE: Record<GaugeColor, string> = {
  success: 'linear-gradient(90deg, rgba(72,187,120,0.9), rgba(56,161,105,0.9))',
  warning: 'linear-gradient(90deg, rgba(246,173,85,0.9), rgba(217,119,6,0.9))',
  error: 'linear-gradient(90deg, rgba(245,101,101,0.9), rgba(229,62,62,0.9))',
  info: 'linear-gradient(90deg, rgba(99,179,237,0.9), rgba(59,130,246,0.9))',
};

const getProgressColor = (value: number | undefined, invert = false): GaugeColor => {
  if (value === undefined) return 'info';
  const safeValue = invert ? 1 - value : value;
  if (safeValue >= 0.85) return 'success';
  if (safeValue >= 0.6) return 'warning';
  return 'error';
};

const MetricCard = ({
  label,
  value,
  helper,
  invert,
  suffix,
}: {
  label: string;
  value?: number;
  helper?: string;
  invert?: boolean;
  suffix?: string;
}) => {
  if (value === undefined) {
    return (
      <Stack spacing={0.5}>
        <Typography variant="subtitle2" color="text.secondary">
          {label}
        </Typography>
        <Typography color="text.disabled">n/a</Typography>
      </Stack>
    );
  }
  const color = getProgressColor(value, invert);
  const percent = invert ? (1 - value) * 100 : value * 100;
  return (
    <Stack spacing={0.75}>
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography variant="subtitle2" color="text.secondary">
          {label}
        </Typography>
        <Chip label={`${percent.toFixed(1)}${suffix ?? '%'}`} size="small" color={color} variant="outlined" />
      </Stack>
      <LinearProgress
        variant="determinate"
        value={Math.max(0, Math.min(percent, 100))}
        color="inherit"
        sx={{
          height: 8,
          borderRadius: 999,
          backgroundColor: 'rgba(255,255,255,0.08)',
          '& .MuiLinearProgress-bar': {
            borderRadius: 999,
            background: PROGRESS_PALETTE[color],
          },
        }}
      />
      {helper && (
        <Typography variant="caption" color="text.secondary">
          {helper}
        </Typography>
      )}
    </Stack>
  );
};

const deriveMetrics = (quality: QualityMetrics) => {
  return [
    {
      label: 'Faithfulness',
      value: quality.faithfulness !== undefined ? Math.min(quality.faithfulness, 1) : undefined,
      helper: 'Similarity to source content',
      invert: false,
    },
    {
      label: 'Precision @3',
      value: quality.precisionAt3 !== undefined ? Math.min(quality.precisionAt3, 1) : undefined,
      helper: 'Share of relevant top-3 answers',
      invert: false,
    },
    {
      label: 'Recall @3',
      value: quality.recallAt3 !== undefined ? Math.min(quality.recallAt3, 1) : undefined,
      helper: 'Coverage of expected facts',
      invert: false,
    },
    {
      label: 'Hallucination Rate',
      value: quality.hallucinationRate !== undefined ? Math.min(quality.hallucinationRate, 1) : undefined,
      helper: 'Lower is better',
      invert: true,
    },
  ];
};

export const QualityPanel = () => {
  const quality = useDashboardStore((state) => state.quality);

  const metrics = useMemo(() => deriveMetrics(quality), [quality]);

  return (
    <Card sx={{ border: '1px solid rgba(255,255,255,0.2)', backgroundColor: 'rgba(15,23,42,0.65)', height: '100%' }}>
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="h6" color="text.primary">
            Quality Signals
          </Typography>
          {quality.lastEvaluatedAt && (
            <Typography variant="caption" color="text.secondary">
              Last audit {new Date(quality.lastEvaluatedAt).toLocaleTimeString()}
            </Typography>
          )}
        </Stack>
        <Stack spacing={2}>{metrics.map((metric) => (
          <MetricCard key={metric.label} {...metric} />
        ))}</Stack>
        <Card elevation={0} sx={{ background: 'rgba(15,23,42,0.25)', borderRadius: 2, border: '1px solid rgba(255,255,255,0.08)' }}>
          <CardContent sx={{ p: 2 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="subtitle2" color="text.secondary">
                Validation Checks
              </Typography>
              <Typography variant="caption" color={quality.validationFailures ? 'error.main' : 'success.main'}>
                {quality.validationFailures ?? 0} failures • {quality.emptyFields ?? 0} empty fields
              </Typography>
            </Stack>
            {quality.layerFailures.length ? (
              <Accordion sx={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.08)' }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ fontSize: 18 }} />}>
                  <Typography variant="body2" color="text.secondary">
                    Show layer failure details
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={1}>
                    {quality.layerFailures.map((failure, index) => (
                      <Typography key={`${failure}-${index}`} variant="body2" color="error.light">
                        • {failure}
                      </Typography>
                    ))}
                  </Stack>
                </AccordionDetails>
              </Accordion>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No outstanding validation issues.
              </Typography>
            )}
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  );
};

