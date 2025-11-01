import { Chip, Paper, Stack, Typography } from '@mui/material';
import { DashboardInsight } from '../types';

interface InsightStripProps {
  insights: DashboardInsight[];
}

const severityColor: Record<DashboardInsight['severity'], 'default' | 'warning' | 'error' | 'info' | 'success'> = {
  info: 'info',
  normal: 'success',
  warning: 'warning',
  critical: 'error',
};

export const InsightStrip = ({ insights }: InsightStripProps) => {
  return (
    <Paper
      sx={{
        p: 2.5,
        borderRadius: 3,
        border: '1px solid rgba(148,163,184,0.3)',
        background: 'rgba(15,23,42,0.6)',
      }}
    >
      <Stack spacing={1.5}>
        <Typography variant="overline" color="primary.light" letterSpacing={2}>
          Insights
        </Typography>
        {insights.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            System nominal. No anomalies detected.
          </Typography>
        ) : (
          <Stack direction="row" flexWrap="wrap" gap={1.25}>
            {insights.map((insight) => (
              <Chip
                key={insight.id}
                label={insight.message}
                color={severityColor[insight.severity]}
                variant="filled"
                size="small"
                sx={{ fontSize: 12, px: 1 }}
              />
            ))}
          </Stack>
        )}
      </Stack>
    </Paper>
  );
};
