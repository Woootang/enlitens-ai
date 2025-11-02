import { Card, CardContent, Chip, Stack, Typography } from '@mui/material';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import { useDashboardStore } from '../state/useDashboardStore';

export const InsightStrip = () => {
  const insights = useDashboardStore((state) => state.insights);

  return (
    <Card sx={{ border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(14, 23, 42, 0.55)' }}>
      <CardContent>
        <Stack direction="row" alignItems="center" spacing={2} flexWrap="wrap" rowGap={1.5}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <LightbulbIcon fontSize="small" />
            <Typography variant="subtitle2" color="text.secondary">
              Insights
            </Typography>
          </Stack>
          {insights.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No anomalies detected in the latest window.
            </Typography>
          ) : (
            insights.map((insight, index) => (
              <Chip key={`${insight}-${index}`} label={insight} color="info" variant="outlined" />
            ))
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

