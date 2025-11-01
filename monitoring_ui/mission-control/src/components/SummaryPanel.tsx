import { Box, IconButton, LinearProgress, Paper, Stack, Tooltip, Typography } from '@mui/material';
import ViewSidebarIcon from '@mui/icons-material/ViewSidebar';
import { DashboardSummary } from '../types';
import { useDashboardStore } from '../state/useDashboardStore';

type SummaryPanelProps = {
  summary: DashboardSummary;
};

const formatSeconds = (value?: number | null) => {
  if (!value) return '—';
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}m ${seconds}s`;
};

export const SummaryPanel = ({ summary }: SummaryPanelProps) => {
  const actions = useDashboardStore((state) => state.actions);
  const layout = useDashboardStore((state) => state.layout);

  return (
    <Paper sx={{ p: 4, borderRadius: 3, border: '1px solid rgba(255,255,255,0.2)', backgroundColor: 'rgba(17,25,40,0.65)' }}>
      <Stack direction="row" justifyContent="space-between" alignItems="flex-start" spacing={2} mb={3}>
        <Stack spacing={0.5}>
          <Typography variant="overline" color="text.secondary" letterSpacing={2}>
            Active Document
          </Typography>
          <Typography variant="h6" color="text.primary">
            {summary.currentDocument ?? 'Idle'}
          </Typography>
        </Stack>
        <Tooltip title={layout.showPlan ? 'Collapse plan panel' : 'Show plan panel'}>
          <IconButton color="primary" onClick={actions.togglePlanVisibility} size="small">
            <ViewSidebarIcon />
          </IconButton>
        </Tooltip>
      </Stack>
      <Stack spacing={3}>
        <Box>
          <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" color="text.secondary">
              Pipeline progress
            </Typography>
            <Typography variant="body2" fontWeight={600}>
              {summary.progressPercentage.toFixed(1)}%
            </Typography>
          </Stack>
          <LinearProgress
            variant="determinate"
            value={summary.progressPercentage}
            sx={{ height: 8, borderRadius: 999, backgroundColor: 'rgba(255,255,255,0.12)' }}
            color="primary"
          />
        </Box>
        <Box
          sx={{
            display: 'grid',
            gap: 3,
            gridTemplateColumns: { xs: '1fr', md: 'repeat(3, minmax(0, 1fr))' },
          }}
        >
          <Stack spacing={0.5}>
            <Typography variant="subtitle2" color="text.secondary">
              Processed
            </Typography>
            <Typography variant="h5" color="text.primary">
              {summary.documentsProcessed}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              of {summary.totalDocuments || '—'} documents
            </Typography>
          </Stack>
          <Stack spacing={0.5}>
            <Typography variant="subtitle2" color="text.secondary">
              Time on document
            </Typography>
            <Typography variant="h5" color="text.primary">
              {formatSeconds(summary.timeOnDocumentSeconds)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Last log {formatSeconds(summary.lastLogSecondsAgo)} ago
            </Typography>
          </Stack>
          <Stack spacing={0.5}>
            <Typography variant="subtitle2" color="text.secondary">
              Alerts
            </Typography>
            <Typography variant="h5" color="text.primary">
              {summary.errors}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Warnings: {summary.warnings}
            </Typography>
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );
};
