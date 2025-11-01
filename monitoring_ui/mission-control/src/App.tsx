import { Box, Container, Stack } from '@mui/material';
import { AlertBanner } from './components/AlertBanner';
import { PipelineGraph } from './components/PipelineGraph';
import { PlanPanel } from './components/PlanPanel';
import { SummaryPanel } from './components/SummaryPanel';
import { useDashboardStore } from './state/useDashboardStore';
import { useTelemetryConnection } from './hooks/useTelemetryConnection';

function App() {
  useTelemetryConnection();
  const summary = useDashboardStore((state) => state.summary);
  const agents = useDashboardStore((state) => state.agents);
  const plan = useDashboardStore((state) => state.plan);
  const connection = useDashboardStore((state) => state.connection);
  const highlightAgentId = useDashboardStore((state) => state.highlightAgentId);
  const layout = useDashboardStore((state) => state.layout);

  return (
    <Box
      minHeight="100vh"
      sx={{
        background: 'linear-gradient(135deg, #0f172a 0%, #1a365d 60%, #0b1120 100%)',
        color: 'text.primary',
        py: { xs: 6, md: 10 },
      }}
    >
      <Container maxWidth="xl">
        <Stack spacing={4}>
          <AlertBanner severity={summary.severity} messages={summary.alertMessages} connectionStatus={connection} />
          <SummaryPanel summary={summary} />
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', xl: '2fr 1fr' },
              gap: 4,
              alignItems: 'stretch',
            }}
          >
            <Box>
              <PipelineGraph agents={agents} highlightAgentId={highlightAgentId} />
            </Box>
            <Box>
              <PlanPanel steps={plan} visible={layout.showPlan} />
            </Box>
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
