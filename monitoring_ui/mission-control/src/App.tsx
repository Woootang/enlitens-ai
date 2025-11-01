import { Box, Container, Stack } from '@mui/material';
import { AlertBanner } from './components/AlertBanner';
import { InsightStrip } from './components/InsightStrip';
import { PipelineGraph } from './components/PipelineGraph';
import { PlanPanel } from './components/PlanPanel';
import { PerformancePanel } from './components/PerformancePanel';
import { QualityPanel } from './components/QualityPanel';
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
  const insights = useDashboardStore((state) => state.insights);
  const performance = useDashboardStore((state) => state.performance);
  const quality = useDashboardStore((state) => state.quality);
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
          <InsightStrip insights={insights} />
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: layout.showPlan ? { xs: '1fr', xl: '2fr 1fr' } : '1fr',
              gap: 4,
              alignItems: 'stretch',
            }}
          >
            <Box>
              <PipelineGraph agents={agents} highlightAgentId={highlightAgentId} />
            </Box>
            {layout.showPlan && (
              <Box>
                <PlanPanel steps={plan} visible={layout.showPlan} />
              </Box>
            )}
          </Box>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', xl: '1fr 1fr' },
              gap: 4,
            }}
          >
            <PerformancePanel stats={performance.stats} trend={performance.trend} />
            <QualityPanel metrics={quality.metrics} layerFailures={quality.layerFailures} />
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
