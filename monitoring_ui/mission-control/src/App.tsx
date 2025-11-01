import { Box, Container, SimpleGrid, Stack } from '@chakra-ui/react';
import { useEffect } from 'react';
import { AlertBanner } from './components/AlertBanner';
import { InsightStrip } from './components/InsightStrip';
import { PipelineGraph } from './components/PipelineGraph';
import { PlanPanel } from './components/PlanPanel';
import { PerformancePanel } from './components/PerformancePanel';
import { QualityPanel } from './components/QualityPanel';
import { SummaryPanel } from './components/SummaryPanel';
import { useTelemetryConnection } from './hooks/useTelemetryConnection';
import { useDashboardStore } from './state/useDashboardStore';

function App() {
  useTelemetryConnection();
  const summary = useDashboardStore((state) => state.summary);
  const agents = useDashboardStore((state) => state.agents);
  const plan = useDashboardStore((state) => state.plan);
  const performance = useDashboardStore((state) => state.performance);
  const quality = useDashboardStore((state) => state.quality);
  const insights = useDashboardStore((state) => state.insights);
  const connection = useDashboardStore((state) => state.connection);
  const highlightAgentId = useDashboardStore((state) => state.highlightAgentId);
  const layout = useDashboardStore((state) => state.layout);
  const togglePlanVisibility = useDashboardStore((state) => state.actions.togglePlanVisibility);

  useEffect(() => {
    if (window.innerWidth < 1024) {
      togglePlanVisibility(false);
    }
  }, [togglePlanVisibility]);

  return (
    <Box minH="100vh" bgGradient="linear(135deg, #0f172a 0%, #1a365d 60%, #0b1120 100%)" py={{ base: 10, md: 16 }}>
      <Container maxW="7xl">
        <Stack spacing={8}>
          <AlertBanner severity={summary.severity} messages={summary.alertMessages} connectionStatus={connection} />
          <SummaryPanel summary={summary} onTogglePlan={togglePlanVisibility} layout={layout} />
          <InsightStrip insights={insights} />

          <SimpleGrid columns={{ base: 1, xl: layout.showPlan ? 2 : 1 }} spacing={6} alignItems="stretch">
            <PipelineGraph agents={agents} highlightAgentId={highlightAgentId} />
            {layout.showPlan && <PlanPanel steps={plan} />}
          </SimpleGrid>

          <SimpleGrid columns={{ base: 1, xl: 2 }} spacing={6} alignItems="stretch">
            <PerformancePanel stats={performance.stats} trend={performance.trend} />
            <QualityPanel metrics={quality.metrics} layerFailures={quality.layerFailures} />
          </SimpleGrid>
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
