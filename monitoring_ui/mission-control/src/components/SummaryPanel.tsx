import { Button, Card, CardBody, Flex, SimpleGrid, Stack, Stat, StatHelpText, StatLabel, StatNumber, Text } from '@chakra-ui/react';
import { LayoutState, SummarySnapshot } from '../types';

interface SummaryPanelProps {
  summary: SummarySnapshot;
  layout: LayoutState;
  onTogglePlan: (value?: boolean) => void;
}

export const SummaryPanel = ({ summary, layout, onTogglePlan }: SummaryPanelProps) => {
  return (
    <Card borderRadius="xl" borderColor="whiteAlpha.200">
      <CardBody>
        <Flex direction={{ base: 'column', md: 'row' }} justify="space-between" align="center" gap={4}>
          <Stack spacing={1} align={{ base: 'flex-start', md: 'flex-start' }}>
            <Text textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200">
              Active Document
            </Text>
            <Text fontSize="3xl" fontWeight="bold" color="white">
              {summary.activeDocumentId ?? '—'}
            </Text>
            <Text fontSize="sm" color="slate.300">
              Mission status indicator updates in real time
            </Text>
          </Stack>
          <Button onClick={() => onTogglePlan()} variant="outline" colorScheme="brand" size="sm">
            {layout.showPlan ? 'Hide Plan' : 'Show Plan'}
          </Button>
        </Flex>

        <SimpleGrid columns={{ base: 1, sm: 3 }} spacing={6} mt={6}>
          <SummaryStat label="Documents in Queue" value={summary.totalDocuments} help="Current processing backlog" />
          <SummaryStat label="Completed Today" value={summary.completedToday} help="Successful missions" />
          <SummaryStat label="Failed Today" value={summary.failedToday} help="Needs review" />
        </SimpleGrid>
      </CardBody>
    </Card>
  );
};

interface SummaryStatProps {
  label: string;
  value: number;
  help: string;
}

const SummaryStat = ({ label, value, help }: SummaryStatProps) => (
  <Stat>
    <StatLabel color="slate.300">{label}</StatLabel>
    <StatNumber color="white">{value}</StatNumber>
    <StatHelpText color="slate.400">{help}</StatHelpText>
  </Stat>
);
