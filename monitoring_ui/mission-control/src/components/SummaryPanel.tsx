import { MdViewSidebar } from 'react-icons/md';
import {
  Card,
  CardBody,
  Flex,
  IconButton,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  Tooltip,
} from '@chakra-ui/react';
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
    <Card borderRadius="xl" borderColor="whiteAlpha.200">
      <CardBody as={Stack} spacing={6}>
        <Flex justify="space-between" align="flex-start">
          <Stack spacing={1}>
            <Text textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200">
              Active Document
            </Text>
            <Text fontSize="lg" fontWeight="semibold" color="slate.100">
              {summary.currentDocument ?? 'Idle'}
            </Text>
          </Stack>
          <Tooltip label={layout.showPlan ? 'Collapse plan panel' : 'Show plan panel'}>
            <IconButton
              aria-label="Toggle plan panel"
              icon={<MdViewSidebar />}
              onClick={actions.togglePlanVisibility}
              variant="ghost"
              colorScheme="blue"
              size="sm"
            />
          </Tooltip>
        </Flex>

        <Stack spacing={2}>
          <Flex justify="space-between" align="center">
            <Text fontSize="sm" color="slate.300">
              Pipeline progress
            </Text>
            <Text fontSize="sm" fontWeight="semibold" color="slate.100">
              {summary.progressPercentage.toFixed(1)}%
            </Text>
          </Flex>
          <Progress
            value={summary.progressPercentage}
            height="10px"
            borderRadius="full"
            bg="whiteAlpha.200"
            colorScheme="blue"
          />
        </Stack>

        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
          <Stack spacing={0.5}>
            <Text fontSize="sm" color="slate.300">
              Processed
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="slate.100">
              {summary.documentsProcessed}
            </Text>
            <Text fontSize="xs" color="slate.400">
              of {summary.totalDocuments || '—'} documents
            </Text>
          </Stack>

          <Stack spacing={0.5}>
            <Text fontSize="sm" color="slate.300">
              Time on document
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="slate.100">
              {formatSeconds(summary.timeOnDocumentSeconds)}
            </Text>
            <Text fontSize="xs" color="slate.400">
              Last log {formatSeconds(summary.lastLogSecondsAgo)} ago
            </Text>
          </Stack>

          <Stack spacing={0.5}>
            <Text fontSize="sm" color="slate.300">
              Alerts
            </Text>
            <Text fontSize="2xl" fontWeight="bold" color="slate.100">
              {summary.errors}
            </Text>
            <Text fontSize="xs" color="slate.400">
              Warnings: {summary.warnings}
            </Text>
          </Stack>
        </SimpleGrid>
      </CardBody>
    </Card>
  );
};
