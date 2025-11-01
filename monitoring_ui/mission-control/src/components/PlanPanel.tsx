import {
  Card,
  CardBody,
  Flex,
  Icon,
  Stack,
  Text,
  VStack,
} from '@chakra-ui/react';
import { CheckCircleIcon, RepeatClockIcon, TimeIcon, WarningIcon } from '@chakra-ui/icons';
import { PlanStep } from '../types';
import { useDashboardStore } from '../state/useDashboardStore';

const iconMap = (status: PlanStep['status']) => {
  switch (status) {
    case 'completed':
      return { icon: CheckCircleIcon, color: 'green.300', label: 'Completed' };
    case 'in_progress':
      return { icon: RepeatClockIcon, color: 'yellow.300', label: 'In Progress' };
    case 'failed':
      return { icon: WarningIcon, color: 'red.300', label: 'Failed' };
    case 'skipped':
      return { icon: TimeIcon, color: 'cyan.300', label: 'Skipped' };
    default:
      return { icon: TimeIcon, color: 'slate.400', label: 'Pending' };
  }
};

type PlanPanelProps = {
  steps: PlanStep[];
  visible: boolean;
};

export const PlanPanel = ({ steps, visible }: PlanPanelProps) => {
  const highlightAgentId = useDashboardStore((state) => state.highlightAgentId);
  const actions = useDashboardStore((state) => state.actions);

  if (!visible) {
    return (
      <Card
        borderStyle="dashed"
        borderColor="whiteAlpha.300"
        h="100%"
        alignItems="center"
        justifyContent="center"
      >
        <CardBody>
          <Text textAlign="center" color="slate.300" fontSize="sm">
            Plan view collapsed — toggle from the summary panel to monitor orchestration.
          </Text>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card h="100%" borderColor="whiteAlpha.200">
      <CardBody overflowY="auto" display="flex" flexDirection="column" gap={3}>
        {steps.length === 0 ? (
          <Flex justify="center" align="center" py={12}>
            <Text color="slate.300">Waiting for supervisor plan…</Text>
          </Flex>
        ) : (
          <VStack spacing={2} align="stretch">
            {steps.map((step) => {
              const { icon, color, label } = iconMap(step.status);
              const isLinked = step.relatedAgentId && step.relatedAgentId === highlightAgentId;
              return (
                <Flex
                  key={step.id}
                  gap={3}
                  align="center"
                  px={3}
                  py={2}
                  borderRadius="lg"
                  borderWidth="1px"
                  borderColor={isLinked ? 'brand.400' : 'transparent'}
                  bg={isLinked ? 'rgba(99, 179, 237, 0.1)' : 'transparent'}
                  onMouseEnter={() => actions.setHighlightPlanStep(step.id, step.relatedAgentId)}
                  onMouseLeave={() => actions.setHighlightPlanStep(undefined, undefined)}
                >
                  <Icon as={icon} boxSize={4} color={color} />
                  <Stack spacing={0} flex={1}>
                    <Text fontWeight="semibold" fontSize="sm">
                      {step.title}
                    </Text>
                    <Text fontSize="xs" color="slate.300">
                      {label}
                    </Text>
                  </Stack>
                </Flex>
              );
            })}
          </VStack>
        )}
      </CardBody>
    </Card>
  );
};
