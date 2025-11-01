import { Badge, Card, CardBody, Collapse, IconButton, List, ListIcon, Stack, Text } from '@chakra-ui/react';
import { useState } from 'react';
import { FiChevronDown, FiChevronRight, FiCircle, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { PlanStep } from '../types';

interface PlanPanelProps {
  steps: PlanStep[];
}

const statusIcon = {
  pending: FiCircle,
  running: FiChevronRight,
  succeeded: FiCheckCircle,
  failed: FiXCircle,
  skipped: FiCircle,
} as const;

export const PlanPanel = ({ steps }: PlanPanelProps) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggle = (id: string) => setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));

  return (
    <Card borderRadius="xl" borderColor="whiteAlpha.200">
      <CardBody as={Stack} spacing={4}>
        <Text textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200">
          Supervisor Plan
        </Text>
        <List spacing={3}>
          {steps.length === 0 ? (
            <Text fontSize="sm" color="slate.300">
              Waiting for supervisor plan...
            </Text>
          ) : (
            steps.map((step) => (
              <PlanItem key={step.id} step={step} depth={0} expanded={expanded} onToggle={toggle} />
            ))
          )}
        </List>
      </CardBody>
    </Card>
  );
};

interface PlanItemProps {
  step: PlanStep;
  depth: number;
  expanded: Record<string, boolean>;
  onToggle: (id: string) => void;
}

const badgeColor = {
  pending: 'gray',
  running: 'blue',
  succeeded: 'green',
  failed: 'red',
  skipped: 'purple',
} as const;

const PlanItem = ({ step, depth, expanded, onToggle }: PlanItemProps) => {
  const hasChildren = (step.children?.length ?? 0) > 0;
  const isExpanded = expanded[step.id] ?? depth === 0;

  return (
    <Stack spacing={2} pl={depth * 4} borderLeft={depth ? '1px dashed rgba(148, 163, 184, 0.3)' : 'none'}>
      <Stack direction="row" align="center" spacing={3}>
        {hasChildren && (
          <IconButton
            aria-label={isExpanded ? 'Collapse step' : 'Expand step'}
            icon={isExpanded ? <FiChevronDown /> : <FiChevronRight />}
            size="xs"
            variant="ghost"
            onClick={() => onToggle(step.id)}
          />
        )}
        {!hasChildren && <span style={{ width: '32px' }} />}
        <ListIcon as={statusIcon[step.status]} color={`${badgeColor[step.status]}.300`} />
        <Text fontSize="sm" color="slate.100">
          {step.label}
        </Text>
        <Badge variant="subtle" colorScheme={badgeColor[step.status]} textTransform="capitalize">
          {step.status}
        </Badge>
      </Stack>
      {hasChildren && (
        <Collapse in={isExpanded} animateOpacity>
          <Stack spacing={2} pl={6} borderLeft="1px solid rgba(148, 163, 184, 0.2)">
            {step.children?.map((child) => (
              <PlanItem key={child.id} step={child} depth={depth + 1} expanded={expanded} onToggle={onToggle} />
            ))}
          </Stack>
        </Collapse>
      )}
    </Stack>
  );
};
