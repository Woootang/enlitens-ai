import { useMemo } from 'react';
import ReactFlow, { Background, Controls, Edge, Node } from 'reactflow';
import { Badge, Box, Card, CardBody, Text, VStack } from '@chakra-ui/react';
import 'reactflow/dist/style.css';
import { PipelineAgent } from '../types';

type PipelineGraphProps = {
  agents: PipelineAgent[];
  highlightAgentId?: string;
};

const STATUS_COLORS: Record<string, string> = {
  running: '#F6AD55',
  completed: '#48BB78',
  failed: '#F56565',
  idle: '#A0AEC0',
  unknown: '#718096',
};

const NODE_WIDTH = 200;

const buildNode = (agent: PipelineAgent, index: number, highlight?: boolean): Node => ({
  id: agent.id,
  position: { x: index * (NODE_WIDTH + 48), y: 0 },
  data: { agent, highlight },
  type: 'pipelineNode',
});

const buildEdge = (source: string, target: string, index: number): Edge => ({
  id: `edge-${source}-${target}-${index}`,
  source,
  target,
  animated: true,
  style: {
    stroke: 'rgba(99, 179, 237, 0.6)',
    strokeWidth: 2,
  },
});

const PipelineNode = ({ data }: { data: { agent: PipelineAgent; highlight?: boolean } }) => {
  const { agent, highlight } = data;
  const statusColor = STATUS_COLORS[agent.status] ?? STATUS_COLORS.unknown;
  return (
    <Card
      variant="elevated"
      borderRadius="xl"
      borderWidth={highlight ? '2px' : '1px'}
      borderColor={highlight ? 'brand.400' : 'whiteAlpha.200'}
      bg="rgba(15, 23, 42, 0.85)"
      w={`${NODE_WIDTH}px`}
      boxShadow={highlight ? 'xl' : 'md'}
    >
      <CardBody>
        <VStack align="stretch" spacing={2}>
          <Text fontWeight="semibold" fontSize="sm" color="slate.100" noOfLines={2}>
            {agent.name}
          </Text>
          <Badge
            alignSelf="flex-start"
            color={statusColor}
            bg={`${statusColor}22`}
            px={2.5}
            py={1}
            fontSize="0.65rem"
            fontWeight="bold"
            borderRadius="full"
          >
            {agent.status.toUpperCase()}
          </Badge>
          <Text fontSize="xs" color="slate.300">
            Execs: {agent.executions ?? 0} • Avg: {agent.avgTimeSeconds ? `${agent.avgTimeSeconds.toFixed(2)}s` : 'n/a'}
          </Text>
        </VStack>
      </CardBody>
    </Card>
  );
};

const nodeTypes = { pipelineNode: PipelineNode } as const;

export const PipelineGraph = ({ agents, highlightAgentId }: PipelineGraphProps) => {
  const { nodes, edges } = useMemo(() => {
    const nodes = agents.map((agent, index) => buildNode(agent, index, agent.id === highlightAgentId || agent.isActive));
    const edges: Edge[] = [];
    for (let i = 0; i < agents.length - 1; i += 1) {
      edges.push(buildEdge(agents[i].id, agents[i + 1].id, i));
    }
    return { nodes, edges };
  }, [agents, highlightAgentId]);

  if (agents.length === 0) {
    return (
      <Box
        p={8}
        textAlign="center"
        color="slate.400"
        borderWidth="1px"
        borderStyle="dashed"
        borderRadius="xl"
        borderColor="whiteAlpha.300"
      >
        Awaiting pipeline activity…
      </Box>
    );
  }

  return (
    <Box h="280px" w="100%" borderRadius="xl" overflow="hidden" borderWidth="1px" borderColor="whiteAlpha.200">
      <ReactFlow
        nodes={nodes as unknown as Node[]}
        edges={edges}
        nodesDraggable={false}
        zoomOnScroll={false}
        panOnScroll
        fitView
        nodeTypes={nodeTypes}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={32} color="rgba(99, 179, 237, 0.1)" />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>
    </Box>
  );
};
