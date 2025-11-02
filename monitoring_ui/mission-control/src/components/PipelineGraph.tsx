import { keyframes } from '@emotion/react';
import { useCallback, useMemo, type MouseEvent } from 'react';
import ReactFlow, { Background, Controls, Edge, Node } from 'reactflow';
import { Box, Chip, Paper, Stack, Tooltip, Typography } from '@mui/material';
import 'reactflow/dist/style.css';
import { PipelineAgent } from '../types';
import { useDashboardStore } from '../state/useDashboardStore';

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

const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(99, 179, 237, 0.4); }
  70% { box-shadow: 0 0 0 12px rgba(99, 179, 237, 0); }
  100% { box-shadow: 0 0 0 0 rgba(99, 179, 237, 0); }
`;

const PipelineNode = ({ data }: { data: { agent: PipelineAgent; highlight?: boolean } }) => {
  const { agent, highlight } = data;
  const statusColor = STATUS_COLORS[agent.status] ?? STATUS_COLORS.unknown;
  return (
    <Tooltip
      title={
        <Stack spacing={0.5}>
          <Typography variant="caption" color="inherit">
            Executions: {agent.executions ?? 0}
          </Typography>
          <Typography variant="caption" color="inherit">
            Avg time: {agent.avgTimeSeconds ? `${agent.avgTimeSeconds.toFixed(2)}s` : 'n/a'}
          </Typography>
          <Typography variant="caption" color="inherit">
            Successes: {agent.successes ?? 0} • Failures: {agent.failures ?? 0}
          </Typography>
        </Stack>
      }
      arrow
      enterDelay={200}
    >
      <Paper
        elevation={highlight ? 10 : 3}
        sx={{
          position: 'relative',
          width: NODE_WIDTH,
          p: 2,
          borderRadius: 3,
          border: highlight ? '2px solid rgba(99,179,237,0.9)' : '1px solid rgba(255,255,255,0.12)',
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          transition: 'transform 160ms ease, border 160ms ease, box-shadow 160ms ease',
          transform: highlight ? 'translateY(-4px)' : 'none',
          '&::after': highlight
            ? {
                content: '""',
                position: 'absolute',
                inset: -4,
                borderRadius: 'inherit',
                border: '2px solid rgba(99,179,237,0.25)',
                animation: `${pulse} 2.4s infinite`,
              }
            : undefined,
        }}
      >
        <Stack spacing={1}>
          <Typography fontWeight={600} fontSize={14} color="text.primary" sx={{ wordBreak: 'break-word' }}>
            {agent.name}
          </Typography>
          <Chip
            label={agent.status.toUpperCase()}
            size="small"
            sx={{
              backgroundColor: `${statusColor}22`,
              color: statusColor,
              fontWeight: 600,
              width: 'fit-content',
            }}
          />
          <Typography fontSize={12} color="text.secondary">
            Execs: {agent.executions ?? 0} • Avg: {agent.avgTimeSeconds ? `${agent.avgTimeSeconds.toFixed(2)}s` : 'n/a'}
          </Typography>
        </Stack>
      </Paper>
    </Tooltip>
  );
};

const nodeTypes = { pipelineNode: PipelineNode } as const;

export const PipelineGraph = ({ agents, highlightAgentId }: PipelineGraphProps) => {
  const plan = useDashboardStore((state) => state.plan);
  const actions = useDashboardStore((state) => state.actions);

  const { nodes, edges } = useMemo(() => {
    const nodes = agents.map((agent, index) => buildNode(agent, index, agent.id === highlightAgentId || agent.isActive));
    const edges: Edge[] = [];
    for (let i = 0; i < agents.length - 1; i += 1) {
      edges.push(buildEdge(agents[i].id, agents[i + 1].id, i));
    }
    return { nodes, edges };
  }, [agents, highlightAgentId]);

  const findRelatedStep = useCallback(
    (agentId: string) => plan.find((step) => step.relatedAgentId === agentId)?.id,
    [plan],
  );

  const handleNodeEnter = useCallback(
    (_: React.MouseEvent, node: Node) => {
      actions.setHighlightAgent(node.id);
      const stepId = findRelatedStep(node.id);
      if (stepId) {
        actions.setHighlightPlanStep(stepId, node.id);
      }
    },
    [actions, findRelatedStep],
  );

  const handleNodeLeave = useCallback(() => {
    actions.setHighlightAgent(undefined);
    actions.setHighlightPlanStep(undefined, undefined);
  }, [actions]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      actions.setHighlightAgent(node.id);
      const stepId = findRelatedStep(node.id);
      if (stepId) {
        actions.setHighlightPlanStep(stepId, node.id);
      }
    },
    [actions, findRelatedStep],
  );

  if (agents.length === 0) {
    return (
      <Box p={6} textAlign="center" color="text.secondary" borderRadius={3} border="1px dashed rgba(255,255,255,0.2)">
        Awaiting pipeline activity…
      </Box>
    );
  }

  return (
    <Box height={280} width="100%" borderRadius={3} overflow="hidden" border="1px solid rgba(255,255,255,0.2)">
      <ReactFlow
        nodes={nodes as unknown as Node[]}
        edges={edges}
        nodesDraggable={false}
        zoomOnScroll={false}
        panOnScroll
        fitView
        nodeTypes={nodeTypes}
        onNodeMouseEnter={handleNodeEnter}
        onNodeMouseLeave={handleNodeLeave}
        onNodeClick={handleNodeClick}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={32} color="rgba(99, 179, 237, 0.1)" />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>
    </Box>
  );
};
