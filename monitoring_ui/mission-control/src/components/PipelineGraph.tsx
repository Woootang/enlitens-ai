import { useMemo } from 'react';
import ReactFlow, { Background, Controls, Edge, Node } from 'reactflow';
import { Box, Chip, Paper, Stack, Typography } from '@mui/material';
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
    <Paper
      elevation={highlight ? 8 : 3}
      sx={{
        width: NODE_WIDTH,
        p: 2,
        borderRadius: 3,
        border: highlight ? '2px solid rgba(99,179,237,0.8)' : '1px solid rgba(255,255,255,0.15)',
        backgroundColor: 'rgba(15, 23, 42, 0.85)',
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
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={32} color="rgba(99, 179, 237, 0.1)" />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>
    </Box>
  );
};
