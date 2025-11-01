import { Card, CardBody, Text } from '@chakra-ui/react';
import ReactFlow, { Background, Controls, Edge, MiniMap, Node, Position } from 'reactflow';
import 'reactflow/dist/style.css';
import { useMemo } from 'react';
import { AgentNode } from '../types';

interface PipelineGraphProps {
  agents: AgentNode[];
  highlightAgentId: string | null;
}

const statusColor: Record<AgentNode['status'], string> = {
  pending: '#475569',
  running: '#38bdf8',
  succeeded: '#34d399',
  failed: '#f87171',
  skipped: '#94a3b8',
};

export const PipelineGraph = ({ agents, highlightAgentId }: PipelineGraphProps) => {
  const nodes = useMemo<Node[]>(() => {
    if (!agents.length) {
      return [
        {
          id: 'placeholder',
          data: { label: 'Awaiting telemetry' },
          position: { x: 0, y: 0 },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          style: {
            background: '#1e293b',
            color: '#cbd5f5',
            padding: 16,
            borderRadius: 16,
            border: '1px solid rgba(148, 163, 184, 0.4)',
          },
        },
      ];
    }

    return agents.map((agent, index) => ({
      id: agent.id,
      data: {
        label: (
          <Text fontWeight={agent.id === highlightAgentId ? 'extrabold' : 'medium'} color="slate.100">
            {agent.name}
          </Text>
        ),
      },
      position: { x: index * 220, y: 0 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: {
        background: agent.id === highlightAgentId ? '#0ea5e9' : '#1e293b',
        color: '#e2e8f0',
        borderRadius: 18,
        border: `2px solid ${statusColor[agent.status]}`,
        padding: 16,
        boxShadow: agent.id === highlightAgentId ? '0 0 20px rgba(14,165,233,0.6)' : 'none',
        transition: 'all 0.2s ease',
      },
    }));
  }, [agents, highlightAgentId]);

  const edges = useMemo<Edge[]>(() => {
    if (agents.length < 2) return [];
    return agents.slice(1).map((agent, index) => ({
      id: `${agents[index].id}-${agent.id}`,
      source: agents[index].id,
      target: agent.id,
      animated: agents[index].status === 'running' || agents[index].status === 'succeeded',
      style: { stroke: 'rgba(148, 163, 184, 0.6)', strokeWidth: 2 },
    }));
  }, [agents]);

  return (
    <Card borderRadius="xl" borderColor="whiteAlpha.200" minH="360px">
      <CardBody>
        <Text textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200" mb={3}>
          Agent Pipeline
        </Text>
        <div style={{ width: '100%', height: 280 }}>
          <ReactFlow nodes={nodes} edges={edges} fitView>
            <Background color="#1f2937" gap={16} size={0.5} />
            <MiniMap pannable zoomable nodeColor={(node) => '#0f172a'} maskColor="rgba(15,23,42,0.6)" />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      </CardBody>
    </Card>
  );
};
