import { useEffect, useRef } from 'react';
import { fetchSnapshot } from '../services/api';
import { ManagedWebSocket, SocketStatus } from '../services/socket';
import { useDashboardStore } from '../state/useDashboardStore';
import { TelemetryEvent } from '../types';

const DEFAULT_WS_URL = import.meta.env.VITE_MONITORING_WS ?? 'ws://localhost:8000/ws';
const SNAPSHOT_POLL_INTERVAL = 20000;

export const useTelemetryConnection = (url: string = DEFAULT_WS_URL) => {
  const actions = useDashboardStore((state) => state.actions);
  const wsRef = useRef<ManagedWebSocket | null>(null);
  const pollRef = useRef<number>();

  useEffect(() => {
    const handleStatus = (status: SocketStatus) => {
      actions.setConnection(status);
      if (status === 'open') {
        refreshSnapshot();
      }
    };

    const toEvent = (payload: Record<string, unknown>): TelemetryEvent => {
      const level = typeof payload.level === 'string' ? payload.level.toUpperCase() : 'INFO';
      const severity: TelemetryEvent['severity'] = level === 'ERROR' ? 'critical' : level === 'WARNING' ? 'warning' : 'normal';
      return {
        id: (payload.id as string) ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        timestamp: (payload.timestamp as string) ?? new Date().toISOString(),
        message: typeof payload.message === 'string' ? payload.message : JSON.stringify(payload),
        agentName: (payload.agent_name as string) ?? (payload.agentName as string | undefined),
        level: (level as TelemetryEvent['level']) ?? 'INFO',
        severity,
      };
    };

    const handleMessage = (message: unknown) => {
      if (typeof message === 'object' && message !== null) {
        const payload = message as Record<string, unknown>;
        if (payload.type === 'snapshot') {
          actions.ingestSnapshot(payload.payload as any);
          return;
        }
        if (payload.type === 'stats') {
          actions.ingestSnapshot((payload.data as any) ?? {});
          return;
        }
        if (payload.type === 'log') {
          actions.ingestEvent(toEvent(payload));
          return;
        }
      }
      const fallback = {
        id: `${Date.now()}-stream`,
        timestamp: new Date().toISOString(),
        level: 'INFO' as const,
        message: typeof message === 'string' ? message : JSON.stringify(message),
        severity: 'normal' as const,
      };
      actions.ingestEvent(fallback);
    };

    const refreshSnapshot = async () => {
      const snapshot = await fetchSnapshot();
      if (snapshot) {
        actions.ingestSnapshot(snapshot);
      }
    };

    const ws = new ManagedWebSocket(url, handleMessage, handleStatus, {
      heartbeatMs: 20000,
      reconnectMs: 6000,
    });
    ws.connect();
    wsRef.current = ws;

    const interval = window.setInterval(refreshSnapshot, SNAPSHOT_POLL_INTERVAL);
    pollRef.current = interval;

    return () => {
      ws.disconnect();
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
      }
    };
  }, [url, actions]);
};
