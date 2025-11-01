export type SocketStatus = 'connecting' | 'open' | 'closed';

type MessageHandler = (data: unknown) => void;
type StatusHandler = (status: SocketStatus) => void;

export interface SocketOptions {
  heartbeatMs?: number;
  reconnectMs?: number;
}

export class ManagedWebSocket {
  private url: string;
  private ws: WebSocket | null = null;
  private messageHandler: MessageHandler;
  private statusHandler?: StatusHandler;
  private heartbeat?: number;
  private reconnectTimer?: number;
  private options: Required<SocketOptions> = {
    heartbeatMs: 30000,
    reconnectMs: 5000,
  };

  constructor(url: string, onMessage: MessageHandler, onStatus?: StatusHandler, options?: SocketOptions) {
    this.url = url;
    this.messageHandler = onMessage;
    this.statusHandler = onStatus;
    this.options = { ...this.options, ...(options ?? {}) };
  }

  public connect(): void {
    if (typeof window === 'undefined') {
      return;
    }

    this.statusHandler?.('connecting');
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.statusHandler?.('open');
      this.startHeartbeat();
    };

    this.ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        this.messageHandler(parsed);
      } catch {
        this.messageHandler(event.data);
      }
    };

    this.ws.onerror = () => {
      this.scheduleReconnect();
    };

    this.ws.onclose = () => {
      this.statusHandler?.('closed');
      this.stopHeartbeat();
      this.scheduleReconnect();
    };
  }

  public disconnect(): void {
    this.stopHeartbeat();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.close();
    }
    this.ws = null;
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
    }
  }

  private startHeartbeat(): void {
    if (this.heartbeat) {
      window.clearInterval(this.heartbeat);
    }
    this.heartbeat = window.setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return;
      }
      this.ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
    }, this.options.heartbeatMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeat) {
      window.clearInterval(this.heartbeat);
      this.heartbeat = undefined;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      return;
    }
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = undefined;
      this.connect();
    }, this.options.reconnectMs);
  }
}
