export type TelemetryMessage = {
  type: string;
  payload: unknown;
};

export type SocketOptions = {
  url: string;
  onMessage: (message: TelemetryMessage) => void;
  onStatusChange?: (status: 'connecting' | 'online' | 'offline') => void;
  heartbeatIntervalMs?: number;
};

export class TelemetrySocket {
  private socket: WebSocket | null = null;
  private heartbeatTimer: number | null = null;
  private readonly options: SocketOptions;

  constructor(options: SocketOptions) {
    this.options = options;
  }

  connect() {
    this.disconnect();
    this.options.onStatusChange?.('connecting');
    this.socket = new WebSocket(this.options.url);
    this.socket.onopen = () => {
      this.options.onStatusChange?.('online');
      this.startHeartbeat();
    };
    this.socket.onclose = () => {
      this.options.onStatusChange?.('offline');
      this.stopHeartbeat();
      window.setTimeout(() => this.connect(), 4000);
    };
    this.socket.onerror = () => {
      this.options.onStatusChange?.('offline');
    };
    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as TelemetryMessage;
        this.options.onMessage(data);
      } catch (error) {
        console.error('Failed to parse telemetry message', error);
      }
    };
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    const interval = this.options.heartbeatIntervalMs ?? 15000;
    this.heartbeatTimer = window.setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ type: 'ping', payload: Date.now() }));
      }
    }, interval);
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      window.clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  disconnect() {
    this.stopHeartbeat();
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}
