export const getTelemetryUrl = () => {
  const url = import.meta.env.VITE_TELEMETRY_WS_URL as string | undefined;
  return url ?? `ws://${window.location.hostname}:8000/ws/telemetry`;
};
