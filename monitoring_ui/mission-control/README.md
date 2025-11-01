# Enlitens Mission Control Dashboard

A Chakra UI + Vite dashboard that streams real-time telemetry from the multi-agent knowledge synthesis pipeline. Phase 2 introduces comparative performance analytics and multi-dimensional quality indicators while retaining the mission-critical pipeline and plan visualisations established in Phase 1.

## Getting Started

```bash
cd monitoring_ui/mission-control
npm install
npm run dev
```

Set `VITE_TELEMETRY_WS_URL` in a `.env` file if the telemetry server runs on a different host or port. By default the dashboard connects to `ws://<host>:8000/ws/telemetry`.

## Available Scripts

- `npm run dev` – start a local dev server.
- `npm run build` – type-check and build the production bundle.
- `npm run preview` – preview the production build locally.

## Accessibility & Responsiveness

- Uses Chakra UI responsive primitives so the layout adapts down to tablet widths.
- Keyboard accessible toggles for plan expansion and chart controls.
- Color ramps meet WCAG contrast guidelines for dark mode backgrounds.

## Telemetry Expectations

The WebSocket feed should deliver JSON messages with the following types:

- `summary`: high-level status snapshot.
- `agents`: array of agent objects with `id`, `name`, and `status`.
- `plan`: hierarchical supervisor plan.
- `performance`: stats per agent plus aggregate latency trend.
- `quality`: quality metrics and validation flags.
- `insight`: array of computed insight messages.

Messages are normalised inside `src/services/normalize.ts` before being applied to the global store.
