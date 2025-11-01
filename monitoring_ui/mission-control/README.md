# Mission Control Dashboard UI

This package contains the next-generation monitoring dashboard for the Enlitens multi-agent system. It ships as a standalone React + Vite application written in TypeScript and styled with Material UI.

## Getting started

```bash
cd monitoring_ui/mission-control
npm install
npm run dev
```

The development server defaults to <http://localhost:5173>. It expects the monitoring FastAPI server to be reachable at `http://localhost:8000`.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `VITE_MONITORING_API` | Base URL for REST snapshot endpoints | `""` (relative) |
| `VITE_MONITORING_WS` | WebSocket URL for real-time telemetry | `ws://localhost:8000/ws` |

Create a `.env` file in this directory if you need to override the defaults:

```bash
VITE_MONITORING_API="https://monitoring.enlitens.internal"
VITE_MONITORING_WS="wss://monitoring.enlitens.internal/ws"
```

## Available scripts

- `npm run dev` – start the Vite dev server
- `npm run build` – type-check and create a production bundle
- `npm run preview` – serve the production build locally

## Project structure

```
mission-control/
├── src/
│   ├── components/        # UI building blocks (pipeline graph, plan panel, etc.)
│   ├── hooks/             # React hooks (websocket lifecycle)
│   ├── services/          # API and telemetry utilities
│   ├── state/             # Zustand store for normalized telemetry
│   ├── theme/             # Material UI theme configuration
│   └── types.ts           # Shared dashboard types
└── public/                # Static assets served by Vite
```

## Quality checks

The CI entry point for the dashboard is `npm run build`, which runs TypeScript checks and compiles the production bundle. The build currently emits a single chunk around ~500 kB after minification; future phases will introduce code-splitting as more features arrive.
