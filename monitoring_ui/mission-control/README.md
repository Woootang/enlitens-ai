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

## Additional tooling

- Run `npm run test:regression` after `npm install`/`npx playwright install` to execute the Playwright regression suite located under `tests/playwright`.
- Accessibility gate: `npx axe http://localhost:5173`
- Lighthouse snapshot: `npx lighthouse http://localhost:5173 --preset=desktop --output=json --output=html --output-path=./lighthouse-report`

## Docker compose profile

A two-container profile is available in `docker/mission-control/docker-compose.yml`:

```bash
cd docker/mission-control
docker compose up --build
```

This publishes:

- `monitoring-api` on http://localhost:8000
- `monitoring-ui` on http://localhost:5173

Adjust the compose file to customise ports or mount alternative knowledge base directories.

## Operator checklist

1. Start the stack (compose or `npm run dev` + FastAPI).
2. Monitor alert + insight banners, pipeline DAG, and plan panel for orchestration issues.
3. Use performance and quality panels to identify regressions.
4. Open the assistant dock (chat icon) to query telemetry or trigger guarded actions (retry, verbose logging) — each action requires confirmation and posts results back to the transcript.
