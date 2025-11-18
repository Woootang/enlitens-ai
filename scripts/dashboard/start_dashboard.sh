#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${ROOT_DIR}/venv/bin"

if [[ ! -x "${VENV_BIN}/gunicorn" ]]; then
  echo "gunicorn not found in ${VENV_BIN}. Did you run 'pip install gunicorn'?" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}"
export ENVIRONMENT="development"

echo "Starting dashboard on http://localhost:5000"
exec "${VENV_BIN}/gunicorn" --workers "${GUNICORN_WORKERS:-4}" --bind "${BIND_ADDRESS:-0.0.0.0:5000}" dashboard.server:app

