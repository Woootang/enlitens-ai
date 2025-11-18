#!/usr/bin/env bash
#
# Bootstrap a vLLM server hosting Llama 3.1 8B Instruct on port 8000.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/start_local_model.sh" llama

