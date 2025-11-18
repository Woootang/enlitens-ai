#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${ROOT_DIR}/backups"
LEDGER_FILE="${ROOT_DIR}/data/knowledge_base/enliten_knowledge_base.jsonl"
VECTOR_DIR="${ROOT_DIR}/data/vector_store/chroma"
LOG_FILE="${ROOT_DIR}/logs/processing.log"

TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
LEDGER_ARCHIVE="${BACKUP_DIR}/ledger-${TIMESTAMP}.jsonl.gz"
VECTOR_ARCHIVE="${BACKUP_DIR}/chroma-${TIMESTAMP}.tar.gz"
LOG_ARCHIVE="${BACKUP_DIR}/logs-${TIMESTAMP}.tar.gz"

mkdir -p "${BACKUP_DIR}"

if [[ -f "${LEDGER_FILE}" ]]; then
  echo "→ Backing up ledger to ${LEDGER_ARCHIVE}"
  gzip -c "${LEDGER_FILE}" > "${LEDGER_ARCHIVE}"
else
  echo "⚠️ Ledger file ${LEDGER_FILE} not found; skipping."
fi

if [[ -d "${VECTOR_DIR}" ]]; then
  echo "→ Backing up vector store to ${VECTOR_ARCHIVE}"
  tar -czf "${VECTOR_ARCHIVE}" -C "${VECTOR_DIR}" .
else
  echo "⚠️ Vector store ${VECTOR_DIR} not found; skipping."
fi

if [[ -f "${LOG_FILE}" ]]; then
  echo "→ Backing up recent logs to ${LOG_ARCHIVE}"
  tar -czf "${LOG_ARCHIVE}" -C "${ROOT_DIR}" "logs"
fi

echo "Backup complete."

