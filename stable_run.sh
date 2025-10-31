#!/usr/bin/env python3
"""Resilient launcher for the Enlitens multi-agent processing pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.utils.settings import get_settings

LOG_DIR = Path("logs")
VLLM_VENV = Path(".venv_vllm")
DEFAULT_MAIN_PORT = 8000
DEFAULT_MONITOR_PORT = 8001


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _local_weights_available(path: Optional[str]) -> bool:
    if not path:
        return False
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        return any(resolved.glob("**/*.bin")) or any(resolved.glob("**/*.safetensors"))
    return resolved.exists()


def _has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def _ensure_virtualenv(path: Path) -> Optional[Path]:
    python_bin = path / "bin" / "python"
    if python_bin.exists():
        return python_bin
    logging.info("Creating virtual environment at %s", path)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime guard
        logging.warning("Failed to create virtual environment: %s", exc)
        return None
    return path / "bin" / "python"


def _start_vllm_server(model_path: str, port: int, log_file: Path, gpu_util: float = 0.7) -> Optional[subprocess.Popen]:
    python_bin = _ensure_virtualenv(VLLM_VENV)
    if not python_bin:
        logging.warning("vLLM virtual environment unavailable; skipping local server startup")
        return None

    flags = [
        str(python_bin),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--dtype",
        "auto",
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "8192",
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "--enable-chunked-prefill",
        "--kv-cache-dtype",
        "auto",
    ]

    if _has_gpu():
        flags.extend([
            "--gpu-memory-utilization",
            str(gpu_util),
            "--max-num-seqs",
            "24",
            "--enforce-eager",
        ])
    else:
        logging.info("GPU not detected; launching server without GPU specific flags")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("w")
    logging.info("Starting vLLM server for %s on port %s", model_path, port)
    process = subprocess.Popen(flags, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    time.sleep(5)
    if process.poll() is not None:
        logging.error("vLLM server exited unexpectedly; see %s", log_file)
        return None
    return process


def _launch_pipeline(output_file: Path, log_file: Path, base_dir: Path) -> subprocess.Popen:
    command = [
        sys.executable,
        "process_multi_agent_corpus.py",
        "--input-dir",
        "enlitens_corpus/input_pdfs",
        "--output-file",
        str(output_file),
        "--st-louis-report",
        "st_louis_health_report.pdf",
    ]
    env = os.environ.copy()
    env["DOCLING_FORCE_CPU"] = "true"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("w")
    logging.info("Launching processing pipeline -> %s", output_file)
    return subprocess.Popen(command, cwd=str(base_dir), stdout=log_handle, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Enlitens processing pipeline with automatic backend selection")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only start backend services without running the pipeline")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    LOG_DIR.mkdir(exist_ok=True)
    for pattern in ("*.json", "*.json.temp", "*.log"):
        for file in Path.cwd().glob(pattern):
            try:
                file.unlink()
            except OSError:
                pass

    settings = get_settings()
    llm_settings = settings.llm
    local_weights_path = llm_settings.local_weights_path
    backend_provider = llm_settings.provider

    logging.info("Detected provider=%s base_url=%s", backend_provider, llm_settings.base_url)

    server_process: Optional[subprocess.Popen] = None
    base_dir = Path(__file__).resolve().parent

    if _local_weights_available(local_weights_path):
        logging.info("Local weights found at %s", local_weights_path)
        server_process = _start_vllm_server(
            model_path=str(Path(local_weights_path).expanduser()),
            port=DEFAULT_MAIN_PORT,
            log_file=LOG_DIR / "vllm-main.log",
        )
    else:
        logging.info("No local weights detected; relying on remote provider %s", backend_provider)

    if args.skip_pipeline:
        logging.info("Pipeline start skipped by user request")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"enlitens_knowledge_base_complete_{timestamp}.json")
    pipeline_log = LOG_DIR / "temp_processing.log"
    process = _launch_pipeline(output_file, pipeline_log, base_dir)

    logging.info("Process started with PID %s", process.pid)
    logging.info("Output target: %s", output_file)
    logging.info("Log file: %s", pipeline_log)

    if server_process is None:
        logging.info("No local vLLM server managed by this launcher")
    else:
        logging.info("Local vLLM server running with PID %s", server_process.pid)


if __name__ == "__main__":
    main()
