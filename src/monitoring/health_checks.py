"""Operational health checks for Enlitens monitoring and inference services."""

import argparse
import asyncio
import sys
from typing import Dict

import httpx

from src.synthesis.ollama_client import VLLMClient, VLLM_DEFAULT_URL, VLLM_DEFAULT_MODEL


def _print_status(name: str, ok: bool, detail: str = "") -> None:
    icon = "✅" if ok else "❌"
    message = f"{icon} {name}"
    if detail:
        message += f" - {detail}"
    print(message)


async def check_processing_latency(base_url: str, threshold_seconds: int) -> bool:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{base_url}/api/stats")
        response.raise_for_status()
        data: Dict = response.json()
        latency = data.get("time_on_document_seconds", 0)
        ok = latency <= threshold_seconds
        detail = f"{latency:.1f}s" if latency else "no active document"
        _print_status("Document latency", ok, detail)
        return ok


async def check_foreman_response(base_url: str, query: str) -> bool:
    payload = {"query": query}
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(f"{base_url}/api/foreman/query", json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data.get("response", "").strip()
        ok = bool(reply)
        _print_status("Foreman responsiveness", ok, reply[:120])
        return ok


async def benchmark_vllm(url: str, model: str, prompt: str) -> bool:
    client = VLLMClient(base_url=url, default_model=model)
    try:
        results = await client.benchmark_batch_sizes(prompt)
    finally:
        await client.cleanup()
    detail = ", ".join(f"{size}:{duration:.2f}s" for size, duration in results.items())
    ok = all(duration < 60 for duration in results.values())
    _print_status("vLLM continuous batching", ok, detail)
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run Enlitens operational health checks")
    parser.add_argument("--monitor-url", default="http://localhost:8765", help="Monitoring server URL")
    parser.add_argument("--latency-threshold", type=int, default=600, help="Latency budget in seconds")
    parser.add_argument("--foreman-query", default="Provide a one line status update.", help="Query to send to the Foreman")
    parser.add_argument("--vllm-url", default=VLLM_DEFAULT_URL, help="vLLM OpenAI-compatible URL")
    parser.add_argument("--vllm-model", default=VLLM_DEFAULT_MODEL, help="Model for the vLLM benchmark")
    parser.add_argument("--vllm-prompt", default="Health check: respond with ok.", help="Prompt for batching test")
    args = parser.parse_args()

    base_url = args.monitor_url.rstrip('/')
    monitor_ok = await check_processing_latency(base_url, args.latency_threshold)
    foreman_ok = await check_foreman_response(base_url, args.foreman_query)
    vllm_ok = await benchmark_vllm(args.vllm_url, args.vllm_model, args.vllm_prompt)

    return 0 if all([monitor_ok, foreman_ok, vllm_ok]) else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except Exception as exc:  # pragma: no cover - operational guard
        _print_status("Health checks", False, str(exc))
        exit_code = 1
    sys.exit(exit_code)
