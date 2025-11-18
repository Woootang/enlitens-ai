#!/usr/bin/env python3
"""
CLI wrapper around LocalModelManager.

Usage:
    python scripts/manage_local_model.py start --model medgemma
    python scripts/manage_local_model.py stop --model llama
    python scripts/manage_local_model.py describe --model medgemma
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.utils.local_model_manager import LocalModelManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local vLLM/SGLang model servers.")
    parser.add_argument("action", choices={"start", "stop", "describe"}, help="Action to perform")
    parser.add_argument("--model", required=True, help="Model key defined in config/local_models.yaml")
    return parser


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    manager = LocalModelManager()
    if args.action == "start":
        manager.start(args.model)
    elif args.action == "stop":
        manager.stop(args.model)
        manager.clear_gpu()
    elif args.action == "describe":
        print(manager.describe(args.model))
    else:
        parser.error(f"Unknown action {args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

