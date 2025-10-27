#!/usr/bin/env python3
"""
Processing Monitor for Enlitens Multi-Agent System

This script monitors the progress of the multi-agent processing system
and provides real-time status updates without interfering with the main process.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Any
import argparse

async def check_system_status():
    """Check overall system status."""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processes": [],
        "gpu_status": {},
        "disk_space": {},
        "memory_usage": {}
    }

    try:
        # Check for Python processes
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'process_.*corpus'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            status["processes"] = [f"Process {pid}" for pid in pids if pid]

        # Check GPU status
        try:
            nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                                          '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True)

            if nvidia_result.returncode == 0:
                gpu_info = nvidia_result.stdout.strip().split(',')
                status["gpu_status"] = {
                    "name": gpu_info[0] if len(gpu_info) > 0 else "Unknown",
                    "memory_used": gpu_info[1] if len(gpu_info) > 1 else "0",
                    "memory_total": gpu_info[2] if len(gpu_info) > 2 else "0",
                    "utilization": gpu_info[3] if len(gpu_info) > 3 else "0",
                    "temperature": gpu_info[4] if len(gpu_info) > 4 else "0"
                }
        except FileNotFoundError:
            status["gpu_status"] = {"error": "nvidia-smi not found"}

        # Check disk space
        try:
            df_result = subprocess.run(['df', '-h', '.'],
                                     capture_output=True, text=True)

            if df_result.returncode == 0:
                lines = df_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        status["disk_space"] = {
                            "filesystem": parts[0],
                            "total": parts[1],
                            "used": parts[2],
                            "available": parts[3],
                            "use_percent": parts[4]
                        }
        except FileNotFoundError:
            status["disk_space"] = {"error": "df command not found"}

    except Exception as e:
        status["error"] = str(e)

    return status

async def check_processing_progress():
    """Check processing progress from log files and output files."""
    progress = {
        "log_files": [],
        "output_files": [],
        "latest_activity": None,
        "documents_processed": 0,
        "errors": [],
        "estimated_completion": None
    }

    try:
        # Find log files
        logs_dir = Path("logs")
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.log"):
                progress["log_files"].append(log_file.name)

                # Check latest activity in log file
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            progress["latest_activity"] = last_line
                except Exception as e:
                    progress["errors"].append(f"Error reading {log_file}: {e}")

        # Find output files
        for json_file in Path(".").glob("enlitens_knowledge_base*.json*"):
            progress["output_files"].append(json_file.name)

            # Check if it's a temp file or final
            if not json_file.name.endswith('.temp'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        progress["documents_processed"] = len(data.get("documents", []))
                except Exception as e:
                    progress["errors"].append(f"Error reading {json_file}: {e}")

        # Check temp files for current progress
        for temp_file in Path(".").glob("enlitens_knowledge_base*.json.temp"):
            try:
                with open(temp_file, 'r') as f:
                    data = json.load(f)
                    temp_count = len(data.get("documents", []))
                    if temp_count > progress["documents_processed"]:
                        progress["documents_processed"] = temp_count
            except Exception as e:
                progress["errors"].append(f"Error reading temp file: {e}")

        # Estimate completion (very rough)
        total_pdfs = 345  # Approximate number from logs
        if progress["documents_processed"] > 0:
            elapsed_ratio = progress["documents_processed"] / total_pdfs
            if elapsed_ratio > 0:
                estimated_total_time = 3600  # Assume 1 hour total
                estimated_remaining = estimated_total_time * (1 - elapsed_ratio)
                progress["estimated_completion"] = time.strftime(
                    "%H:%M:%S", time.gmtime(estimated_remaining)
                )

    except Exception as e:
        progress["errors"].append(f"Error checking progress: {e}")

    return progress

async def display_status():
    """Display comprehensive status."""
    print("🔍 ENLITENS MULTI-AGENT SYSTEM STATUS")
    print("=" * 50)

    # System status
    system_status = await check_system_status()

    print(f"📊 Timestamp: {system_status['timestamp']}")

    if system_status.get("processes"):
        print(f"⚡ Active Processes: {len(system_status['processes'])}")
        for process in system_status["processes"]:
            print(f"   • {process}")
    else:
        print("⚡ Active Processes: None")

    # GPU status
    gpu = system_status.get("gpu_status", {})
    if "error" not in gpu:
        print(f"🔥 GPU: {gpu.get('name', 'Unknown')}")
        print(f"   Memory: {gpu.get('memory_used', '0')}MB / {gpu.get('memory_total', '0')}MB")
        print(f"   Utilization: {gpu.get('utilization', '0')}%")
        print(f"   Temperature: {gpu.get('temperature', '0')}°C")
    else:
        print(f"🔥 GPU: {gpu['error']}")

    # Disk space
    disk = system_status.get("disk_space", {})
    if "error" not in disk:
        print(f"💽 Disk: {disk.get('available', '?')} available")
        print(f"   Usage: {disk.get('use_percent', '?')}")
    else:
        print(f"💽 Disk: {disk['error']}")

    print()

    # Processing progress
    progress = await check_processing_progress()

    print("📈 PROCESSING PROGRESS")
    print("-" * 30)

    if progress["output_files"]:
        print(f"📁 Output Files: {len(progress['output_files'])}")
        for file in progress["output_files"]:
            print(f"   • {file}")
    else:
        print("📁 Output Files: None")

    print(f"📊 Documents Processed: {progress['documents_processed']}")

    if progress["latest_activity"]:
        print(f"🕐 Latest Activity: {progress['latest_activity'][:100]}...")

    if progress["estimated_completion"]:
        print(f"⏰ Est. Completion: {progress['estimated_completion']} remaining")

    if progress["errors"]:
        print("❌ Errors Detected:")
        for error in progress["errors"]:
            print(f"   • {error}")

    print()
    print("💡 Tips:")
    print("   • Monitor logs: tail -f logs/*.log")
    print("   • Check GPU: nvidia-smi")
    print("   • View output: ls -la enlitens_knowledge_base*.json*")

async def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Monitor Multi-Agent Processing")
    parser.add_argument("--continuous", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")

    args = parser.parse_args()

    if args.continuous:
        print("🔄 Starting continuous monitoring...")
        print(f"📊 Updates every {args.interval} seconds")
        print("🛑 Press Ctrl+C to stop")
        print()

        try:
            while True:
                await display_status()
                print("\n" + "="*50 + "\n")
                await asyncio.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n✅ Monitoring stopped")
    else:
        await display_status()

if __name__ == "__main__":
    asyncio.run(main())
