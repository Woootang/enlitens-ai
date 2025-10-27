#!/usr/bin/env python3
"""
Progress Monitor for Enlitens Multi-Agent System

This script monitors the progress of the multi-agent processing without relying on logs.
"""

import json
import os
from pathlib import Path
from datetime import datetime

def check_system_status():
    """Check overall system status."""
    print("🔍 ENLITENS MULTI-AGENT SYSTEM STATUS")
    print("=" * 50)

    # Check if process is running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'process_multi_agent'], capture_output=True)
    if result.returncode == 0:
        print("✅ Multi-agent processing is RUNNING")
        print(f"   Process ID: {result.stdout.decode().strip()}")
    else:
        print("❌ Multi-agent processing is NOT running")
        return

    # Check GPU usage
    try:
        nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                                      '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True)

        if nvidia_result.returncode == 0:
            gpu_info = nvidia_result.stdout.strip().split(',')
            print("🔥 GPU Status:")
            print(f"   Memory: {gpu_info[0]}MB / {gpu_info[1]}MB")
            print(f"   Utilization: {gpu_info[2]}%")
            print(f"   Temperature: {gpu_info[3]}°C")
    except:
        print("🔥 GPU: Unable to check status")

    # Check output files
    print("\n📁 Output Files:")
    output_files = list(Path(".").glob("*.json"))
    for file in sorted(output_files):
        if file.name.startswith("debug_output") or file.name.startswith("enlitens_knowledge_base"):
            size = file.stat().st_size
            print(f"   {file.name}: {size:,} bytes")

            # Check if it's a temp file
            if file.name.endswith('.temp'):
                print("   📝 Status: Processing in progress (temp file)")
            else:
                print("   ✅ Status: Processing completed")

                # Try to load and show progress
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        doc_count = len(data.get('documents', []))
                        total_docs = data.get('total_documents', 0)
                        print(f"   📊 Progress: {doc_count}/{total_docs} documents processed")
                except:
                    print("   📊 Progress: Unable to read")

    # Check log files
    print("\n📝 Log Files:")
    log_files = list(Path(".").glob("*.log")) + list(Path("logs").glob("*.log"))
    for log_file in sorted(log_files):
        size = log_file.stat().st_size
        if size > 0:
            print(f"   {log_file.name}: {size:,} bytes")
            # Show last few lines
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        print(f"   📄 Last entry: {last_line[:100]}...")
            except:
                print("   📄 Last entry: Unable to read")
        else:
            print(f"   {log_file.name}: {size:,} bytes (empty)")

    print(f"\n⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Tips:")
    print("   • The enhanced system processes documents through multiple AI agents")
    print("   • Each document takes 10-20 minutes with quality validation")
    print("   • GPU memory usage indicates active processing")
    print("   • Check progress every 15-30 minutes")

if __name__ == "__main__":
    check_system_status()
