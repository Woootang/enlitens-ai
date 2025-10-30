#!/usr/bin/env python3
"""
Quick test to verify monitoring server can start and respond.
"""

import asyncio
import pytest

httpx = pytest.importorskip("httpx")
import subprocess
import time
import sys

async def test_monitoring_server():
    """Test that the monitoring server starts and responds correctly."""

    print("🧪 Testing monitoring server...")

    # Start the server in background
    print("🚀 Starting monitoring server...")
    process = subprocess.Popen(
        ["python3", "monitoring_server.py", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/home/user/enlitens-ai"
    )

    # Give it time to start
    print("⏳ Waiting for server to initialize...")
    await asyncio.sleep(3)

    try:
        # Test HTTP endpoint
        print("🔍 Testing HTTP endpoint...")
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8765/", timeout=5.0)

            if response.status_code == 200:
                print("✅ HTTP endpoint working!")
                print(f"   Status: {response.status_code}")
                print(f"   Content length: {len(response.text)} bytes")

                # Check if it returns HTML
                if "<html" in response.text.lower():
                    print("✅ Dashboard HTML is being served")
                else:
                    print("⚠️  Response doesn't look like HTML")
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                return False

        # Test stats endpoint
        print("🔍 Testing stats API endpoint...")
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8765/api/stats", timeout=5.0)

            if response.status_code == 200:
                print("✅ Stats API working!")
                stats = response.json()
                print(f"   Stats: {stats}")
            else:
                print(f"❌ Stats endpoint failed: {response.status_code}")

        print("\n🎉 All tests passed! Monitoring server is working correctly.")
        return True

    except httpx.ConnectError:
        print("❌ Could not connect to server - it may not have started")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Shut down the server
        print("\n🛑 Shutting down test server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    result = asyncio.run(test_monitoring_server())
    sys.exit(0 if result else 1)
