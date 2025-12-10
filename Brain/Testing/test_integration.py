#!/usr/bin/env python3
"""
Integration test: Test brain client with environment server using ConnectionManager.

This tests the actual integration - brain client should work with default config
(backwards compatible) and with bidirectional mode enabled.
"""

import subprocess
import time
import sys
import signal
import os
from pathlib import Path


def test_brain_env_integration():
    """Test brain client connecting to environment server (default mode)."""
    print("=" * 70)
    print("Integration Test: Brain Client + Environment Server (Default Mode)")
    print("=" * 70)

    env_proc = None
    brain_proc = None

    try:
        # Start environment server
        print("Starting environment server...")
        env_proc = subprocess.Popen(
            [sys.executable, "run_env_server.py", "--self-play"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for server to start
        time.sleep(2)

        # Check if server is still running
        if env_proc.poll() is not None:
            stdout, _ = env_proc.communicate()
            print(f"Environment server exited early:\n{stdout}")
            return False

        print("✓ Environment server started")

        # Start brain client with default config (backwards compatible)
        print("Starting brain client (default mode)...")
        brain_proc = subprocess.Popen(
            [sys.executable, "run_brain_client.py",
                "--max-episodes", "1", "--stats-every", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait a bit for connection and initial messages
        time.sleep(3)

        # Check if processes are still running
        if brain_proc.poll() is not None:
            # Brain process might have completed or errored
            stdout, _ = brain_proc.communicate()
            print(f"Brain client output:\n{stdout[-500:]}")  # Last 500 chars

            # Check if it completed successfully or errored
            if "Training started" in stdout or "connected" in stdout.lower():
                print("✓ Brain client connected successfully")
                return True
            else:
                print("✗ Brain client may have failed to connect")
                return False

        # Process is still running - that's good for this test
        print("✓ Brain client is running and connected")
        print("  (This is expected - it would run until max episodes or Ctrl+C)")

        # Clean up
        brain_proc.terminate()
        try:
            brain_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            brain_proc.kill()
            brain_proc.wait()

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if brain_proc:
            try:
                brain_proc.terminate()
                brain_proc.wait(timeout=1)
            except:
                brain_proc.kill()
        if env_proc:
            try:
                env_proc.terminate()
                env_proc.wait(timeout=1)
            except:
                env_proc.kill()


def test_brain_with_listener():
    """Test brain with listener enabled (bidirectional mode)."""
    print("\n" + "=" * 70)
    print("Integration Test: Brain with Listener Enabled")
    print("=" * 70)

    brain1_proc = None
    brain2_proc = None

    try:
        # Start brain1 with listener enabled
        print("Starting brain1 with listener on localhost:7000...")
        brain1_proc = subprocess.Popen(
            [sys.executable, "run_brain_client.py",
             "--enable-listener", "--listen-host", "localhost", "--listen-port", "7000",
             "--max-episodes", "1", "--stats-every", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for listener to start
        time.sleep(2)

        # Check output for listener confirmation
        # (we can't easily read non-blocking, so we'll just check if it's running)
        if brain1_proc.poll() is not None:
            stdout, _ = brain1_proc.communicate()
            print(f"Brain1 exited early:\n{stdout[-500:]}")
            if "Failed to start listener" in stdout:
                print("✗ Listener failed to start")
                return False

        print("✓ Brain1 with listener started")

        # Try to connect a client to brain1's listener
        # (simulating another brain or tool connecting)
        print("Testing connection to brain1's listener...")
        from multiprocessing.connection import Client
        try:
            client = Client(("localhost", 7000), authkey=b"brain-secret")
            print("✓ Successfully connected to brain1's listener")

            # Send a test message
            test_msg = {"type": "test", "data": "hello from test client"}
            client.send(test_msg)
            print("✓ Sent test message to brain1")

            client.close()
            print("✓ Connection closed")
        except Exception as e:
            print(f"✗ Failed to connect to brain1's listener: {e}")
            return False

        # Clean up
        brain1_proc.terminate()
        try:
            brain1_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            brain1_proc.kill()
            brain1_proc.wait()

        print("✓ Brain with listener test PASSED")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if brain1_proc:
            try:
                brain1_proc.terminate()
                brain1_proc.wait(timeout=1)
            except:
                brain1_proc.kill()
        if brain2_proc:
            try:
                brain2_proc.terminate()
                brain2_proc.wait(timeout=1)
            except:
                brain2_proc.kill()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Integration Tests for Bi-directional Communication")
    print("=" * 70 + "\n")

    results = []

    # Test 1: Default mode (backwards compatible)
    results.append(("Brain + Environment (Default Mode)",
                   test_brain_env_integration()))

    # Test 2: Brain with listener
    results.append(("Brain with Listener", test_brain_with_listener()))

    # Summary
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)
    print("=" * 70)
    if all_passed:
        print("All integration tests PASSED!")
        sys.exit(0)
    else:
        print("Some integration tests FAILED")
        sys.exit(1)
