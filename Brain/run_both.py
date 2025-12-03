#!/usr/bin/env python3
"""
Run both environment server and brain client in separate processes.
Useful for testing and development.
"""

import subprocess
import sys
import time
import signal
import os


def main():
    print("=" * 70)
    print("Starting TicTacToe Environment Server and Brain Client")
    print("=" * 70)
    print()

    # Start environment server
    print("Starting environment server...")
    env_process = subprocess.Popen(
        [sys.executable, "run_env_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Wait a moment for server to start
    time.sleep(1)

    # Start brain client
    print("Starting brain client...")
    brain_process = subprocess.Popen(
        [sys.executable, "run_brain_client.py"] + sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        brain_process.terminate()
        env_process.terminate()
        brain_process.wait()
        env_process.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Print output from both processes
    print("\n" + "=" * 70)
    print("Output (Ctrl+C to stop):")
    print("=" * 70 + "\n")

    try:
        # Read from both processes
        while True:
            # Check if processes are still running
            if env_process.poll() is not None:
                print(
                    f"[env] Process exited with code {env_process.returncode}")
                break
            if brain_process.poll() is not None:
                print(
                    f"[brain] Process exited with code {brain_process.returncode}")
                break

            # Read from environment
            if env_process.stdout:
                line = env_process.stdout.readline()
                if line:
                    print(f"[env] {line.rstrip()}")

            # Read from brain
            if brain_process.stdout:
                line = brain_process.stdout.readline()
                if line:
                    print(f"[brain] {line.rstrip()}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        signal_handler(None, None)


if __name__ == '__main__':
    main()
