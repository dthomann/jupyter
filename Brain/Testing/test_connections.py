#!/usr/bin/env python3
"""
Test script for connection system.
Tests various startup/shutdown combinations of environment and brains.
"""

import subprocess
import time
import signal
import os
import sys
from pathlib import Path
import threading
import queue


class Component:
    def __init__(self, name, cmd, log_file):
        self.name = name
        self.cmd = cmd
        self.log_file = log_file
        self.process = None
        self.log_queue = queue.Queue()
        self.log_thread = None

    def start(self):
        """Start the component."""
        print(f"[TEST] Starting {self.name}...")
        with open(self.log_file, 'w') as f:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        print(f"[TEST] {self.name} started (PID: {self.process.pid})")
        return self.process

    def stop(self):
        """Stop the component."""
        if self.process:
            print(f"[TEST] Stopping {self.name}...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[TEST] Force killing {self.name}...")
                self.process.kill()
                self.process.wait()
            self.process = None
            print(f"[TEST] {self.name} stopped")

    def is_running(self):
        """Check if component is running."""
        if self.process:
            return self.process.poll() is None
        return False

    def read_logs(self, lines=50):
        """Read recent log lines."""
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except FileNotFoundError:
            return []


def test_scenario(name, components, wait_time=3, check_interval=0.5):
    """Test a scenario with given components."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    # Start components in order
    for comp in components:
        comp.start()
        time.sleep(0.5)  # Small delay between starts

    # Wait and check
    print(f"[TEST] Waiting {wait_time} seconds for connections...")
    time.sleep(wait_time)

    # Check if all are still running
    all_running = all(comp.is_running() for comp in components)

    # Print logs
    print(
        f"\n[TEST] Status: {'All running' if all_running else 'Some failed'}")
    print(f"\n[TEST] Recent logs:")
    for comp in components:
        print(f"\n--- {comp.name} logs ---")
        logs = comp.read_logs(30)
        for line in logs[-20:]:  # Last 20 lines
            print(line.rstrip())

    # Stop all
    print(f"\n[TEST] Stopping all components...")
    for comp in reversed(components):  # Stop in reverse order
        comp.stop()
        time.sleep(0.3)

    time.sleep(1)  # Cleanup time
    return all_running


def main():
    """Run all test scenarios."""
    test_dir = Path("test_logs")
    test_dir.mkdir(exist_ok=True)

    # Clean up old log files
    for f in test_dir.glob("*.log"):
        f.unlink()

    # Define components
    env = Component(
        "Environment",
        [sys.executable, "run_env_server.py", "--port", "6000"],
        test_dir / "env.log"
    )

    brain1 = Component(
        "Brain1",
        [sys.executable, "run_brain_client.py", "--name",
            "brain1", "--play-against-opponent"],
        test_dir / "brain1.log"
    )

    brain2 = Component(
        "Brain2",
        [sys.executable, "run_brain_client.py", "--name",
            "brain2", "--play-against-opponent"],
        test_dir / "brain2.log"
    )

    results = []

    # Test 1: Environment first, then brains
    print("\n" + "="*70)
    print("TEST SUITE: Connection System")
    print("="*70)

    results.append((
        "1. Env -> Brain1 -> Brain2",
        test_scenario(
            "Environment starts first, then Brain1, then Brain2",
            [env, brain1, brain2],
            wait_time=5
        )
    ))

    time.sleep(2)

    # Test 2: Brain1 first, then env, then brain2
    results.append((
        "2. Brain1 -> Env -> Brain2",
        test_scenario(
            "Brain1 starts first, then Environment, then Brain2",
            [brain1, env, brain2],
            wait_time=5
        )
    ))

    time.sleep(2)

    # Test 3: Both brains first, then env
    results.append((
        "3. Brain1 -> Brain2 -> Env",
        test_scenario(
            "Both brains start first, then Environment",
            [brain1, brain2, env],
            wait_time=5
        )
    ))

    time.sleep(2)

    # Test 4: Env and Brain1, then Brain2 joins later
    results.append((
        "4. Env+Brain1 -> Brain2 joins",
        test_scenario(
            "Environment and Brain1 start, Brain2 joins later",
            [env, brain1],
            wait_time=3
        )
    ))
    # Now add brain2
    brain2.start()
    time.sleep(3)
    print(f"\n[TEST] Brain2 joined. Checking status...")
    for comp in [env, brain1, brain2]:
        print(f"\n--- {comp.name} logs (after Brain2 joined) ---")
        logs = comp.read_logs(20)
        for line in logs[-15:]:
            print(line.rstrip())
    brain2.stop()
    env.stop()
    brain1.stop()
    time.sleep(1)

    # Test 5: Single brain with environment (self-play mode)
    env_single = Component(
        "Environment (self-play)",
        [sys.executable, "run_env_server.py", "--port", "6001", "--self-play"],
        test_dir / "env_single.log"
    )
    brain_single = Component(
        "Brain (self-play)",
        [sys.executable, "run_brain_client.py", "--name", "brain_single"],
        test_dir / "brain_single.log"
    )

    results.append((
        "5. Self-play mode",
        test_scenario(
            "Environment and single brain (self-play)",
            [env_single, brain_single],
            wait_time=4
        )
    ))

    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print(f"\n{'='*70}")
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Check logs in test_logs/")
    print(f"{'='*70}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
        sys.exit(1)
