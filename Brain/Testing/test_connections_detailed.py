#!/usr/bin/env python3
"""Test connection establishment specifically."""

import subprocess
import time
import sys
import threading
from collections import defaultdict


def monitor_output(proc, name, events):
    """Monitor process output."""
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if 'Connected to' in line and 'brain' in line.lower():
                events['env_connected'].append((name, line))
            elif 'Connected to' in line and 'environment' in line.lower():
                events['brain_connected'].append((name, line))
            elif 'Assigned brain' in line:
                events['assigned'].append((name, line))
            elif 'both players connected' in line:
                events['both_connected'].append((name, line))
            elif 'Found.*connected brains' in line or 'Debug: Found' in line:
                events['found_brains'].append((name, line))
            elif 'Error' in line or 'Failed' in line:
                events['errors'].append((name, line))
    except:
        pass


def test_connections():
    """Test connection establishment."""
    print("="*70)
    print("CONNECTION ESTABLISHMENT TEST")
    print("="*70)

    events = defaultdict(list)

    # Start environment
    print("\n[1/3] Starting Environment...")
    env = subprocess.Popen(
        [sys.executable, 'run_env_server.py', '--port', '6000'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    env_monitor = threading.Thread(
        target=monitor_output, args=(env, 'ENV', events), daemon=True)
    env_monitor.start()
    time.sleep(2)

    # Start brain1
    print("[2/3] Starting Brain1...")
    brain1 = subprocess.Popen(
        [sys.executable, 'run_brain_client.py', '--name',
            'brain1', '--play-against-opponent'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    brain1_monitor = threading.Thread(
        target=monitor_output, args=(brain1, 'B1', events), daemon=True)
    brain1_monitor.start()
    time.sleep(3)

    # Start brain2
    print("[3/3] Starting Brain2...")
    brain2 = subprocess.Popen(
        [sys.executable, 'run_brain_client.py', '--name',
            'brain2', '--play-against-opponent'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    brain2_monitor = threading.Thread(
        target=monitor_output, args=(brain2, 'B2', events), daemon=True)
    brain2_monitor.start()
    time.sleep(8)

    # Stop processes
    print("\nStopping processes...")
    for p in [brain2, brain1, env]:
        try:
            p.terminate()
            p.wait(timeout=2)
        except:
            p.kill()

    time.sleep(1)

    # Analyze
    print("\n" + "="*70)
    print("CONNECTION ANALYSIS")
    print("="*70)

    print(f"\n[Environment -> Brain Connections]")
    unique_connections = set()
    for name, line in events['env_connected']:
        print(f"  {line}")
        # Extract brain ID from line
        if 'brain_' in line:
            brain_id = line.split('brain_')[1].split()[
                0] if 'brain_' in line else 'unknown'
            unique_connections.add(brain_id)
    print(
        f"  Unique brain connections: {len(unique_connections)} (expected: 2)")

    print(f"\n[Environment Found Connected Brains]")
    for name, line in events['found_brains'][-10:]:  # Last 10
        print(f"  {line}")

    print(f"\n[Brain Assignments]")
    for name, line in events['assigned']:
        print(f"  {line}")

    print(f"\n[Both Players Connected]")
    for name, line in events['both_connected']:
        print(f"  {line}")

    if events['errors']:
        print(f"\n[ERRORS]")
        for name, line in events['errors'][:10]:
            print(f"  {name}: {line}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(
        f"Environment connected to unique brains: {len(unique_connections)} (expected: 2)")
    print(f"Brains assigned: {len(events['assigned'])} (expected: 2)")
    print(
        f"Both players connected: {'✓' if events['both_connected'] else '✗'}")

    success = len(unique_connections) >= 2 and len(
        events['assigned']) >= 2 and events['both_connected']
    print(f"\nOverall: {'✓ PASS' if success else '✗ FAIL'}")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(test_connections())
