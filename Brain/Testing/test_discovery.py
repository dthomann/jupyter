#!/usr/bin/env python3
"""
Test discovery mechanism specifically.
Since brain1 and brain2 are identical, focus on whether:
1. Both brains send HELLO messages
2. Environment receives HELLO from both
3. Discovery callbacks are triggered for both
"""

import subprocess
import time
import sys
import threading
from collections import defaultdict


def monitor_output(proc, name, events):
    """Monitor process output and collect discovery-related events."""
    try:
        for line in proc.stdout:
            line = line.rstrip()
            # Track key discovery events
            if 'sending first HELLO' in line:
                events['hello_sent'].append((name, line))
            elif 'Discovered new' in line:
                events['discovered'].append((name, line))
            elif 'discovered' in line.lower() and 'ConnectionManager' in line:
                events['callback'].append((name, line))
            elif 'will connect' in line or 'Starting connection' in line:
                events['connection_attempt'].append((name, line))
            elif 'Connected to' in line and 'brain' in line:
                events['connected'].append((name, line))
            elif 'Error' in line or 'Failed' in line:
                events['errors'].append((name, line))
    except Exception as e:
        events['errors'].append((name, f"Monitor error: {e}"))


def test_discovery():
    """Test discovery with detailed monitoring."""
    print("="*70)
    print("DISCOVERY MECHANISM TEST")
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
    time.sleep(3)  # Give brain1 time to start and send HELLO

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
    time.sleep(8)  # Wait for discovery and connections

    # Stop processes
    print("\nStopping processes...")
    for p in [brain2, brain1, env]:
        try:
            p.terminate()
            p.wait(timeout=2)
        except:
            p.kill()

    time.sleep(1)  # Let monitors finish

    # Analyze results
    print("\n" + "="*70)
    print("DISCOVERY ANALYSIS")
    print("="*70)

    print(f"\n[HELLO Messages Sent]")
    for name, line in events['hello_sent']:
        print(f"  {name}: {line}")
    print(f"  Total: {len(events['hello_sent'])} (expected: 2 - B1 and B2)")

    print(f"\n[Discovered Messages (MulticastDiscovery)]")
    for name, line in events['discovered']:
        print(f"  {name}: {line}")
    print(f"  Total: {len(events['discovered'])} (expected: 2 - both brains)")

    print(f"\n[Discovery Callbacks (ConnectionManager)]")
    for name, line in events['callback']:
        print(f"  {name}: {line}")
    print(f"  Total: {len(events['callback'])} (expected: 2 - both brains)")

    print(f"\n[Connection Attempts]")
    for name, line in events['connection_attempt']:
        print(f"  {name}: {line}")
    print(
        f"  Total: {len(events['connection_attempt'])} (expected: 2 - both brains)")

    print(f"\n[Successful Connections]")
    for name, line in events['connected']:
        print(f"  {name}: {line}")
    print(f"  Total: {len(events['connected'])} (expected: 2 - both brains)")

    if events['errors']:
        print(f"\n[ERRORS]")
        for name, line in events['errors']:
            print(f"  {name}: {line}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    brain1_hello = sum(1 for name, _ in events['hello_sent'] if name == 'B1')
    brain2_hello = sum(1 for name, _ in events['hello_sent'] if name == 'B2')
    env_discovered = sum(
        1 for name, _ in events['discovered'] if name == 'ENV')
    env_callbacks = sum(
        1 for name, _ in events['callback'] if name == 'ENV' and 'brain' in str(events['callback']))

    print(f"Brain1 HELLO sent: {'✓' if brain1_hello > 0 else '✗'}")
    print(f"Brain2 HELLO sent: {'✓' if brain2_hello > 0 else '✗'}")
    print(f"Environment discovered brains: {env_discovered} (expected: 2)")
    print(
        f"Environment connection callbacks: {len([e for e in events['callback'] if e[0] == 'ENV'])}")
    print(
        f"Environment connection attempts: {len(events['connection_attempt'])} (expected: 2)")
    print(
        f"Environment successful connections: {len(events['connected'])} (expected: 2)")

    success = (brain1_hello > 0 and brain2_hello > 0 and
               env_discovered >= 2 and len(events['connected']) >= 2)

    print(f"\nOverall: {'✓ PASS' if success else '✗ FAIL'}")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(test_discovery())
