#!/usr/bin/env python3
"""Test the full flow: discovery -> connection -> assignment -> training start."""

import subprocess
import time
import sys
import threading
from collections import defaultdict


def monitor(proc, name, events):
    """Monitor process output."""
    try:
        for line in proc.stdout:
            line = line.rstrip()
            # Show all output for debugging
            if name in ['B1', 'B2'] and any(keyword in line for keyword in ['Sending', 'tick()', 'Training started', 'Calling tick', 'initial action']):
                print(f"[{name}] {line}")
            if 'both players connected' in line:
                events['both_connected'] = True
                events['messages'].append((name, line))
            elif 'Assigned brain' in line:
                events['messages'].append((name, line))
            elif 'games played' in line or ('Episode' in line and 'Training started' not in line):
                events['training_started'] = True
                events['messages'].append((name, line))
            elif 'Training started' in line:
                # Brain received observation = training started
                if 'training_started' not in events or not events['training_started']:
                    events['training_started'] = True
                events['messages'].append((name, line))
            elif 'Sending initial action' in line or 'tick() returned None' in line:
                events['messages'].append((name, line))
    except:
        pass


def test_full_flow():
    """Test complete flow."""
    print("="*70)
    print("FULL FLOW TEST")
    print("="*70)

    events = {'both_connected': False,
              'training_started': False, 'messages': []}

    # Start all components
    print("\nStarting Environment...")
    env = subprocess.Popen(
        [sys.executable, 'run_env_server.py', '--port', '6000'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    env_monitor = threading.Thread(
        target=monitor, args=(env, 'ENV', events), daemon=True)
    env_monitor.start()
    time.sleep(2)

    print("Starting Brain1...")
    brain1 = subprocess.Popen(
        [sys.executable, 'run_brain_client.py', '--name',
            'brain1', '--play-against-opponent'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    brain1_monitor = threading.Thread(
        target=monitor, args=(brain1, 'B1', events), daemon=True)
    brain1_monitor.start()
    time.sleep(3)

    print("Starting Brain2...")
    brain2 = subprocess.Popen(
        [sys.executable, 'run_brain_client.py', '--name',
            'brain2', '--play-against-opponent'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    brain2_monitor = threading.Thread(
        target=monitor, args=(brain2, 'B2', events), daemon=True)
    brain2_monitor.start()

    # Wait up to 15 seconds for training to start
    print("\nWaiting for training to start (max 15 seconds)...")
    for i in range(30):
        time.sleep(0.5)
        if events['both_connected'] and events['training_started']:
            break
        if i % 4 == 0:
            print(f"  ... {i*0.5:.1f}s")

    # Stop
    print("\nStopping processes...")
    for p in [brain2, brain1, env]:
        try:
            p.terminate()
            p.wait(timeout=2)
        except:
            p.kill()

    time.sleep(1)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(
        f"\nBoth players connected: {'✓' if events['both_connected'] else '✗'}")
    print(f"Training started: {'✓' if events['training_started'] else '✗'}")

    print(f"\nKey messages:")
    for name, msg in events['messages']:
        print(f"  {name}: {msg}")

    success = events['both_connected'] and events['training_started']
    print(
        f"\nOverall: {'✓ PASS - System is working!' if success else '✗ FAIL - Training did not start'}")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(test_full_flow())
