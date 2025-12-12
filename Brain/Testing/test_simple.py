#!/usr/bin/env python3
"""Simple test to debug brain2 discovery issue."""

import subprocess
import time
import sys

print("Starting Environment...")
env = subprocess.Popen([sys.executable, 'run_env_server.py', '--port', '6000'],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
time.sleep(2)

print("Starting Brain1...")
brain1 = subprocess.Popen([sys.executable, 'run_brain_client.py', '--name', 'brain1', '--play-against-opponent'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
time.sleep(3)

print("Starting Brain2...")
brain2 = subprocess.Popen([sys.executable, 'run_brain_client.py', '--name', 'brain2', '--play-against-opponent'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
time.sleep(8)

print("\n=== ENVIRONMENT OUTPUT (key lines) ===")
env_output = []
try:
    for _ in range(200):
        line = env.stdout.readline()
        if not line:
            break
        env_output.append(line)
        if any(keyword in line for keyword in ['Discovered', 'Connected', 'Assigned', 'both players', 'Debug: Found']):
            print(line.rstrip())
except:
    pass

print("\n=== Checking if both brains discovered ===")
discovered_brains = [
    line for line in env_output if 'Discovered new brain' in line]
print(f"Found {len(discovered_brains)} 'Discovered new brain' messages:")
for line in discovered_brains:
    print(f"  {line.rstrip()}")

print("\n=== Checking connection attempts ===")
connection_attempts = [
    line for line in env_output if 'will connect' in line or 'Starting connection' in line or 'Connected to' in line]
for line in connection_attempts:
    print(f"  {line.rstrip()}")

# Cleanup
print("\nStopping processes...")
for p in [brain2, brain1, env]:
    try:
        p.terminate()
        p.wait(timeout=2)
    except:
        p.kill()
