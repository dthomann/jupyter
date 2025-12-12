#!/usr/bin/env python3
"""Simple test with full output capture."""

import subprocess
import time
import sys

print("Starting Environment...")
env = subprocess.Popen(
    [sys.executable, 'run_env_server.py', '--port', '6000'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)

time.sleep(2)

print("Starting Brain1...")
brain1 = subprocess.Popen(
    [sys.executable, 'run_brain_client.py', '--name',
        'brain1', '--play-against-opponent'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)

time.sleep(3)

print("Starting Brain2...")
brain2 = subprocess.Popen(
    [sys.executable, 'run_brain_client.py', '--name',
        'brain2', '--play-against-opponent'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)

print("\n=== Waiting 8 seconds, then showing output ===\n")
time.sleep(8)

# Show recent output
print("=== ENV OUTPUT (last 30 lines) ===")
env_lines = []
for line in env.stdout:
    env_lines.append(line)
    if len(env_lines) > 30:
        env_lines.pop(0)
for line in env_lines[-30:]:
    print(line.rstrip())

print("\n=== BRAIN1 OUTPUT (last 20 lines) ===")
brain1_lines = []
for line in brain1.stdout:
    brain1_lines.append(line)
    if len(brain1_lines) > 20:
        brain1_lines.pop(0)
for line in brain1_lines[-20:]:
    print(line.rstrip())

print("\n=== BRAIN2 OUTPUT (last 20 lines) ===")
brain2_lines = []
for line in brain2.stdout:
    brain2_lines.append(line)
    if len(brain2_lines) > 20:
        brain2_lines.pop(0)
for line in brain2_lines[-20:]:
    print(line.rstrip())

# Cleanup
for p in [brain2, brain1, env]:
    try:
        p.terminate()
        p.wait(timeout=1)
    except:
        p.kill()


