#!/usr/bin/env python3
"""Test to see if brain2 starts and sends HELLO messages."""

import subprocess
import time
import sys
import threading


def read_output(proc, name, output_list):
    """Read output from a process."""
    try:
        for line in proc.stdout:
            output_list.append((name, line))
            if len(output_list) > 100:
                break
    except:
        pass


# Start environment
print("Starting Environment...")
env = subprocess.Popen([sys.executable, 'run_env_server.py', '--port', '6000'],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
time.sleep(2)

# Start brain1
print("Starting Brain1...")
brain1 = subprocess.Popen([sys.executable, 'run_brain_client.py', '--name', 'brain1', '--play-against-opponent'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
time.sleep(3)

# Start brain2
print("Starting Brain2...")
brain2 = subprocess.Popen([sys.executable, 'run_brain_client.py', '--name', 'brain2', '--play-against-opponent'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# Collect output in background
output = []
threads = []
for name, proc in [('ENV', env), ('B1', brain1), ('B2', brain2)]:
    t = threading.Thread(target=read_output, args=(
        proc, name, output), daemon=True)
    t.start()
    threads.append(t)

time.sleep(8)

print("\n=== BRAIN2 OUTPUT ===")
brain2_lines = [line for name, line in output if name == 'B2']
for line in brain2_lines[:30]:
    print(line.rstrip())

print("\n=== ENVIRONMENT DISCOVERY ===")
env_lines = [line for name, line in output if name == 'ENV' and (
    'Discovered' in line or 'will connect' in line or 'Connected to' in line)]
for line in env_lines:
    print(line.rstrip())

# Cleanup
for p in [brain2, brain1, env]:
    try:
        p.terminate()
        p.wait(timeout=2)
    except:
        p.kill()
