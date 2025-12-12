#!/usr/bin/env python3
"""Minimal test - just start brain2 and see what happens."""

import subprocess
import sys
import time

print("Starting brain2 standalone...")
proc = subprocess.Popen(
    [sys.executable, 'run_brain_client.py', '--name',
        'brain2', '--play-against-opponent'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print("Reading output for 5 seconds...")
start = time.time()
lines_read = 0
while time.time() - start < 5:
    line = proc.stdout.readline()
    if not line:
        time.sleep(0.1)
        continue
    lines_read += 1
    print(f"[{lines_read}] {line.rstrip()}")
    if lines_read > 50:
        break

print(f"\nRead {lines_read} lines. Terminating...")
proc.terminate()
try:
    proc.wait(timeout=2)
except:
    proc.kill()
