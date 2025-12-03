#!/usr/bin/env python3
"""
Simple test to verify brain and environment can connect and exchange messages.
"""

import subprocess
import time
import sys

def test_connection():
    print("Testing brain-environment connection...")
    print("=" * 70)
    
    # Start environment
    print("Starting environment server...")
    env = subprocess.Popen(
        [sys.executable, "run_env_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to start
    time.sleep(2)
    
    # Start brain
    print("Starting brain client...")
    brain = subprocess.Popen(
        [sys.executable, "run_brain_client.py", "--lr", "0.001", "--entropy", "0.001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Let them run for a few seconds
    print("Running for 10 seconds...")
    time.sleep(10)
    
    # Check if processes are still running
    env_alive = env.poll() is None
    brain_alive = brain.poll() is None
    
    print(f"\nEnvironment alive: {env_alive}")
    print(f"Brain alive: {brain_alive}")
    
    if not brain_alive:
        print("\nBrain output:")
        print(brain.stdout.read())
    
    # Cleanup
    env.terminate()
    if brain_alive:
        brain.terminate()
    env.wait()
    if brain_alive:
        brain.wait()
    
    print("\nTest complete!")

if __name__ == '__main__':
    test_connection()

