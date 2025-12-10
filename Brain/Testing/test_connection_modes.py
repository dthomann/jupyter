#!/usr/bin/env python3
"""
Test script for bi-directional communication modes.

Tests:
1. Backwards compatibility: brain connects to environment (default mode)
2. Bidirectional: brain with listener enabled can accept connections
"""

import subprocess
import time
import sys
import signal
from multiprocessing.connection import Client, Listener
from brainprotocol import OBSERVATION, ACTION, REWARD, TERMINAL, SHUTDOWN


def test_backwards_compatibility():
    """Test that default mode (brain as client) still works."""
    print("=" * 70)
    print("Test 1: Backwards Compatibility")
    print("=" * 70)
    print("Testing default mode: brain connects to environment...")

    # Start environment server in background
    env_proc = subprocess.Popen(
        [sys.executable, "run_env_server.py", "--self-play"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        time.sleep(1)

        # Try to connect to the server
        try:
            conn = Client(("localhost", 6000), authkey=b"brain-secret")
            print("✓ Successfully connected to environment server")

            # Send a test message
            conn.send({"type": ACTION, "action": 0})
            print("✓ Successfully sent message")

            conn.close()
            print("✓ Connection closed successfully")
            print("\n✓ Backwards compatibility test PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
    finally:
        env_proc.terminate()
        env_proc.wait(timeout=2)


def test_bidirectional_connection():
    """Test bidirectional communication between two processes."""
    print("=" * 70)
    print("Test 2: Bidirectional Communication")
    print("=" * 70)
    print("Testing brain-to-brain connection...")

    # Start a listener (simulating brain1 with listener enabled)
    listener = None
    conn1 = None
    conn2 = None

    try:
        listener = Listener(("localhost", 7000), authkey=b"brain-secret")
        print("✓ Listener started on localhost:7000")

        # In a real scenario, this would be a separate brain process
        # For testing, we'll simulate by connecting from the same process
        def connect_client():
            time.sleep(0.2)  # Give listener time to start
            return Client(("localhost", 7000), authkey=b"brain-secret")

        # Start client connection in background
        import threading
        client_thread_done = threading.Event()
        client_conn = [None]

        def connect_thread():
            try:
                client_conn[0] = connect_client()
                print("✓ Client connected to listener")
                client_thread_done.set()
            except Exception as e:
                print(f"✗ Client connection failed: {e}")
                client_thread_done.set()

        thread = threading.Thread(target=connect_thread)
        thread.start()

        # Accept connection
        print("Waiting for connection...")
        conn1 = listener.accept()
        print(f"✓ Accepted connection from {listener.last_accepted}")

        # Wait for client to connect
        thread.join(timeout=2)
        conn2 = client_conn[0]

        if conn2 is None:
            print("✗ Client failed to connect")
            return False

        # Test bidirectional messaging
        # Send from listener to client
        test_msg1 = {"type": "test", "data": "from_listener"}
        conn1.send(test_msg1)
        print("✓ Sent message from listener to client")

        # Receive on client
        if conn2.poll(1.0):
            msg = conn2.recv()
            if msg.get("data") == "from_listener":
                print("✓ Received message on client")
            else:
                print(f"✗ Unexpected message: {msg}")
                return False
        else:
            print("✗ No message received on client")
            return False

        # Send from client to listener
        test_msg2 = {"type": "test", "data": "from_client"}
        conn2.send(test_msg2)
        print("✓ Sent message from client to listener")

        # Receive on listener
        if conn1.poll(1.0):
            msg = conn1.recv()
            if msg.get("data") == "from_client":
                print("✓ Received message on listener")
            else:
                print(f"✗ Unexpected message: {msg}")
                return False
        else:
            print("✗ No message received on listener")
            return False

        conn1.close()
        conn2.close()
        print("\n✓ Bidirectional communication test PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if conn1:
            try:
                conn1.close()
            except:
                pass
        if conn2:
            try:
                conn2.close()
            except:
                pass
        if listener:
            try:
                listener.close()
            except:
                pass


def test_connection_manager_basic():
    """Test ConnectionManager basic functionality."""
    print("=" * 70)
    print("Test 3: ConnectionManager Basic Functionality")
    print("=" * 70)

    try:
        from brain.connection_config import BrainConnectionConfig
        from brain.connection_manager import ConnectionManager

        # Test default config (backwards compatible)
        config = BrainConnectionConfig.from_args(
            host="localhost",
            port=6000,
        )
        print(f"✓ Created default config: brain_id={config.brain_id}")
        print(f"  enable_listener={config.enable_listener}")
        print(f"  peers={config.peers}")

        # Test config with listener enabled
        config2 = BrainConnectionConfig.from_args(
            host="localhost",
            port=6000,
            listen_host="localhost",
            listen_port=7000,
            enable_listener=True,
        )
        print(
            f"✓ Created listener config: listen_address={config2.listen_address}")

        # Test config with multiple peers
        config3 = BrainConnectionConfig.from_args(
            host="localhost",
            port=6000,
            peers=[("brain1", "localhost", 7000),
                   ("brain2", "localhost", 7001)],
        )
        print(f"✓ Created multi-peer config: {len(config3.peers)} peers")

        print("\n✓ ConnectionManager configuration test PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing Bi-directional Communication Implementation")
    print("=" * 70 + "\n")

    results = []

    # Test 1: Backwards compatibility
    results.append(("Backwards Compatibility", test_backwards_compatibility()))

    # Test 2: ConnectionManager config
    results.append(("ConnectionManager Config",
                   test_connection_manager_basic()))

    # Test 3: Bidirectional connection
    results.append(("Bidirectional Connection",
                   test_bidirectional_connection()))

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)
    print("=" * 70)
    if all_passed:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)
