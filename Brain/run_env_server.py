#!/usr/bin/env python3
"""
Run the TicTacToe environment server.
This listens for brain client connections and runs tic-tac-toe episodes.
"""

from tic_tac_toe_env import run_env_server
import sys

if __name__ == '__main__':
    print("=" * 70)
    print("TicTacToe Environment Server")
    print("=" * 70)
    print("Listening for brain connections...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        run_env_server(
            host="localhost",
            port=6000,
            authkey=b"brain-secret",
            env_dt=0.05,
        )
    except KeyboardInterrupt:
        print("\n[env] Shutting down...")
        sys.exit(0)

