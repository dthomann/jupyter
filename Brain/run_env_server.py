#!/usr/bin/env python3
"""
Run the TicTacToe environment server.
This listens for brain client connections and runs tic-tac-toe episodes.

Modes:
  - Two-brain mode (default): Waits for two brain connections and runs matches
    where brains play against each other.
  - Self-play mode: Single brain connection, brain plays against itself.
"""

from tic_tac_toe_env import run_env_server
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run TicTacToe environment server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-brain mode (default): brains play against each other
  python run_env_server.py
  
  # Self-play mode: single brain plays against itself
  python run_env_server.py --self-play
  
  # Custom port
  python run_env_server.py --port 7000
        """)

    parser.add_argument('--host', type=str, default='localhost',
                        help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=6000,
                        help='Server port (default: 6000)')
    parser.add_argument('--authkey', type=str, default='brain-secret',
                        help='Connection authkey (default: brain-secret)')
    parser.add_argument('--self-play', action='store_true',
                        help='Enable self-play mode (single brain plays against itself)')
    parser.add_argument('--training-mode', action='store_true', default=True,
                        help='Enable training mode: forces immediate wins/blocks (default: True)')
    parser.add_argument('--no-training-mode', dest='training_mode', action='store_false',
                        help='Disable training mode: allow all valid moves')

    args = parser.parse_args()

    print("=" * 70)
    print("TicTacToe Environment Server")
    print("=" * 70)
    if args.self_play:
        print("Mode: Self-play (single brain)")
        print("Listening for one brain connection...")
    else:
        print("Mode: Two-brain matches")
        print("Listening for two brain connections...")
    print(f"Training mode: {'ENABLED' if args.training_mode else 'DISABLED'}")
    if args.training_mode:
        print("  - Forces immediate wins when available")
        print("  - Forces blocks when opponent threatens win")
    print("Press Ctrl+C to stop")
    print()

    try:
        run_env_server(
            host=args.host,
            port=args.port,
            authkey=args.authkey.encode() if isinstance(
                args.authkey, str) else args.authkey,
            env_dt=0.05,
            require_two_brains=not args.self_play,
            training_mode=args.training_mode,
        )
    except KeyboardInterrupt:
        print("\n[env] Shutting down...")
        sys.exit(0)
