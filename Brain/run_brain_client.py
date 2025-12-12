#!/usr/bin/env python3
"""
Run the Brain client - a biologically-inspired learning agent.

The brain learns from the environment through:
- Neuromodulators (Dopamine, Norepinephrine, Serotonin, Acetylcholine, Cortisol)
- Hippocampal memory (Short-term → Episodic consolidation)
- Intrinsic motivation (Curiosity, Competence, Autonomy)
- Homeostatic drives (Energy, Boredom)

Learning rates and exploration are dynamically regulated by internal chemical states,
not fixed hyperparameters. The brain adapts to the environment automatically.

Usage:
    # Start environment server first:
    python run_env_server.py
    
    # Then start brain client (simplest):
    python run_brain_client.py
    
    # Self-play mode (default):
    python run_brain_client.py --load brain.pkl --save brain.pkl
    
    # Play against opponent (two-brain mode):
    python run_brain_client.py --play-against-opponent
    
    # With options:
    python run_brain_client.py --load brain.pkl --save brain.pkl --max-episodes 10000
"""

from brain import BrainAgent
from brain.actor_critic import ActorCritic
from brain.connection_config import BrainConnectionConfig
import sys
import argparse
import random
import os
from pathlib import Path

import numpy as np


def ensure_brainstates_path(filepath: str) -> str:
    """
    Ensure brain state files are saved/loaded from Brainstates/ folder by default.
    If an absolute path is provided, use it as-is.
    If a relative path is provided, prepend Brainstates/.
    Creates Brainstates/ directory if it doesn't exist.
    """
    path = Path(filepath)

    # If absolute path, use as-is
    if path.is_absolute():
        return str(path)

    # For relative paths, prepend Brainstates/
    brainstates_dir = Path("Brainstates")
    brainstates_dir.mkdir(exist_ok=True)

    return str(brainstates_dir / path)


try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int):
    """Seed python, numpy, and torch RNGs for reproducibility."""
    if seed is None or seed < 0:
        return

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description='Run biologically-inspired Brain client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses Brainstates/brain_default.pkl, can resume on restart)
  python run_brain_client.py
  
  # Play against another brain (two-brain mode with named brains)
  python run_brain_client.py --name player1 --play-against-opponent
  python run_brain_client.py --name player2 --play-against-opponent
  
  # Load existing brain and continue training (from Brainstates/ folder)
  python run_brain_client.py --load brain.pkl
  
  # Train for more episodes
  python run_brain_client.py --max-episodes 10000
  
  # Custom environment server
  python run_brain_client.py --host 192.168.1.100 --port 7000
  
  # Enable bidirectional communication (brain accepts incoming connections)
  python run_brain_client.py --enable-listener --listen-host localhost --listen-port 7000
  
  # Connect to additional peers (besides environment)
  python run_brain_client.py --peer other_brain localhost 7001

Note: The brain self-regulates learning through neuromodulators.
No manual hyperparameter tuning needed - it adapts automatically!

Brain state files are saved/loaded from Brainstates/ folder by default.
You can use absolute paths to save elsewhere if needed.

When using --play-against-opponent, make sure the environment server
is running in two-brain mode (default) and start two brain clients.
        """)

    # Essential connection parameters
    parser.add_argument('--host', type=str, default='localhost',
                        help='Environment server host (default: localhost)')
    parser.add_argument('--port', type=int, default=6000,
                        help='Environment server port (default: 6000)')
    parser.add_argument('--authkey', type=str, default='brain-secret',
                        help='Connection authkey (default: brain-secret)')

    # Bi-directional communication (new features)
    parser.add_argument('--brain-id', type=str, default=None,
                        help='Unique brain identifier (default: auto-generated)')
    parser.add_argument('--listen-host', type=str, default=None,
                        help='Host to listen on for incoming connections (default: None, listener disabled)')
    parser.add_argument('--listen-port', type=int, default=None,
                        help='Port to listen on for incoming connections (requires --listen-host)')
    parser.add_argument('--enable-listener', action='store_true',
                        help='Enable incoming connection listener (requires --listen-host and --listen-port)')
    parser.add_argument('--peer', action='append', nargs=3, metavar=('PEER_ID', 'HOST', 'PORT'),
                        help='[DEPRECATED] Peers are now discovered via multicast. This argument is ignored.')

    # Persistence
    parser.add_argument('--load', type=str, default=None,
                        help='Load brain from file (continues training from saved state). Defaults to Brainstates/ folder if relative path.')
    parser.add_argument('--save', type=str, default=None,
                        help='Save brain to file periodically (e.g., brain.pkl). Defaults to Brainstates/ folder if relative path.')
    parser.add_argument('--name', type=str, default=None,
                        help='Brain name for multiple simultaneous brains (creates Brainstates/brain_{name}.pkl). If not specified, uses Brainstates/brain_default.pkl')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save every N episodes (default: 500)')

    # Training control
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum episodes to train (default: unlimited, Ctrl+C to stop)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42, use -1 for random)')

    # Optional: Biological parameters (rarely need adjustment)
    parser.add_argument('--curiosity', type=float, default=None,
                        help='Curiosity scale (default: auto, based on environment)')
    parser.add_argument('--competence', type=float, default=None,
                        help='Competence scale (default: auto)')

    # Game mode
    parser.add_argument('--play-against-opponent', action='store_true',
                        help='Play against another brain (disable self-play random moves). Use with two-brain server mode.')

    # Optional: Monitoring/logging
    parser.add_argument('--stats-every', type=int, default=100,
                        help='Print stats every N episodes (default: 100)')
    parser.add_argument('--metrics-log', type=str, default='logs/brain_metrics.jsonl',
                        help='Log metrics to JSONL file (default: logs/brain_metrics.jsonl, use "none" to disable)')

    # Hidden/advanced options (not shown in help by default)
    parser.add_argument('--dt', type=float, default=0.02,
                        help=argparse.SUPPRESS)  # Internal tick rate
    parser.add_argument('--stats-window', type=int, default=200,
                        help=argparse.SUPPRESS)
    parser.add_argument('--eval-games', type=int, default=200,
                        help=argparse.SUPPRESS)
    parser.add_argument('--metrics-window', type=int, default=200,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Determine brain file to use
    # Priority:
    # 1. If --load is provided, use that file (or --save if explicitly different)
    # 2. If --save is provided, use that file
    # 3. If --name is provided, use brain_{name}.pkl (for multiple simultaneous brains)
    # 4. Otherwise, use brain_default.pkl (fixed default, can resume on restart)
    if args.load:
        # If --load is provided, use it for loading
        # For saving: use --save if explicitly provided, otherwise use --load file
        brain_file = args.save if args.save else args.load
    elif args.save:
        # Only --save provided, use it for both
        brain_file = args.save
    elif args.name:
        # Multiple brains: use name-based file
        brain_file = f"brain_{args.name}.pkl"
        print(f"[brain] Using named brain file: {brain_file}")
    else:
        # Single brain: use fixed default file (can resume on restart)
        brain_file = "brain_default.pkl"
        print(f"[brain] No brain file specified, using default: {brain_file}")

    # Note: brain_file will be updated to include Brainstates/ prefix below

    # Ensure brain files are in Brainstates/ folder by default
    brain_file = ensure_brainstates_path(brain_file)
    print(f"[brain] Brain state file: {brain_file}")

    # Update args for consistency (for loading logic below)
    if not args.load:
        args.load = brain_file
    else:
        # Ensure --load path is also in Brainstates/ if relative
        args.load = ensure_brainstates_path(args.load)

    seed_value = None if args.seed is None or args.seed < 0 else args.seed
    if seed_value is not None:
        set_global_seed(seed_value)

    stats_window = max(1, args.stats_window)
    metrics_window = max(1, args.metrics_window)
    eval_games = max(1, args.eval_games)

    max_episodes = args.max_episodes  # None = unlimited

    metrics_log_path = None
    if args.metrics_log and args.metrics_log.lower() != "none":
        metrics_log_path = str(Path(args.metrics_log).expanduser())

    print("=" * 70)
    print("🧠 Biologically-Inspired Brain Client")
    print("=" * 70)
    print("Learning through:")
    print("  • Neuromodulators (DA, NE, 5-HT, ACh, Cortisol)")
    print("  • Hippocampal memory (STM → Episodic consolidation)")
    print("  • Intrinsic motivation (Curiosity, Competence, Autonomy)")
    print("  • Homeostatic drives (Energy, Boredom)")
    print("=" * 70)

    # Create or load brain
    if args.load and os.path.exists(args.load):
        print(f"Loading brain from {args.load}...")
        agent = BrainAgent.load(args.load)
        print("Brain loaded successfully")
        agent.stats_window = stats_window
    else:
        # Create new brain (either file doesn't exist or no load path specified)
        if args.load:
            print(f"Brain file {args.load} not found, will create new brain")
        else:
            print("Creating new brain...")
        # Create agent with episode-based learning enabled
        agent_rng = np.random.RandomState(
            seed_value) if seed_value is not None else None
        actor_seed = seed_value + 1 if seed_value is not None else None
        actor_rng = np.random.RandomState(
            actor_seed) if actor_seed is not None else None

        # Mode variable for behavioral diversity
        mode_dim = 8  # Dimension of mode variable

        # Larger network for "Brain" - size chosen to handle complex environments
        # but will only use what's needed for simple tasks
        # State dimension includes mode variable
        actor_critic = ActorCritic(
            state_dim=9 + mode_dim,  # 9 (board) + mode_dim
            n_actions=9,
            policy_hidden_dims=(128, 64),
            value_hidden_dims=(128, 64),
            activation="relu",
            entropy_coeff=0.001,  # Base exploration, neuromodulators modulate this
            rng=actor_rng,
        )

        agent = BrainAgent(
            obs_dim=9,  # 3x3 board
            # Larger world model for general learning
            latent_dims=[128, 64, 32],
            n_actions=9,
            lr_model=1e-3,  # Base world model learning rate
            # Base policy learning rate (neuromodulators modulate this)
            lr_policy=0.001,
            replay_batch_size=32,
            use_raw_obs_for_policy=True,
            episode_based_learning=True,
            entropy_coeff=0.001,  # Base exploration
            reward_shaping=None,
            mode_dim=mode_dim,
            z_mode_sigma_base=1.0,
            k_z=0.5,
            rng=agent_rng,
        )

        # Replace actor_critic with the correctly configured one
        agent.actor_critic = actor_critic

        # Configure intrinsic motivation (can be overridden via args)
        if args.curiosity is not None:
            agent.intrinsic.curiosity_scale = args.curiosity
        else:
            # Auto-scale based on environment complexity
            # For simple games like tic-tac-toe, use lower curiosity
            agent.intrinsic.curiosity_scale = 0.05

        if args.competence is not None:
            agent.intrinsic.competence_scale = args.competence
        else:
            agent.intrinsic.competence_scale = 0.05

        agent.intrinsic.autonomy_scale = 0.01

        print("✓ Brain created with biologically-inspired components")
        print(f"  Curiosity scale: {agent.intrinsic.curiosity_scale:.3f}")
        print(f"  Competence scale: {agent.intrinsic.competence_scale:.3f}")
        print(f"  Autonomy scale: {agent.intrinsic.autonomy_scale:.3f}")

        agent.stats_window = stats_window

    print(f"\nConnecting to environment at {args.host}:{args.port}...")
    if args.play_against_opponent:
        print("Mode: Playing against opponent (self-play random moves disabled)")
    else:
        print("Mode: Self-play (with random opponent decay)")
    if max_episodes:
        print(f"Training for {max_episodes} episodes")
    else:
        print("Training indefinitely (press Ctrl+C to stop)")
    if seed_value is not None:
        print(f"Random seed: {seed_value}")
    if metrics_log_path:
        print(f"Metrics logging to: {metrics_log_path}")
    print(f"Stats printed every {args.stats_every} episodes")
    print(f"Brain will be saved to: {brain_file}")
    print("\n" + "=" * 70)
    print("The brain will self-regulate learning through neuromodulators.")
    print("No manual hyperparameter tuning needed - it adapts automatically!")
    print("=" * 70 + "\n")

    # Track episodes for saving
    initial_episode = agent.episode_index

    # Build connection config from args
    # Note: peers are now discovered via multicast, so --peer argument is ignored
    if args.peer:
        print(
            "[brain] Warning: --peer argument is ignored (peers are discovered via multicast)")

    # Validate listener args
    if args.enable_listener and args.listen_host is None:
        parser.error("--enable-listener requires --listen-host")
    if args.listen_port is not None and args.listen_host is None:
        parser.error("--listen-port requires --listen-host")

    connection_config = BrainConnectionConfig.from_args(
        host=args.host,  # Ignored - kept for backwards compatibility
        port=args.port,  # Ignored - kept for backwards compatibility
        authkey=args.authkey.encode() if isinstance(
            args.authkey, str) else args.authkey,
        brain_id=args.brain_id,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        enable_listener=args.enable_listener,
    )

    try:
        # Run brain client with biologically-inspired defaults
        # Learning rates and exploration are dynamically regulated by neuromodulators
        # No need for manual decay schedules - the brain adapts automatically

        agent.run_brain_client(
            host=args.host,
            port=args.port,
            authkey=args.authkey.encode() if isinstance(
                args.authkey, str) else args.authkey,
            dt=args.dt,
            save_path=brain_file,
            save_every=args.save_every,
            max_episodes=max_episodes,
            stats_every=args.stats_every,
            # Biological defaults: neuromodulators handle learning rate modulation
            entropy_start=0.001,  # Initial exploration
            entropy_end=0.0,  # Gradually reduce exploration as competence increases
            entropy_decay_episodes=max_episodes if max_episodes else 5000,
            entropy_decay_type='linear',
            # Base learning rate (neuromodulators will modulate this dynamically)
            lr_start=0.001,
            lr_end=0.0001,  # Slight decay, but neuromodulators do most of the work
            lr_decay_episodes=max_episodes if max_episodes else 5000,
            lr_decay_type='linear',
            # Self-play training (both players learn)
            # If playing against opponent, random moves are disabled automatically
            random_opponent_prob_start=1.0,  # Start with random opponent
            random_opponent_prob_end=0.0,  # Transition to self-play
            random_opponent_decay_episodes=max_episodes if max_episodes else 5000,
            random_opponent_decay_type='linear',
            play_against_opponent=args.play_against_opponent,
            metrics_log_path=metrics_log_path,
            metrics_window=metrics_window,
            eval_games=eval_games,
            random_loss_patience=None,  # No early stopping by default
            random_loss_min_delta=0.0,
            connection_config=connection_config,
        )
    except KeyboardInterrupt:
        print("\n[brain] Shutting down...")
    except Exception as e:
        import traceback
        print(f"\n[brain] Unexpected error: {e}")
        traceback.print_exc()
        raise
    finally:
        # Always save on exit (auto-save)
        try:
            print(f"\n[brain] Auto-saving brain to {brain_file}...")
            agent.save(brain_file)
            # Use environment's starting episode if available, otherwise use initial_episode
            if hasattr(agent, '_environment_starting_episode') and agent._environment_starting_episode is not None:
                start_episode = agent._environment_starting_episode
            else:
                start_episode = initial_episode
            episodes_trained = agent.episode_index - start_episode
            # Debug: show what values we're using
            print(
                f"[brain] Brain saved successfully (trained for {episodes_trained} episodes) "
                f"(episode_index={agent.episode_index}, start_episode={start_episode})")
        except Exception as e:
            print(f"[brain] Warning: Failed to save brain: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
