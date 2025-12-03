#!/usr/bin/env python3
"""
Run the Brain client.
This connects to the environment server and learns to play tic-tac-toe.
"""

from brain import BrainAgent
from brain.actor_critic import ActorCritic
import sys
import argparse
import random
from pathlib import Path

import numpy as np

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
        description='Run Brain client for tic-tac-toe')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Environment server host')
    parser.add_argument('--port', type=int, default=6000,
                        help='Environment server port')
    parser.add_argument('--authkey', type=str,
                        default='brain-secret', help='Connection authkey')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Tick interval (seconds)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lr-end', type=float, default=None,
                        help='Final learning rate (after decay). If None, no decay.')
    parser.add_argument('--lr-decay-episodes', type=int, default=None,
                        help='Number of episodes to decay learning rate over (default: same as max-episodes)')
    parser.add_argument('--lr-decay-type', type=str, default='linear',
                        choices=['linear', 'exponential', 'cosine'],
                        help='Learning rate decay schedule: linear, exponential, or cosine')
    parser.add_argument('--entropy', type=float, default=0.001,
                        help='Initial entropy coefficient')
    parser.add_argument('--entropy-end', type=float, default=0.0,
                        help='Final entropy coefficient (after decay)')
    parser.add_argument('--entropy-decay-episodes', type=int, default=None,
                        help='Number of episodes to decay entropy over (default: same as max-episodes)')
    parser.add_argument('--entropy-decay-type', type=str, default='linear',
                        choices=['linear', 'exponential', 'cosine'],
                        help='Entropy decay schedule: linear, exponential, or cosine')
    parser.add_argument('--load', type=str, default=None,
                        help='Load brain from file')
    parser.add_argument('--save', type=str, default=None,
                        help='Save brain to file periodically')
    parser.add_argument('--save-every', type=int,
                        default=500, help='Save every N episodes')
    parser.add_argument('--max-episodes', type=int, default=5000,
                        help='Maximum number of episodes to train (capped at 5000)')
    parser.add_argument('--stats-every', type=int, default=100,
                        help='Print performance statistics every N episodes')
    parser.add_argument('--stats-window', type=int, default=200,
                        help='Window size for self-play statistics')
    parser.add_argument('--eval-games', type=int, default=200,
                        help='Games per random-opponent evaluation')
    parser.add_argument('--metrics-window', type=int, default=200,
                        help='Episodes to average when printing loss metrics')
    parser.add_argument('--metrics-log', type=str, default='logs/brain_metrics.jsonl',
                        help='JSONL file for per-episode metrics (use "none" to disable)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for numpy/torch/random (set -1 to disable)')
    parser.add_argument('--random-opponent-prob-start', type=float, default=1.0,
                        help='Initial probability of random opponent (O player) per game (default: 1.0 = 100%%)')
    parser.add_argument('--random-opponent-prob-end', type=float, default=0.0,
                        help='Final probability of random opponent per game (default: 0.0 = 0%%)')
    parser.add_argument('--random-opponent-decay-episodes', type=int, default=None,
                        help='Number of episodes to decay random opponent probability over (default: same as max-episodes)')
    parser.add_argument('--random-opponent-decay-type', type=str, default='linear',
                        choices=['linear', 'exponential', 'cosine'],
                        help='Random opponent probability decay schedule: linear, exponential, or cosine')
    parser.add_argument('--train-against-random-only', action='store_true',
                        help='Train exclusively against random opponents (100%% probability, no decay). Overrides random-opponent-prob-* settings.')
    parser.add_argument('--random-loss-patience', type=int, default=None,
                        help='Number of random-eval checks without improvement before early stop (disabled if None)')
    parser.add_argument('--random-loss-min-delta', type=float, default=0.0,
                        help='Required decrease in random loss to reset patience')

    args = parser.parse_args()

    seed_value = None if args.seed is None or args.seed < 0 else args.seed
    if seed_value is not None:
        set_global_seed(seed_value)

    stats_window = max(1, args.stats_window)
    metrics_window = max(1, args.metrics_window)
    eval_games = max(1, args.eval_games)

    if args.max_episodes is None or args.max_episodes <= 0:
        max_episodes = 5000
        print(
            "[brain] max-episodes not specified; defaulting to 5000 to keep runs bounded.")
    else:
        max_episodes = args.max_episodes

    metrics_log_path = None
    if args.metrics_log and args.metrics_log.lower() != "none":
        metrics_log_path = str(Path(args.metrics_log).expanduser())

    # Handle train-against-random-only flag
    random_opponent_prob_start = args.random_opponent_prob_start
    random_opponent_prob_end = args.random_opponent_prob_end
    random_opponent_decay_episodes = args.random_opponent_decay_episodes
    random_opponent_decay_type = args.random_opponent_decay_type

    if args.train_against_random_only:
        random_opponent_prob_start = 1.0
        random_opponent_prob_end = 1.0  # No decay - always 100% random
        # Set to max_episodes to effectively disable decay
        random_opponent_decay_episodes = max_episodes
        random_opponent_decay_type = 'linear'
        print(
            "[brain] Training mode: EXCLUSIVELY against random opponents (100%%, no decay)")
        # Warn if entropy decays too quickly (reduces exploration needed to exploit random)
        if args.entropy_end < 0.001:
            print(
                "[brain] WARNING: Low entropy-end ({:.4f}) may reduce exploration. "
                "Consider --entropy-end 0.001 or higher for better learning against random.".format(
                    args.entropy_end))

    print("=" * 70)
    print("Brain Client - TicTacToe Learning")
    print("=" * 70)

    # Create or load brain
    if args.load:
        print(f"Loading brain from {args.load}...")
        agent = BrainAgent.load(args.load)
        print("Brain loaded successfully")
    else:
        print("Creating new brain...")
        # Create agent with episode-based learning enabled
        agent_rng = np.random.RandomState(
            seed_value) if seed_value is not None else None
        actor_seed = seed_value + 1 if seed_value is not None else None
        actor_rng = np.random.RandomState(
            actor_seed) if actor_seed is not None else None
        actor_critic = ActorCritic(
            state_dim=9,
            n_actions=9,
            policy_hidden_dims=(32,),
            value_hidden_dims=(32,),
            activation="relu",
            entropy_coeff=args.entropy,
            rng=actor_rng,
        )

        agent = BrainAgent(
            obs_dim=9,  # 3x3 board
            latent_dims=[32, 16],
            n_actions=9,
            lr_model=1e-3,
            lr_policy=args.lr,
            replay_batch_size=32,
            use_raw_obs_for_policy=True,
            episode_based_learning=True,
            entropy_coeff=args.entropy,
            reward_shaping=None,
            rng=agent_rng,
        )

        # Replace actor_critic with the correctly configured one
        agent.actor_critic = actor_critic

        # Disable intrinsic motivation for tic-tac-toe
        agent.intrinsic.curiosity_scale = 0.0
        agent.intrinsic.learning_progress_scale = 0.0
        print("Brain created successfully")

    agent.stats_window = stats_window

    print(f"Connecting to environment at {args.host}:{args.port}...")
    print(f"Training for {max_episodes} episodes (cap enforced at 5000)")
    if seed_value is not None:
        print(f"Using deterministic seed: {seed_value}")
    if metrics_log_path:
        print(f"Per-episode metrics will be appended to {metrics_log_path}")
    print(f"Performance stats every {args.stats_every} episodes")
    print()

    # Track episodes for saving
    initial_episode = agent.episode_index

    try:
        # Run brain client
        agent.run_brain_client(
            host=args.host,
            port=args.port,
            authkey=args.authkey.encode() if isinstance(
                args.authkey, str) else args.authkey,
            dt=args.dt,
            save_path=args.save,
            save_every=args.save_every if args.save else None,
            max_episodes=max_episodes,
            stats_every=args.stats_every,
            entropy_start=args.entropy,
            entropy_end=args.entropy_end,
            entropy_decay_episodes=args.entropy_decay_episodes,
            entropy_decay_type=args.entropy_decay_type,
            lr_start=args.lr,
            lr_end=args.lr_end,
            lr_decay_episodes=args.lr_decay_episodes,
            lr_decay_type=args.lr_decay_type,
            random_opponent_prob_start=random_opponent_prob_start,
            random_opponent_prob_end=random_opponent_prob_end,
            random_opponent_decay_episodes=random_opponent_decay_episodes,
            random_opponent_decay_type=random_opponent_decay_type,
            metrics_log_path=metrics_log_path,
            metrics_window=metrics_window,
            eval_games=eval_games,
            random_loss_patience=args.random_loss_patience,
            random_loss_min_delta=args.random_loss_min_delta,
        )
    except KeyboardInterrupt:
        print("\n[brain] Shutting down...")
    except Exception as e:
        import traceback
        print(f"\n[brain] Unexpected error: {e}")
        traceback.print_exc()
        raise
    finally:
        # Save on exit if requested
        if args.save:
            print(f"\nSaving brain to {args.save}...")
            agent.save(args.save)
            print(
                f"Brain saved (trained for {agent.episode_index - initial_episode} episodes)")


if __name__ == '__main__':
    main()
