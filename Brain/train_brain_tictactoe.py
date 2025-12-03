#!/usr/bin/env python3
"""
Train BrainAgent on Tic-Tac-Toe using episode-based learning and legal action masking.
Similar to generic_tictactoe.py but uses the full BrainAgent architecture.
"""

from brain import BrainAgent
from tic_tac_toe_env import TicTacToeEnv
import numpy as np
import sys
from pathlib import Path
import random
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))


EMPTY = 0
X = 1
O = -1
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


def check_winner(board):
    """Check if there's a winner. Returns 1 for X, -1 for O, 0 for draw, None for ongoing."""
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0
    return None


def get_legal_moves(board):
    """Get list of legal move indices."""
    return [i for i in range(9) if board[i] == EMPTY]


def reward_shaping_tictactoe(reward):
    """
    Shape rewards to match generic_tictactoe.py:
    - Win: +1
    - Loss: -1
    - Draw: 0
    - Invalid: -1 (already handled by env)
    """
    if reward >= 10:  # Win
        return 1.0
    elif reward <= -10:  # Loss
        return -1.0
    elif reward == 1:  # Draw
        return 0.0
    else:
        return reward  # Keep other rewards as-is


def play_episode_simple(agent, training=True):
    """
    Play a single episode of tic-tac-toe using simple board (like generic_tictactoe.py).
    Both players use the same policy. No env, no temperature.
    Returns: (outcome, steps)
    """
    import torch
    import torch.distributions as distributions

    board = [0]*9
    player = 1
    states = []
    actions = []
    players = []
    masks = []

    while True:
        # Create inputs exactly like generic_tictactoe.py
        inp = torch.tensor(board, dtype=torch.float32)
        mask = torch.tensor([1.0 if c == 0 else float("-inf") for c in board])

        logits = agent.actor_critic.policy_net(inp)
        logits = logits + mask
        probs = torch.softmax(logits, dim=0)
        dist = distributions.Categorical(probs)
        action = dist.sample()

        if training:
            states.append(np.array(board, dtype=np.float32))
            actions.append(action.item())
            players.append(player)
            masks.append(np.array([0.0 if c == 0 else float(
                "-inf") for c in board], dtype=np.float32))

        board[action.item()] = player
        w = check_winner(board)
        if w is not None:
            outcome = 0.0 if w == 0 else float(w)
            break
        player = -player

    # Update policy
    if training and len(states) > 0:
        # Compute rewards for each player (like generic_tictactoe.py: r = outcome * player)
        rewards = [outcome * p for p in players]
        agent.actor_critic.update_reinforce(
            states, actions, rewards, masks, entropy_coeff=0.001, lr=agent.lr_policy)

    return outcome, len(states)


def play_episode(agent, env, training=True, temperature=1.0, self_play=True):
    """
    Play a single episode of tic-tac-toe using TicTacToeEnv.
    If self_play=True, both players use the same agent.
    Returns: (outcome, steps)
    """
    import torch
    import torch.distributions as distributions

    obs = env.reset()
    if training:
        agent.reset_episode()
    steps = 0
    current_player = 1  # X starts

    states = []
    actions = []
    players = []
    masks = []

    while not env.done:
        # Use policy_net directly (like generic_tictactoe.py)
        board = obs.tolist()
        inp = torch.tensor(board, dtype=torch.float32)
        mask = torch.tensor([1.0 if c == 0 else float("-inf") for c in board])

        with torch.no_grad():
            logits = agent.actor_critic.policy_net(inp)
            logits = logits + mask
            probs = torch.softmax(logits, dim=0)
            dist = distributions.Categorical(probs)
            action = dist.sample().item()

        if training:
            states.append(np.array(board, dtype=np.float32))
            actions.append(action)
            players.append(current_player)
            masks.append(np.array([0.0 if c == 0 else float(
                "-inf") for c in board], dtype=np.float32))

        # Make move
        if current_player == 1:
            obs, _, done, _ = env.step(action)
        else:
            obs, _, done, _ = env.make_opponent_move(action)

        steps += 1

        if done:
            break

        current_player = -current_player

    # Determine outcome
    if env.winner == 'X':
        outcome = 1.0
    elif env.winner == 'O':
        outcome = -1.0
    else:
        outcome = 0.0

    # Update policy (like generic_tictactoe.py)
    if training and len(states) > 0:
        rewards = [outcome * p for p in players]
        agent.actor_critic.update_reinforce(
            states, actions, rewards, masks, entropy_coeff=0.001, lr=agent.lr_policy)

    return outcome, steps


def test_against_random(agent, num_games=1000):
    """
    Test the agent against a random player (using simple board, not env).
    Matches generic_tictactoe.py test methodology.

    Returns:
        dict with statistics: wins, losses, draws, win_rate, loss_rate, draw_rate
    """
    import torch

    agent.actor_critic.eval()

    wins = 0
    losses = 0
    draws = 0

    for game in range(num_games):
        board = [EMPTY] * 9
        current_player = X  # X always goes first

        while True:
            if current_player == X:
                # Agent's turn - use policy_net directly
                inp = torch.tensor(board, dtype=torch.float32)
                with torch.no_grad():
                    logits = agent.actor_critic.policy_net(inp)
                    # CRITICAL FIX: Mask should be 0.0 for legal moves, -inf for illegal
                    mask = torch.tensor(
                        [0.0 if c == 0 else float("-inf") for c in board])
                    logits = logits + mask
                    action = torch.argmax(logits).item()
            else:
                # Random opponent's turn
                legal = get_legal_moves(board)
                action = random.choice(legal)

            board[action] = current_player
            winner = check_winner(board)

            if winner is not None:
                if winner == 0:
                    draws += 1
                elif winner == X:
                    wins += 1
                else:
                    losses += 1
                break

            current_player = -current_player

    agent.actor_critic.train()

    stats = {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games,
        'win_rate': wins / num_games,
        'loss_rate': losses / num_games,
        'draw_rate': draws / num_games
    }

    return stats


def print_test_stats(stats):
    """Print test statistics in a readable format."""
    print("\n" + "="*50)
    print("Test Results Against Random Player")
    print("="*50)
    print(f"Total Games:     {stats['total_games']}")
    print(f"Wins:            {stats['wins']} ({stats['win_rate']*100:.2f}%)")
    print(
        f"Losses:          {stats['losses']} ({stats['loss_rate']*100:.2f}%)")
    print(f"Draws:           {stats['draws']} ({stats['draw_rate']*100:.2f}%)")
    print("="*50 + "\n")


def train_agent(episodes=5000, lr=0.001, entropy_coeff=0.001, self_play=True,
                use_simple_training=True):
    """
    Train BrainAgent on tic-tac-toe using episode-based learning.

    Args:
        episodes: Number of training episodes
        lr: Learning rate for policy
        entropy_coeff: Entropy bonus coefficient
        self_play: If True, both players learn from same agent (like generic_tictactoe.py)
        use_simple_training: If True, use simplified training that matches generic_tictactoe.py
    """
    env = TicTacToeEnv()

    # Create agent with episode-based learning enabled
    # Use exact architecture from generic_tictactoe.py: 9->32->9 with ReLU
    from brain.actor_critic import ActorCritic
    actor_critic = ActorCritic(
        state_dim=9,
        n_actions=9,
        # Match generic_tictactoe.py exactly: 9->32->9
        policy_hidden_dims=(32,),
        value_hidden_dims=(32,),  # Value network also 9->32->1
        activation="relu",  # Match generic_tictactoe.py
        entropy_coeff=entropy_coeff,
        scale=0.1,  # Default weight initialization scale
    )

    agent = BrainAgent(
        obs_dim=9,  # 3x3 board
        latent_dims=[32, 16],  # World model can keep original architecture
        n_actions=9,
        lr_model=1e-3,
        lr_policy=lr,
        replay_batch_size=32,
        use_raw_obs_for_policy=True,  # Use raw board state
        episode_based_learning=True,  # Enable episode-based learning
        entropy_coeff=entropy_coeff,
        reward_shaping=None,  # Don't shape rewards in self-play mode
    )

    # Replace actor_critic with the correctly configured one
    agent.actor_critic = actor_critic

    # Disable intrinsic motivation for tic-tac-toe (it adds noise)
    agent.intrinsic.curiosity_scale = 0.0
    agent.intrinsic.learning_progress_scale = 0.0

    print("Training BrainAgent on Tic-Tac-Toe...")
    print(f"Episodes: {episodes}")
    print(f"Learning rate: {lr}")
    print(f"Entropy coefficient: {entropy_coeff}")
    print(f"Episode-based learning: {agent.episode_based_learning}")
    print(f"Self-play: {self_play}")
    print()

    results = []

    for episode in range(episodes):
        # Use simple or full training based on config
        if use_simple_training:
            outcome, steps = play_episode_simple(agent, training=True)
        else:
            outcome, steps = play_episode(
                agent, env, training=True, temperature=1.0, self_play=self_play)
        results.append(outcome)

        # Report progress
        if (episode + 1) % 500 == 0:
            recent = results[-500:] if len(results) >= 500 else results
            wins = sum(1 for r in recent if r > 0)
            losses = sum(1 for r in recent if r < 0)
            draws = sum(1 for r in recent if r == 0)

            # Quick test
            test_stats = test_against_random(agent, num_games=200)

            print(f"Episode {episode + 1}/{episodes} | "
                  f"Recent: W={wins} L={losses} D={draws} | "
                  f"Test Loss: {test_stats['losses']} ({100*test_stats['loss_rate']:.1f}%)")

    return agent


def main():
    print("=" * 70)
    print("BrainAgent Tic-Tac-Toe Training")
    print("=" * 70)

    print("\n1. Training agent with episode-based learning (5000 episodes)...")
    agent = train_agent(episodes=5000, lr=0.001,
                        entropy_coeff=0.001, self_play=True, use_simple_training=True)

    print(f"\n2. Final test vs random (1000 games)...")
    stats = test_against_random(agent, num_games=1000)
    print_test_stats(stats)

    loss_rate = stats['loss_rate']
    print(f"\n3. Results:")
    print("=" * 70)
    if loss_rate == 0:
        print("✓ PERFECT: Agent achieves 0% loss rate!")
        return True
    elif loss_rate < 0.002:
        print(
            f"✓ NEAR-PERFECT: {stats['losses']} losses in 1000 games ({100*loss_rate:.2f}%)")
        return True
    elif loss_rate < 0.01:
        print("✓ EXCELLENT: < 1% loss rate")
        return True
    elif loss_rate < 0.05:
        print("✓ GOOD: < 5% loss rate")
        return True
    else:
        print(f"✗ NEEDS IMPROVEMENT: {100*loss_rate:.1f}% loss rate")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
