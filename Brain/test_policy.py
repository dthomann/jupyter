#!/usr/bin/env python3
"""
Test what the agent actually learned - inspect policy weights and test against simple strategies.
"""

from brain import BrainAgent
from tic_tac_toe_env import TicTacToeEnv
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_against_random(agent, n_games=100):
    """Test agent against random opponent."""
    env = TicTacToeEnv()
    stats = {'agent_wins': 0, 'opponent_wins': 0, 'draws': 0}

    for _ in range(n_games):
        obs = env.reset()
        agent_is_x = True  # Agent always X

        while not env.done:
            if agent_is_x:
                # Agent's turn - filter valid actions first
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                # Get Q-values and pick best valid action
                _, z, value, x = agent.act(obs, temperature=0.0, greedy=True)
                if agent.use_raw_obs_for_policy:
                    policy_state = x
                else:
                    policy_state = z
                logits = agent.actor_critic.policy_logits(policy_state)
                # Mask invalid actions
                masked_logits = logits.copy()
                for i in range(9):
                    if i not in valid_actions:
                        masked_logits[i] = -1e10
                action = int(np.argmax(masked_logits))
                next_obs, _, done, _ = env.step(action)
            else:
                # Random opponent
                valid_actions = env.get_valid_actions()
                if valid_actions:
                    action = env.rng.choice(valid_actions)
                    next_obs, _, done, _ = env.make_opponent_move(action)
                else:
                    break

            if done:
                if env.winner == 'X':
                    stats['agent_wins'] += 1
                elif env.winner == 'O':
                    stats['opponent_wins'] += 1
                elif env.winner == 'draw':
                    stats['draws'] += 1
                break

            obs = next_obs
            agent_is_x = not agent_is_x

    return stats


def test_against_minimax(agent, n_games=50):
    """Test agent against perfect minimax opponent."""
    # Simple minimax for Tic Tac Toe
    def minimax_move(board):
        # Check for immediate win
        for i in range(9):
            if board[i] == 0:
                test_board = board.copy()
                test_board[i] = -1
                if _check_win(test_board) == 'O':
                    return i

        # Block opponent win
        for i in range(9):
            if board[i] == 0:
                test_board = board.copy()
                test_board[i] = 1
                if _check_win(test_board) == 'X':
                    return i

        # Center if available
        if board[4] == 0:
            return 4

        # Corner
        for i in [0, 2, 6, 8]:
            if board[i] == 0:
                return i

        # Any available
        for i in range(9):
            if board[i] == 0:
                return i
        return 0

    def _check_win(board):
        board_2d = board.reshape(3, 3)
        for i in range(3):
            if abs(board_2d[i].sum()) == 3:
                return 'X' if board_2d[i].sum() > 0 else 'O'
            if abs(board_2d[:, i].sum()) == 3:
                return 'X' if board_2d[:, i].sum() > 0 else 'O'
        diag1 = board_2d[0, 0] + board_2d[1, 1] + board_2d[2, 2]
        diag2 = board_2d[0, 2] + board_2d[1, 1] + board_2d[2, 0]
        if abs(diag1) == 3:
            return 'X' if diag1 > 0 else 'O'
        if abs(diag2) == 3:
            return 'X' if diag2 > 0 else 'O'
        return None

    env = TicTacToeEnv()
    stats = {'agent_wins': 0, 'opponent_wins': 0, 'draws': 0}

    for _ in range(n_games):
        obs = env.reset()
        agent_is_x = True

        while not env.done:
            if agent_is_x:
                # Agent's turn - filter valid actions first
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                # Get Q-values and pick best valid action
                _, z, value, x = agent.act(obs, temperature=0.0, greedy=True)
                if agent.use_raw_obs_for_policy:
                    policy_state = x
                else:
                    policy_state = z
                logits = agent.actor_critic.policy_logits(policy_state)
                # Mask invalid actions
                masked_logits = logits.copy()
                for i in range(9):
                    if i not in valid_actions:
                        masked_logits[i] = -1e10
                action = int(np.argmax(masked_logits))
                next_obs, _, done, _ = env.step(action)
            else:
                action = minimax_move(env.board)
                next_obs, _, done, _ = env.make_opponent_move(action)

            if done:
                if env.winner == 'X':
                    stats['agent_wins'] += 1
                elif env.winner == 'O':
                    stats['opponent_wins'] += 1
                elif env.winner == 'draw':
                    stats['draws'] += 1
                break

            obs = next_obs
            agent_is_x = not agent_is_x

    return stats


def _policy_weight_std(actor):
    weights = []
    for layer in actor.policy_layers:
        weights.append(layer["W"].ravel())
        weights.append(layer["b"].ravel())
    weights.append(actor.W_policy_out.ravel())
    weights.append(actor.b_policy.ravel())
    return np.std(np.concatenate(weights))


def _value_weight_std(actor):
    weights = []
    for layer in actor.value_layers:
        weights.append(layer["W"].ravel())
        weights.append(layer["b"].ravel())
    weights.append(actor.W_value_out.ravel())
    weights.append(np.array([actor.b_value]))
    return np.std(np.concatenate(weights))


def inspect_policy(agent):
    """Inspect what the policy learned."""
    print("\nPolicy Inspection:")
    policy_layer_shapes = [layer["W"].shape for layer in agent.actor_critic.policy_layers]
    value_layer_shapes = [layer["W"].shape for layer in agent.actor_critic.value_layers]
    print(f"  Policy layers: {policy_layer_shapes} -> {agent.actor_critic.W_policy_out.shape}")
    print(f"  Value layers: {value_layer_shapes} -> {agent.actor_critic.W_value_out.shape}")
    print(f"  Policy weight std: {_policy_weight_std(agent.actor_critic):.4f}")
    print(f"  Value weight std: {_value_weight_std(agent.actor_critic):.4f}")
    print(f"  Using raw obs for policy: {agent.use_raw_obs_for_policy}")

    # Test on empty board
    empty_board = np.zeros(9)
    z, x = agent.encode_state(empty_board)
    # Use raw observation if policy uses it, otherwise use latent state
    policy_state = x if agent.use_raw_obs_for_policy else z
    logits = agent.actor_critic.policy_logits(policy_state)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    print(f"\n  Empty board action probabilities:")
    for i, p in enumerate(probs):
        print(f"    Position {i}: {p:.3f}")
    print(f"  Best action (empty board): {np.argmax(probs)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent', type=str, default='tictactoe_agent.pkl', help='Agent file to test')
    args = parser.parse_args()

    print("="*60)
    print("Testing Agent Policy")
    print("="*60)

    # Load agent
    try:
        agent = BrainAgent.load(args.agent)
        print(f"\nLoaded agent with {agent.global_step} training steps")
    except:
        print(f"\nCould not load {args.agent}, creating new agent")
        agent = BrainAgent(
            obs_dim=9,
            latent_dims=[32, 16],
            n_actions=9,
            lr_model=1e-3,
            lr_policy=1e-2,
            replay_batch_size=32,
            use_raw_obs_for_policy=True,  # Use raw board state for stable policy learning
        )
        agent.intrinsic.curiosity_scale = 0.0
        agent.intrinsic.learning_progress_scale = 0.0

    inspect_policy(agent)

    # Test against random
    print("\n" + "="*60)
    print("Testing against Random Opponent (100 games):")
    print("="*60)
    random_stats = test_against_random(agent, n_games=100)
    total = sum(random_stats.values())
    print(
        f"Agent wins: {random_stats['agent_wins']} ({100*random_stats['agent_wins']/total:.1f}%)")
    print(
        f"Opponent wins: {random_stats['opponent_wins']} ({100*random_stats['opponent_wins']/total:.1f}%)")
    print(
        f"Draws: {random_stats['draws']} ({100*random_stats['draws']/total:.1f}%)")

    # Test against minimax
    print("\n" + "="*60)
    print("Testing against Perfect Minimax Opponent (50 games):")
    print("="*60)
    minimax_stats = test_against_minimax(agent, n_games=50)
    total = sum(minimax_stats.values())
    print(
        f"Agent wins: {minimax_stats['agent_wins']} ({100*minimax_stats['agent_wins']/total:.1f}%)")
    print(
        f"Opponent wins: {minimax_stats['opponent_wins']} ({100*minimax_stats['opponent_wins']/total:.1f}%)")
    print(
        f"Draws: {minimax_stats['draws']} ({100*minimax_stats['draws']/total:.1f}%)")

    if minimax_stats['draws'] / total > 0.8:
        print("\n✓ Agent learned optimal play!")
    elif minimax_stats['draws'] / total > 0.5:
        print("\n? Agent learned decent play but not optimal")
    else:
        print("\n✗ Agent did not learn optimal play")


if __name__ == '__main__':
    main()
