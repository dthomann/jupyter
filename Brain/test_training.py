#!/usr/bin/env python3
"""
Tic Tac Toe training - Monte Carlo learning with canonical states (like tictactoe.py).
Uses episode-based updates for faster convergence.
"""

from tic_tac_toe_env import TicTacToeEnv
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

# Symmetry transformations from tictactoe.py
IDX = list(range(9))
ROT90 = [6, 3, 0, 7, 4, 1, 8, 5, 2]
ROT180 = [8, 7, 6, 5, 4, 3, 2, 1, 0]
ROT270 = [2, 5, 8, 1, 4, 7, 0, 3, 6]
REF_H = [2, 1, 0, 5, 4, 3, 8, 7, 6]
REF_V = [6, 7, 8, 3, 4, 5, 0, 1, 2]
REF_MAIN = [0, 3, 6, 1, 4, 7, 2, 5, 8]
REF_ANTI = [8, 5, 2, 7, 4, 1, 6, 3, 0]

SYMMETRIES = [IDX, ROT90, ROT180, ROT270, REF_H, REF_V, REF_MAIN, REF_ANTI]

EMPTY = "."
MARKS = ["X", "O"]


def board_numpy_to_string(board_np):
    """Convert numpy board (-1, 0, 1) to string representation (O, ., X)."""
    result = []
    for val in board_np:
        if val == 1:
            result.append("X")
        elif val == -1:
            result.append("O")
        else:
            result.append(".")
    return "".join(result)


def flip_marks(board_str):
    """Swap X and O, keep EMPTY."""
    return board_str.replace("X", "T").replace("O", "X").replace("T", "O")


def canonical(board_str, mark_to_move):
    """
    Return (canonical_board, transform).
    canonical_board is the lexicographically smallest symmetric variant
    after possibly flipping marks so that the player to move is always 'X'.
    transform[i] gives the index in the original board that corresponds
    to canonical_board[i].
    """
    if mark_to_move == "X":
        base = board_str
    else:
        base = flip_marks(board_str)

    best = None
    best_map = None

    for M in SYMMETRIES:
        cand = "".join(base[M[i]] for i in range(9))
        if best is None or cand < best:
            best = cand
            best_map = M

    return best, best_map


def get_valid_actions(board_str):
    """Get valid action indices from board string."""
    return [i for i, c in enumerate(board_str) if c == EMPTY]


def smart_opponent_move(board_np, player_piece):
    """Opponent that makes smart moves: win > block > center > corner > any."""
    valid = [i for i in range(9) if board_np[i] == 0]
    if not valid:
        return None

    opponent_piece = -player_piece

    # 1. Win if possible
    for move in valid:
        test = board_np.copy()
        test[move] = player_piece
        if check_win_numpy(test, player_piece):
            return move

    # 2. Block opponent win
    for move in valid:
        test = board_np.copy()
        test[move] = opponent_piece
        if check_win_numpy(test, opponent_piece):
            return move

    # 3. Take center
    if 4 in valid:
        return 4

    # 4. Take corner
    corners = [0, 2, 6, 8]
    available_corners = [c for c in corners if c in valid]
    if available_corners:
        return np.random.choice(available_corners)

    # 5. Any valid move
    return np.random.choice(valid)


def check_win_numpy(board_np, piece):
    """Check if piece wins on numpy board."""
    b = board_np.reshape(3, 3)
    # Rows, cols
    for i in range(3):
        if np.all(b[i, :] == piece) or np.all(b[:, i] == piece):
            return True
    # Diagonals
    if b[0, 0] == piece and b[1, 1] == piece and b[2, 2] == piece:
        return True
    if b[0, 2] == piece and b[1, 1] == piece and b[2, 0] == piece:
        return True
    return False


class MonteCarloQAgent:
    """Monte Carlo Q-learning agent using canonical states and visit-count learning rate."""

    def __init__(self, epsilon=0.3):
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(9))
        self.N = defaultdict(lambda: np.zeros(9, dtype=np.int32))

    def _ensure_state(self, state):
        """Ensure state exists in Q and N tables."""
        if state not in self.Q:
            self.Q[state] = np.zeros(9)
            self.N[state] = np.zeros(9, dtype=np.int32)

    def select_action(self, board_np, agent_is_x, training=True):
        """Select action using canonical state. Returns (real_action_index, canonical_state, canonical_action_index)."""
        board_str = board_numpy_to_string(board_np)
        mark = "X" if agent_is_x else "O"
        canon, mapping = canonical(board_str, mark)
        self._ensure_state(canon)

        # Get valid actions in canonical board
        avail_can = get_valid_actions(canon)

        if training and np.random.random() < self.epsilon:
            a_can = np.random.choice(avail_can)
        else:
            qvals = self.Q[canon]
            best_q = -np.inf
            best_actions = []
            for a in avail_can:
                if qvals[a] > best_q:
                    best_q = qvals[a]
                    best_actions = [a]
                elif qvals[a] == best_q:
                    best_actions.append(a)
            a_can = np.random.choice(best_actions)

        # Map canonical action back to real board
        a_real = mapping[a_can]
        return a_real, canon, a_can

    def update_from_episode(self, episode, winner_id):
        """
        Monte Carlo update: assign final reward to all moves in episode.
        episode is list of (canonical_state, canonical_action_index, player_id).
        winner_id: 0 if X won, 1 if O won, None if draw (like tictactoe.py).
        """
        for canon_state, action_canon, player_id in episode:
            self._ensure_state(canon_state)

            # Assign reward based on outcome (like tictactoe.py)
            if winner_id is None:
                G = 0.0  # Draw
            elif winner_id == player_id:
                G = 1.0  # This player won
            else:
                G = -1.0  # This player lost

            # Visit-count based learning rate (like tictactoe.py)
            self.N[canon_state][action_canon] += 1
            n = self.N[canon_state][action_canon]
            alpha = 1.0 / n  # Decaying learning rate

            # Update Q-value
            old_q = self.Q[canon_state][action_canon]
            self.Q[canon_state][action_canon] = old_q + alpha * (G - old_q)


def test_agent(agent, n_games=500):
    """Test agent against random opponent."""
    env = TicTacToeEnv()
    stats = {'wins': 0, 'losses': 0, 'draws': 0}

    for _ in range(n_games):
        obs = env.reset()
        agent_is_x = np.random.choice([True, False])

        if not agent_is_x:
            valid = env.get_valid_actions()
            if valid:
                opp_action = np.random.choice(valid)
                obs, _, done, _ = env.step(opp_action)
                if done:
                    stats['losses' if env.winner == 'X' else 'draws'] += 1
                    continue

        while not env.done:
            action, _, _ = agent.select_action(obs, agent_is_x, training=False)

            if agent_is_x:
                obs, _, done, _ = env.step(action)
            else:
                obs, _, done, _ = env.make_opponent_move(action)

            if done:
                if env.winner == 'X':
                    stats['wins' if agent_is_x else 'losses'] += 1
                elif env.winner == 'O':
                    stats['losses' if agent_is_x else 'wins'] += 1
                else:
                    stats['draws'] += 1
                break

            valid = env.get_valid_actions()
            if valid:
                opp_action = np.random.choice(valid)
                if agent_is_x:
                    obs, _, done, _ = env.make_opponent_move(opp_action)
                else:
                    obs, _, done, _ = env.step(opp_action)

                if done:
                    if env.winner == 'X':
                        stats['wins' if agent_is_x else 'losses'] += 1
                    elif env.winner == 'O':
                        stats['losses' if agent_is_x else 'wins'] += 1
                    else:
                        stats['draws'] += 1
                    break

    return stats


def train_agent(n_games=1000, epsilon_start=0.3, epsilon_end=0.0,
                use_self_play=True):
    """
    Train agent using Monte Carlo learning with episode-based updates.
    Like tictactoe.py - uses self-play where both players learn from same Q-table.
    """
    agent = MonteCarloQAgent(epsilon=epsilon_start)
    env = TicTacToeEnv()
    results = []

    for game_num in range(n_games):
        obs = env.reset()

        # Epsilon decay (linear schedule like tictactoe.py)
        if n_games > 1:
            agent.epsilon = epsilon_start + \
                (epsilon_end - epsilon_start) * (game_num / (n_games - 1))
        else:
            agent.epsilon = epsilon_end

        # Store episode: (canonical_state, canonical_action, player_id)
        # player_id: 0 for X, 1 for O (like tictactoe.py)
        episode = []
        current_player_id = 0  # X starts

        # Play self-play game
        while not env.done:
            agent_is_x = (current_player_id == 0)
            action, canon_state, action_canon = agent.select_action(
                obs, agent_is_x, training=True)

            # Store transition with player_id
            episode.append((canon_state, action_canon, current_player_id))

            # Make move
            if agent_is_x:
                next_obs, _, done, _ = env.step(action)
            else:
                next_obs, _, done, _ = env.make_opponent_move(action)

            if done:
                # Determine winner (like tictactoe.py: 0=X, 1=O, None=draw)
                if env.winner == 'X':
                    winner_id = 0
                    result = 'X'
                elif env.winner == 'O':
                    winner_id = 1
                    result = 'O'
                else:
                    winner_id = None
                    result = 'draw'

                # Update all moves in episode with final outcome
                agent.update_from_episode(episode, winner_id)
                results.append(result)
                break

            # Switch player
            current_player_id = 1 - current_player_id
            obs = next_obs

        # Report every 100 games
        if (game_num + 1) % 100 == 0:
            recent = results[-200:] if len(results) >= 200 else results
            x_wins = recent.count('X')
            o_wins = recent.count('O')
            draws = recent.count('draw')
            total = len(recent)

            test_stats = test_agent(agent, n_games=200)
            test_losses = test_stats['losses']

            print(f"Games {game_num + 1} | "
                  f"Train X/O/D: {x_wins}/{o_wins}/{draws} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"States: {len(agent.Q)} | "
                  f"Test Loss: {test_losses} ({100*test_losses/200:.1f}%)")

    return agent


def main():
    print("=" * 70)
    print("Monte Carlo Q-Learning with Canonical States")
    print("=" * 70)

    print("\n1. Training agent with self-play (1000 games)...")
    print("   Using Monte Carlo episode-based updates (like tictactoe.py)")
    agent = train_agent(n_games=1500, epsilon_start=0.9, epsilon_end=0.0,
                        use_self_play=True)

    print(f"\n2. Final test vs random (2000 games)...")
    print(f"   Q-table has {len(agent.Q)} canonical states")
    stats = test_agent(agent, n_games=2000)
    print(f"   Wins: {stats['wins']} ({100*stats['wins']/2000:.1f}%)")
    print(f"   Losses: {stats['losses']} ({100*stats['losses']/2000:.1f}%)")
    print(f"   Draws: {stats['draws']} ({100*stats['draws']/2000:.1f}%)")

    loss_rate = stats['losses'] / 2000
    print(f"\n3. Results:")
    print("=" * 70)
    if loss_rate == 0:
        print("✓ PERFECT: Agent achieves 0% loss rate!")
        return True
    elif loss_rate < 0.002:
        print(
            f"✓ NEAR-PERFECT: {stats['losses']} losses in 2000 games ({100*loss_rate:.2f}%)")
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
