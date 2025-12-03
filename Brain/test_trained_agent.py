#!/usr/bin/env python3
"""
Test a trained agent against random and heuristic players to verify learning.
"""

import random
import torch
import numpy as np
from brain import BrainAgent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


EMPTY = 0
X = 1
O = -1
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]
CORNERS = [0, 2, 6, 8]
CENTER = 4
EDGES = [1, 3, 5, 7]


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


def find_winning_move(board, player):
    """Find a winning move for the given player, or None if no winning move exists."""
    for move in get_legal_moves(board):
        board[move] = player
        if check_winner(board) == player:
            board[move] = EMPTY
            return move
        board[move] = EMPTY
    return None


def find_blocking_move(board, opponent):
    """Find a move that blocks the opponent from winning, or None if no blocking move needed."""
    return find_winning_move(board, opponent)


def heuristic_move(board, player):
    """
    Make a move using a good heuristic strategy:
    1. Win if possible
    2. Block opponent from winning
    3. Take center if available
    4. Take a corner if available
    5. Take an edge if available
    """
    opponent = -player
    legal = get_legal_moves(board)

    if not legal:
        return None

    # 1. Win if possible
    winning = find_winning_move(board, player)
    if winning is not None:
        return winning

    # 2. Block opponent from winning
    blocking = find_blocking_move(board, opponent)
    if blocking is not None:
        return blocking

    # 3. Take center if available
    if CENTER in legal:
        return CENTER

    # 4. Take a corner if available
    available_corners = [c for c in CORNERS if c in legal]
    if available_corners:
        return random.choice(available_corners)

    # 5. Take any edge
    available_edges = [e for e in EDGES if e in legal]
    if available_edges:
        return random.choice(available_edges)

    # Fallback (shouldn't happen)
    return random.choice(legal)


def test_against_opponent(agent_path, num_games=1000, opponent_type='random'):
    """
    Test the trained agent against an opponent.

    Args:
        agent_path: Path to saved agent file
        num_games: Number of games to play
        opponent_type: 'random' or 'heuristic'

    Returns:
        dict with statistics: wins, losses, draws, win_rate, loss_rate, draw_rate, opponent_type
    """
    print(f"Loading agent from {agent_path}...")
    agent = BrainAgent.load(agent_path)
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
                    # (not 1.0 for legal, which biases the policy incorrectly)
                    mask = torch.tensor(
                        [0.0 if c == 0 else float("-inf") for c in board])
                    logits = logits + mask
                    action = torch.argmax(logits).item()
            else:
                # Opponent's turn
                if opponent_type == 'random':
                    legal = get_legal_moves(board)
                    action = random.choice(legal)
                elif opponent_type == 'heuristic':
                    action = heuristic_move(board, current_player)
                else:
                    raise ValueError(f"Unknown opponent_type: {opponent_type}")

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
        'draw_rate': draws / num_games,
        'opponent_type': opponent_type
    }

    return stats


def print_test_stats(stats):
    """Print test statistics in a readable format."""
    opponent_name = stats.get('opponent_type', 'Unknown').title()
    print("\n" + "="*50)
    print(f"Test Results Against {opponent_name} Player")
    print("="*50)
    print(f"Total Games:     {stats['total_games']}")
    print(f"Wins:            {stats['wins']} ({stats['win_rate']*100:.2f}%)")
    print(
        f"Losses:          {stats['losses']} ({stats['loss_rate']*100:.2f}%)")
    print(f"Draws:           {stats['draws']} ({stats['draw_rate']*100:.2f}%)")
    print("="*50 + "\n")


def evaluate_performance(stats):
    """Evaluate and print performance assessment."""
    if stats['loss_rate'] == 0:
        print("✓ PERFECT: Agent achieves 0% loss rate!")
    elif stats['loss_rate'] < 0.002:
        print(
            f"✓ NEAR-PERFECT: {stats['losses']} losses in {stats['total_games']} games ({100*stats['loss_rate']:.2f}%)")
    elif stats['loss_rate'] < 0.01:
        print("✓ EXCELLENT: < 1% loss rate")
    elif stats['loss_rate'] < 0.05:
        print("✓ GOOD: < 5% loss rate")
    else:
        print(f"✗ NEEDS IMPROVEMENT: {100*stats['loss_rate']:.1f}% loss rate")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Test trained agent against random and heuristic players')
    parser.add_argument('--agent', type=str, required=True,
                        help='Path to saved agent file')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of test games per opponent')
    args = parser.parse_args()

    # Test against random player
    print("\n" + "="*60)
    print("TESTING AGAINST RANDOM PLAYER")
    print("="*60)
    stats_random = test_against_opponent(
        args.agent, num_games=args.games, opponent_type='random')
    print_test_stats(stats_random)
    evaluate_performance(stats_random)

    # Test against heuristic player
    print("\n" + "="*60)
    print("TESTING AGAINST HEURISTIC PLAYER")
    print("="*60)
    stats_heuristic = test_against_opponent(
        args.agent, num_games=args.games, opponent_type='heuristic')
    print_test_stats(stats_heuristic)
    evaluate_performance(stats_heuristic)

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"Random Player:    Win Rate: {stats_random['win_rate']*100:.2f}%, "
          f"Loss Rate: {stats_random['loss_rate']*100:.2f}%, "
          f"Draw Rate: {stats_random['draw_rate']*100:.2f}%")
    print(f"Heuristic Player: Win Rate: {stats_heuristic['win_rate']*100:.2f}%, "
          f"Loss Rate: {stats_heuristic['loss_rate']*100:.2f}%, "
          f"Draw Rate: {stats_heuristic['draw_rate']*100:.2f}%")
    print("="*60 + "\n")
