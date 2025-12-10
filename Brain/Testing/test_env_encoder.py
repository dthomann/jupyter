#!/usr/bin/env python3
"""
Quick probes to validate TicTacToe environment dynamics and the multi-modal encoder.
These checks help ensure training logs represent the expected game semantics.
"""

import numpy as np

from tic_tac_toe_env import TicTacToeEnv
from brain.encoder import MultiModalEncoder


def assert_initial_state():
    env = TicTacToeEnv()
    obs = env.reset()
    assert obs.shape == (9,)
    assert np.all(obs == 0), "Reset board should be empty"
    assert env.get_valid_actions() == list(
        range(9)), "All moves should be legal at reset"


def assert_invalid_move_penalty():
    env = TicTacToeEnv()
    env.reset()
    env.step(0)  # valid first move
    obs_before = env.board.copy()
    _, reward, done, info = env.step(0)
    assert reward == -1.0, "Invalid move should incur -1 reward"
    assert not done, "Invalid move should not end the game"
    assert np.array_equal(
        env.board, obs_before), "Board should remain unchanged after invalid move"


def assert_x_win_sequence():
    env = TicTacToeEnv()
    env.reset()
    x_moves = [0, 1, 2]
    o_moves = [3, 4]

    for idx, x_action in enumerate(x_moves):
        _, reward, done, info = env.step(x_action)
        if idx < len(o_moves) and not done:
            _, _, done, _ = env.make_opponent_move(o_moves[idx])
        if done:
            break

    assert env.winner == 'X', f"Expected X to win, got {env.winner}"
    assert reward == 1.0, "Winning move should reward +1"
    assert done, "Game should terminate once X wins"


def assert_draw_sequence():
    env = TicTacToeEnv()
    env.reset()
    moves = [
        ('X', 0), ('O', 1),
        ('X', 2), ('O', 4),
        ('X', 3), ('O', 5),
        ('X', 7), ('O', 6),
        ('X', 8),
    ]

    for player, action in moves:
        if player == 'X':
            _, reward, done, info = env.step(action)
        else:
            _, reward, done, info = env.make_opponent_move(action)
        if done:
            break

    assert env.winner == 'draw', f"Expected draw, got {env.winner}"
    assert reward == 0.0, "Draw should have zero reward"
    assert done and info.get('winner') == 'draw'


def assert_legal_mask_matches_valid_actions():
    env = TicTacToeEnv()
    env.reset()
    played = [0, 4, 8]
    for move in played:
        env.step(move)
        env.make_opponent_move((move + 1) % 9)
    valid_actions = set(env.get_valid_actions())
    mask = [0.0 if i in valid_actions else float('-inf') for i in range(9)]
    for idx, val in enumerate(mask):
        if idx in valid_actions:
            assert val == 0.0
        else:
            assert val == float('-inf')


def assert_encoder_shapes_and_errors():
    dims = {"vision": 4, "audio": 2}
    encoder = MultiModalEncoder(
        dims, hidden_dim_per_modality=3, rng=np.random.RandomState(0))
    sample = {
        "vision": np.ones(4, dtype=np.float32),
        "audio": np.zeros(2, dtype=np.float32),
    }
    encoded = encoder.encode(sample)
    assert encoded.shape == (
        6,), "Encoder output should concat modality embeddings"

    encoded_missing = encoder.encode({"vision": np.arange(4)})
    assert encoded_missing.shape == (
        6,), "Missing modalities should be zero-filled"

    try:
        encoder.encode({"vision": np.ones(5)})
    except ValueError:
        pass
    else:
        raise AssertionError("Dimension mismatch should raise ValueError")


def run_all_checks():
    assert_initial_state()
    assert_invalid_move_penalty()
    assert_x_win_sequence()
    assert_draw_sequence()
    assert_legal_mask_matches_valid_actions()
    assert_encoder_shapes_and_errors()
    print("✓ Environment and encoder sanity checks passed.")


if __name__ == "__main__":
    run_all_checks()
