#!/usr/bin/env python3
"""Test Monte Carlo update logic."""

from brain import BrainAgent
from tic_tac_toe_env import TicTacToeEnv
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


# Create agent
agent = BrainAgent(
    obs_dim=9,
    latent_dims=[32, 16],
    n_actions=9,
    lr_model=1e-3,
    lr_policy=1e-2,
    replay_batch_size=32,
    use_raw_obs_for_policy=True,
)
agent.intrinsic.curiosity_scale = 0.0
agent.intrinsic.learning_progress_scale = 0.0

# Simulate a short game
env = TicTacToeEnv()
obs = env.reset()
game_history = []
agent_is_x = True

print("Simulating a game...")
print(f"Initial board: {obs}")

# Move 1: X plays
current_obs = obs.copy() if agent_is_x else -obs
action1, z1, value1, x1 = agent.act(current_obs, temperature=1.0, greedy=False)
game_history.append({'obs': current_obs.copy(), 'z': z1,
                    'x': x1, 'action': action1, 'is_x': agent_is_x})
next_obs1, _, done1, _ = env.step(action1)
print(f"Move 1 (X): action={action1}, board after: {next_obs1}, done={done1}")

if not done1:
    obs = next_obs1
    agent_is_x = False

    # Move 2: O plays
    current_obs = obs.copy() if agent_is_x else -obs
    action2, z2, value2, x2 = agent.act(
        current_obs, temperature=1.0, greedy=False)
    game_history.append({'obs': current_obs.copy(), 'z': z2,
                        'x': x2, 'action': action2, 'is_x': agent_is_x})
    next_obs2, _, done2, _ = env.make_opponent_move(action2)
    print(
        f"Move 2 (O): action={action2}, board after: {next_obs2}, done={done2}")

    if not done2:
        obs = next_obs2
        agent_is_x = True

        # Move 3: X plays and wins
        current_obs = obs.copy() if agent_is_x else -obs
        action3, z3, value3, x3 = agent.act(
            current_obs, temperature=1.0, greedy=False)
        game_history.append({'obs': current_obs.copy(
        ), 'z': z3, 'x': x3, 'action': action3, 'is_x': agent_is_x})
        next_obs3, _, done3, _ = env.step(action3)
        print(
            f"Move 3 (X): action={action3}, board after: {next_obs3}, done={done3}, winner={env.winner}")

# Now do Monte Carlo updates
print(f"\nGame finished. Winner: {env.winner}")
print(f"Number of moves: {len(game_history)}")

if env.winner == 'X':
    x_reward = 1.0
    o_reward = -1.0
elif env.winner == 'O':
    x_reward = -1.0
    o_reward = 1.0
else:
    x_reward = 0.0
    o_reward = 0.0

print(f"\nRewards: x_reward={x_reward}, o_reward={o_reward}")

gamma = 0.99
final_obs = env._get_obs()
print(f"Final board: {final_obs}")

print("\nMonte Carlo updates:")
for i in range(len(game_history) - 1, -1, -1):
    move_data = game_history[i]
    steps_to_end = len(game_history) - 1 - i

    if move_data['is_x']:
        move_reward = x_reward * (gamma ** steps_to_end)
        final_obs_for_move = final_obs
    else:
        move_reward = o_reward * (gamma ** steps_to_end)
        final_obs_for_move = -final_obs

    print(f"\nMove {i} ({'X' if move_data['is_x'] else 'O'}):")
    print(f"  Stored obs: {move_data['obs']}")
    print(f"  Final obs for move: {final_obs_for_move}")
    print(f"  Action: {move_data['action']}")
    print(f"  Steps to end: {steps_to_end}")
    print(f"  Move reward (MC return): {move_reward:.4f}")

    # Check if stored obs matches what we're using
    if move_data['is_x']:
        if not np.allclose(move_data['obs'], final_obs_for_move[:len(move_data['obs'])]):
            print(f"  WARNING: Stored obs doesn't match final_obs!")
    else:
        expected_flipped = -final_obs
        if not np.allclose(move_data['obs'], expected_flipped):
            print(f"  WARNING: Stored obs doesn't match flipped final_obs!")
            print(f"    Expected: {expected_flipped}")
            print(f"    Got: {move_data['obs']}")
