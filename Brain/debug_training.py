#!/usr/bin/env python3
"""Debug script to check what's happening during training."""

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

print("Intrinsic motivation scales:")
print(f"  curiosity_scale: {agent.intrinsic.curiosity_scale}")
print(f"  learning_progress_scale: {agent.intrinsic.learning_progress_scale}")

# Test a single update
env = TicTacToeEnv()
obs = env.reset()

# Agent's move
action, z, value, x = agent.act(obs, temperature=1.0, greedy=False)
next_obs, reward, done, info = env.step(action)

print(f"\nBefore update:")
print(f"  External reward: {reward}")
print(f"  Value estimate: {value:.4f}")

# Check intrinsic reward
z_next, x_next = agent.encode_state(next_obs)
neuromod_factor = 0.5 * agent.neuromodulators.norepinephrine + \
    0.5 * agent.neuromodulators.acetylcholine
_, pred_error_norm = agent.world_model.learn(
    x=x, neuromod_factor=neuromod_factor, lr_model=agent.lr_model)
intrinsic_reward, components = agent.intrinsic.compute(pred_error_norm)
drive_vec = agent.drives.update(components)
drive_gain = 1.0 + np.tanh(drive_vec.mean())
total_reward = float(reward) + drive_gain * intrinsic_reward

print(f"\nAfter world model update:")
print(f"  Prediction error norm: {pred_error_norm:.4f}")
print(f"  Intrinsic reward: {intrinsic_reward:.4f}")
print(f"  Drive gain: {drive_gain:.4f}")
print(f"  Total reward: {total_reward:.4f}")

# Do the update
td_error, pred_err, intrinsic, drives = agent.online_update(
    obs=obs, x=x, z=z, action=action, external_reward=reward, next_obs=next_obs, done=done
)

print(f"\nAfter online_update:")
print(f"  TD error: {td_error:.4f}")
print(f"  Intrinsic reward returned: {intrinsic:.4f}")
print(
    f"  Policy weight change (first action): {np.max(np.abs(agent.actor_critic.W_policy_out[:, action])):.6f}")

# Test with a win
env2 = TicTacToeEnv()
obs2 = env2.reset()
# Make moves to create a winning position
env2.board[0] = 1  # X
env2.board[3] = 1  # X
action2 = 6  # X wins
obs2_final, reward2, done2, info2 = env2.step(action2)
print(f"\n\nWin scenario:")
print(f"  External reward: {reward2}")
print(f"  Winner: {env2.winner}")

# Test with a loss
env3 = TicTacToeEnv()
obs3 = env3.reset()
env3.board[0] = -1  # O
env3.board[1] = -1  # O
env3.board[2] = -1  # O wins
env3.done = True
env3.winner = 'O'
print(f"\n\nLoss scenario:")
print(f"  Winner: {env3.winner}")
