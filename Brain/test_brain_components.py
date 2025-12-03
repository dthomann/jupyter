#!/usr/bin/env python3
"""
Diagnostic test to identify why BrainAgent doesn't match generic_tictactoe.py performance.
Tests components progressively to isolate the issue.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from brain.actor_critic import ActorCritic
from brain import BrainAgent
from tic_tac_toe_env import TicTacToeEnv

EMPTY = 0
X = 1
O = -1
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


def check_winner(b):
    for a, b1, c in WIN_LINES:
        if b[a] != 0 and b[a] == b[b1] == b[c]:
            return b[a]
    if 0 not in b:
        return 0
    return None


def legal_mask(board):
    """Same as generic_tictactoe.py"""
    m = torch.tensor([1.0 if c == 0 else float("-inf") for c in board])
    return m


def get_legal_moves(board):
    return [i for i in range(9) if board[i] == EMPTY]


def test_against_random_torch(model, num_games=1000):
    """Test a torch model against random player."""
    model.eval()
    wins, losses, draws = 0, 0, 0
    
    for _ in range(num_games):
        board = [EMPTY] * 9
        current_player = X
        
        while True:
            if current_player == X:
                inp = torch.tensor(board, dtype=torch.float32)
                with torch.no_grad():
                    # Use policy_net directly
                    if hasattr(model, 'policy_net'):
                        logits = model.policy_net(inp)
                    else:
                        logits = model(inp)
                    logits = logits + legal_mask(board)
                    move = torch.argmax(logits).item()
            else:
                legal = get_legal_moves(board)
                move = random.choice(legal)
            
            board[move] = current_player
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
    
    return {'wins': wins, 'losses': losses, 'draws': draws, 'total': num_games,
            'loss_rate': losses / num_games}


# ============================================================================
# TEST 1: Pure ActorCritic with EXACT same training as generic_tictactoe.py
# ============================================================================
def test_pure_actor_critic():
    """Test ActorCritic directly, bypassing BrainAgent completely."""
    print("\n" + "="*70)
    print("TEST 1: Pure ActorCritic (same training as generic_tictactoe.py)")
    print("="*70)
    
    # Create ActorCritic with same architecture
    ac = ActorCritic(
        state_dim=9,
        n_actions=9,
        policy_hidden_dims=(32,),
        value_hidden_dims=(32,),
        activation="relu",
        entropy_coeff=0.001,
    )
    
    opt = optim.Adam(ac.parameters(), lr=0.001)
    
    for episode in range(5000):
        board = [0]*9
        player = 1
        logps = []
        
        # Play episode exactly like generic_tictactoe.py
        while True:
            inp = torch.tensor(board, dtype=torch.float32)
            logits = ac.policy_net(inp)
            logits = logits + legal_mask(board)
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logps.append((dist.log_prob(action), player))
            
            board[action.item()] = player
            w = check_winner(board)
            if w is not None:
                outcome = 0.0 if w == 0 else float(w)
                break
            player = -player
        
        # Train exactly like generic_tictactoe.py
        opt.zero_grad()
        loss = 0
        for logp, p in logps:
            r = outcome * p
            loss -= logp * r
            loss -= 0.001 * (-logp.exp() * logp)  # entropy bonus
        loss.backward()
        opt.step()
        
        if (episode + 1) % 1000 == 0:
            stats = test_against_random_torch(ac, num_games=200)
            print(f"Episode {episode+1}: Loss rate = {100*stats['loss_rate']:.1f}%")
    
    stats = test_against_random_torch(ac, num_games=1000)
    print(f"\nFinal: Wins={stats['wins']}, Losses={stats['losses']}, Draws={stats['draws']}")
    print(f"Loss rate: {100*stats['loss_rate']:.2f}%")
    return stats['loss_rate'] < 0.05


# ============================================================================
# TEST 2: ActorCritic with update_reinforce method
# ============================================================================
def test_update_reinforce():
    """Test if update_reinforce method works correctly."""
    print("\n" + "="*70)
    print("TEST 2: ActorCritic with update_reinforce method")
    print("="*70)
    
    ac = ActorCritic(
        state_dim=9,
        n_actions=9,
        policy_hidden_dims=(32,),
        value_hidden_dims=(32,),
        activation="relu",
        entropy_coeff=0.001,
    )
    
    # Create optimizer once
    ac.optimizer = optim.Adam(ac.parameters(), lr=0.001)
    
    for episode in range(5000):
        board = [0]*9
        player = 1
        states = []
        actions = []
        players = []
        masks = []
        
        # Play episode
        while True:
            inp = board.copy()
            mask = [0.0 if c == 0 else float("-inf") for c in board]
            
            inp_t = torch.tensor(inp, dtype=torch.float32)
            logits = ac.policy_net(inp_t)
            logits = logits + torch.tensor(mask)
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            
            states.append(np.array(inp, dtype=np.float32))
            actions.append(action)
            players.append(player)
            masks.append(np.array(mask, dtype=np.float32))
            
            board[action] = player
            w = check_winner(board)
            if w is not None:
                outcome = 0.0 if w == 0 else float(w)
                break
            player = -player
        
        # Compute rewards for each player
        rewards = [outcome * p for p in players]
        
        # Use update_reinforce
        ac.update_reinforce(states, actions, rewards, masks, entropy_coeff=0.001)
        
        if (episode + 1) % 1000 == 0:
            stats = test_against_random_torch(ac, num_games=200)
            print(f"Episode {episode+1}: Loss rate = {100*stats['loss_rate']:.1f}%")
    
    stats = test_against_random_torch(ac, num_games=1000)
    print(f"\nFinal: Wins={stats['wins']}, Losses={stats['losses']}, Draws={stats['draws']}")
    print(f"Loss rate: {100*stats['loss_rate']:.2f}%")
    return stats['loss_rate'] < 0.05


# ============================================================================
# TEST 3: BrainAgent with minimal components (no world model encoding)
# ============================================================================
def test_brain_minimal():
    """Test BrainAgent with minimal components."""
    print("\n" + "="*70)
    print("TEST 3: BrainAgent with minimal components (use_raw_obs=True)")
    print("="*70)
    
    ac = ActorCritic(
        state_dim=9,
        n_actions=9,
        policy_hidden_dims=(32,),
        value_hidden_dims=(32,),
        activation="relu",
        entropy_coeff=0.001,
    )
    ac.optimizer = optim.Adam(ac.parameters(), lr=0.001)
    
    agent = BrainAgent(
        obs_dim=9,
        latent_dims=[32, 16],
        n_actions=9,
        lr_model=1e-3,
        lr_policy=0.001,
        use_raw_obs_for_policy=True,
        episode_based_learning=True,
        entropy_coeff=0.001,
    )
    agent.actor_critic = ac
    agent.intrinsic.curiosity_scale = 0.0
    agent.intrinsic.learning_progress_scale = 0.0
    
    for episode in range(5000):
        board = [0]*9
        player = 1
        states = []
        actions = []
        players = []
        masks = []
        
        # Play episode using simple board (not env)
        while True:
            obs = np.array(board, dtype=np.float32)
            mask = np.array([0.0 if c == 0 else float("-inf") for c in board], dtype=np.float32)
            
            # DON'T use agent.act - use policy_net directly like Test 2
            # This tests if encode_state or agent.act is the issue
            inp_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits = agent.actor_critic.policy_net(inp_t)
                logits = logits + torch.tensor(mask)
                probs = torch.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            # Still get x from encode_state for comparison
            _, x = agent.encode_state(obs)
            
            states.append(x.copy())
            actions.append(action)
            players.append(player)
            masks.append(mask.copy())
            
            board[action] = player
            w = check_winner(board)
            if w is not None:
                outcome = 0.0 if w == 0 else float(w)
                break
            player = -player
        
        # Compute rewards for each player
        rewards = [outcome * p for p in players]
        
        # Use actor_critic.update_reinforce directly
        agent.actor_critic.update_reinforce(states, actions, rewards, masks, entropy_coeff=0.001)
        
        if (episode + 1) % 1000 == 0:
            stats = test_against_random_torch(agent.actor_critic, num_games=200)
            print(f"Episode {episode+1}: Loss rate = {100*stats['loss_rate']:.1f}%")
    
    stats = test_against_random_torch(agent.actor_critic, num_games=1000)
    print(f"\nFinal: Wins={stats['wins']}, Losses={stats['losses']}, Draws={stats['draws']}")
    print(f"Loss rate: {100*stats['loss_rate']:.2f}%")
    return stats['loss_rate'] < 0.05


# ============================================================================
# TEST 4: BrainAgent with TicTacToeEnv (identifying env-related issues)
# ============================================================================
def test_brain_with_env():
    """Test BrainAgent with TicTacToeEnv."""
    print("\n" + "="*70)
    print("TEST 4: BrainAgent with TicTacToeEnv")
    print("="*70)
    
    ac = ActorCritic(
        state_dim=9,
        n_actions=9,
        policy_hidden_dims=(32,),
        value_hidden_dims=(32,),
        activation="relu",
        entropy_coeff=0.001,
    )
    ac.optimizer = optim.Adam(ac.parameters(), lr=0.001)
    
    agent = BrainAgent(
        obs_dim=9,
        latent_dims=[32, 16],
        n_actions=9,
        lr_model=1e-3,
        lr_policy=0.001,
        use_raw_obs_for_policy=True,
        episode_based_learning=True,
        entropy_coeff=0.001,
    )
    agent.actor_critic = ac
    agent.intrinsic.curiosity_scale = 0.0
    agent.intrinsic.learning_progress_scale = 0.0
    
    env = TicTacToeEnv()
    
    for episode in range(5000):
        obs = env.reset()
        player = 1  # X starts
        states = []
        actions = []
        players = []
        masks = []
        
        while not env.done:
            # Convert obs to list for legal_mask
            board = obs.tolist()
            mask = np.array([0.0 if c == 0 else float("-inf") for c in board], dtype=np.float32)
            
            x = np.array(obs, dtype=np.float32)
            
            # Get action from policy
            inp_t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                logits = agent.actor_critic.policy_net(inp_t)
                logits = logits + torch.tensor(mask)
                probs = torch.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            states.append(x.copy())
            actions.append(action)
            players.append(player)
            masks.append(mask.copy())
            
            # Make move using env
            if player == 1:
                obs, _, done, _ = env.step(action)
            else:
                obs, _, done, _ = env.make_opponent_move(action)
            
            if done:
                break
            
            player = -player
        
        # Determine outcome
        if env.winner == 'X':
            outcome = 1.0
        elif env.winner == 'O':
            outcome = -1.0
        else:
            outcome = 0.0
        
        # Compute rewards for each player
        rewards = [outcome * p for p in players]
        
        # Update
        agent.actor_critic.update_reinforce(states, actions, rewards, masks, entropy_coeff=0.001)
        
        if (episode + 1) % 1000 == 0:
            stats = test_against_random_torch(agent.actor_critic, num_games=200)
            print(f"Episode {episode+1}: Loss rate = {100*stats['loss_rate']:.1f}%")
    
    stats = test_against_random_torch(agent.actor_critic, num_games=1000)
    print(f"\nFinal: Wins={stats['wins']}, Losses={stats['losses']}, Draws={stats['draws']}")
    print(f"Loss rate: {100*stats['loss_rate']:.2f}%")
    return stats['loss_rate'] < 0.05


# ============================================================================
# TEST 5: Compare update_reinforce loss calculation
# ============================================================================
def test_loss_calculation():
    """Compare loss calculation between generic_tictactoe and update_reinforce."""
    print("\n" + "="*70)
    print("TEST 5: Comparing loss calculation")
    print("="*70)
    
    # Sample data
    states = [np.array([1, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)]
    actions = [2]
    rewards = [1.0]  # X wins
    masks = [np.array([float("-inf"), 0, 0, 0, float("-inf"), 0, 0, 0, 0], dtype=np.float32)]
    
    # Method 1: Generic tictactoe style (accumulate, then backward)
    ac1 = ActorCritic(state_dim=9, n_actions=9, policy_hidden_dims=(32,), activation="relu")
    
    inp = torch.tensor(states[0], dtype=torch.float32)
    logits = ac1.policy_net(inp)
    logits = logits + torch.tensor(masks[0])
    probs = torch.softmax(logits, dim=0)
    dist = torch.distributions.Categorical(probs)
    logp = dist.log_prob(torch.tensor(actions[0]))
    
    loss1 = -logp * rewards[0]
    loss1 = loss1 - 0.001 * (-logp.exp() * logp)
    print(f"Method 1 (generic): loss = {loss1.item():.6f}")
    
    # Method 2: update_reinforce (uses mean)
    ac2 = ActorCritic(state_dim=9, n_actions=9, policy_hidden_dims=(32,), activation="relu")
    # Copy weights
    ac2.load_state_dict(ac1.state_dict())
    
    # Simulate update_reinforce calculation
    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    masks_t = torch.tensor(np.array(masks), dtype=torch.float32)
    
    logits2 = ac2.policy_net(states_t)
    logits2 = logits2 + masks_t
    probs2 = torch.softmax(logits2, dim=1)
    dist2 = torch.distributions.Categorical(probs2)
    log_probs2 = dist2.log_prob(actions_t)
    
    policy_loss = -(log_probs2 * rewards_t).mean()
    selected_probs = probs2.gather(1, actions_t.unsqueeze(1)).squeeze(1)
    entropy_term = -(selected_probs * log_probs2)
    loss2 = policy_loss - 0.001 * entropy_term.mean()
    print(f"Method 2 (update_reinforce): loss = {loss2.item():.6f}")
    
    print(f"\nDifference: {abs(loss1.item() - loss2.item()):.6f}")
    print("If difference is significant, this explains the performance gap.")


def main():
    print("="*70)
    print("DIAGNOSTIC TESTS: BrainAgent vs generic_tictactoe.py")
    print("="*70)
    
    # Test 5 first to check loss calculation
    test_loss_calculation()
    
    results = {}
    
    # Test 1: Pure ActorCritic (should match generic_tictactoe.py)
    results['test1'] = test_pure_actor_critic()
    
    # Test 2: update_reinforce method
    results['test2'] = test_update_reinforce()
    
    # Test 3: BrainAgent minimal
    results['test3'] = test_brain_minimal()
    
    # Test 4: BrainAgent with env
    results['test4'] = test_brain_with_env()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test}: {status}")


if __name__ == '__main__':
    main()

