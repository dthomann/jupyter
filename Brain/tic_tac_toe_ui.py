#!/usr/bin/env python3
"""
Tic Tac Toe game with UI for playing against a BrainAgent.
The agent learns and improves over time.
"""

import numpy as np
import os
from tic_tac_toe_env import TicTacToeEnv
from brain import BrainAgent


class TicTacToeUI:
    """UI for playing Tic Tac Toe against a learning agent."""
    
    def __init__(self, agent_path=None, save_path='tictactoe_agent.pkl'):
        self.env = TicTacToeEnv()
        self.save_path = save_path
        
        # Initialize agent
        if agent_path and os.path.exists(agent_path):
            print(f"Loading agent from {agent_path}...")
            self.agent = BrainAgent.load(agent_path)
            print(f"Loaded agent with {self.agent.global_step} training steps")
        else:
            print("Creating new agent...")
            # Tic Tac Toe: 9 positions, 9 actions
            self.agent = BrainAgent(
                obs_dim=9,  # 3x3 board flattened
                latent_dims=[32, 16],
                n_actions=9,
                lr_model=1e-3,
                lr_policy=1e-2,
                replay_batch_size=32,
                use_raw_obs_for_policy=True,  # Use raw board state for stable policy learning
            )
            print("New agent created")
        
        self.stats = {
            'games_played': 0,
            'agent_wins': 0,
            'human_wins': 0,
            'draws': 0,
        }
    
    def print_board(self):
        """Display the board with position numbers."""
        board_2d = self.env.board.reshape(3, 3)
        symbols = {1: 'X', -1: 'O', 0: ' '}
        
        print("\n" + "="*30)
        print("Current Board (Agent=X, You=O)")
        print("="*30)
        print("\n  0   1   2")
        for i in range(3):
            row_str = f"{i} "
            for j in range(3):
                val = board_2d[i, j]
                pos = i * 3 + j
                if val == 0:
                    row_str += f" {pos} "  # Show position number if empty
                else:
                    row_str += f" {symbols[val]} "
                if j < 2:
                    row_str += "|"
            print(row_str)
            if i < 2:
                print("  -----------")
        print()
    
    def get_human_move(self):
        """Get move from human player."""
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None
        
        while True:
            try:
                print(f"Valid positions: {valid_actions}")
                move_str = input("Enter your move (0-8, or 'q' to quit): ").strip()
                
                if move_str.lower() == 'q':
                    return None
                
                move = int(move_str)
                if move in valid_actions:
                    return move
                else:
                    print(f"Invalid move! Position {move} is not available.")
            except ValueError:
                print("Please enter a number between 0-8 or 'q' to quit.")
            except KeyboardInterrupt:
                print("\nQuitting...")
                return None
    
    def play_game(self, agent_first=True, temperature=0.1):
        """Play a single game."""
        obs = self.env.reset()
        self.stats['games_played'] += 1
        
        print(f"\n{'='*50}")
        print(f"Game #{self.stats['games_played']}")
        print(f"{'='*50}")
        
        if agent_first:
            print("Agent goes first (X)")
        else:
            print("You go first (O)")
        
        agent_turn = agent_first
        prev_obs = None
        prev_z = None
        prev_x = None
        prev_action = None
        
        while not self.env.done:
            self.print_board()
            
            if agent_turn:
                # Agent's turn
                print("Agent's turn (thinking)...")
                action, z, value, x = self.agent.act(obs, temperature=temperature, greedy=False)
                
                # If agent chooses invalid move, pick random valid one
                if not self.env._is_valid_move(action):
                    valid_actions = self.env.get_valid_actions()
                    if valid_actions:
                        action = self.env.rng.choice(valid_actions)
                    else:
                        break
                
                print(f"Agent plays position {action}")
                next_obs, reward, done, info = self.env.step(action)
                
                if done:
                    self.print_board()
                    if self.env.winner == 'X':
                        print("🤖 Agent wins!")
                        self.stats['agent_wins'] += 1
                        agent_reward = 1.0
                    elif self.env.winner == 'draw':
                        print("🤝 It's a draw!")
                        self.stats['draws'] += 1
                        agent_reward = 0.0
                    else:
                        agent_reward = 0.0
                    
                    # Learn from this winning/drawing move
                    self.agent.online_update(
                        obs=obs,
                        x=x,
                        z=z,
                        action=action,
                        external_reward=agent_reward,
                        next_obs=next_obs,
                        done=True,
                    )
                    break
                
                # Store agent's move for learning after opponent responds
                prev_obs = obs.copy()
                prev_z = z
                prev_x = x
                prev_action = action
                obs = next_obs
                
            else:
                # Human's turn
                print("Your turn (O)")
                action = self.get_human_move()
                
                if action is None:
                    print("Game cancelled.")
                    return False
                
                next_obs, reward, done, info = self.env.make_opponent_move(action)
                
                if done:
                    self.print_board()
                    if self.env.winner == 'O':
                        print("🎉 You win!")
                        self.stats['human_wins'] += 1
                        agent_reward = -1.0
                    elif self.env.winner == 'draw':
                        print("🤝 It's a draw!")
                        self.stats['draws'] += 1
                        agent_reward = 0.0
                    else:
                        agent_reward = 0.0
                    
                    # Agent learns from losing/drawing (if agent had made a move before)
                    if prev_obs is not None:
                        self.agent.online_update(
                            obs=prev_obs,
                            x=prev_x,
                            z=prev_z,
                            action=prev_action,
                            external_reward=agent_reward,
                            next_obs=next_obs,
                            done=True,
                        )
                    break
                
                # Agent learns from the transition (if agent had made a move before)
                if prev_obs is not None:
                    self.agent.online_update(
                        obs=prev_obs,
                        x=prev_x,
                        z=prev_z,
                        action=prev_action,
                        external_reward=0.0,  # No immediate reward
                        next_obs=next_obs,
                        done=False,
                    )
                
                obs = next_obs
            
            agent_turn = not agent_turn
        
        return True
    
    def print_stats(self):
        """Print game statistics."""
        print("\n" + "="*50)
        print("Statistics")
        print("="*50)
        print(f"Games played: {self.stats['games_played']}")
        print(f"Agent wins: {self.stats['agent_wins']} ({100*self.stats['agent_wins']/max(1, self.stats['games_played']):.1f}%)")
        print(f"Human wins: {self.stats['human_wins']} ({100*self.stats['human_wins']/max(1, self.stats['games_played']):.1f}%)")
        print(f"Draws: {self.stats['draws']} ({100*self.stats['draws']/max(1, self.stats['games_played']):.1f}%)")
        print(f"Agent training steps: {self.agent.global_step}")
        print("="*50)
    
    def save_agent(self):
        """Save the agent state."""
        self.agent.save(self.save_path)
        print(f"\nAgent saved to {self.save_path}")
    
    def run(self):
        """Main game loop."""
        print("\n" + "="*60)
        print("Tic Tac Toe - Play against a Learning Agent!")
        print("="*60)
        print("\nInstructions:")
        print("- Enter a number 0-8 to place your O")
        print("- Positions are numbered:")
        print("  0 1 2")
        print("  3 4 5")
        print("  6 7 8")
        print("- Type 'q' to quit")
        print("- The agent learns from each game and improves!")
        print()
        
        try:
            while True:
                # Ask who goes first
                first = input("Who goes first? (a)gent / (y)ou / (r)andom: ").strip().lower()
                if first == 'q':
                    break
                elif first == 'r':
                    agent_first = np.random.choice([True, False])
                elif first == 'y':
                    agent_first = False
                else:
                    agent_first = True
                
                # Play game
                continue_game = self.play_game(agent_first=agent_first)
                
                if not continue_game:
                    break
                
                # Show stats
                self.print_stats()
                
                # Ask to continue
                cont = input("\nPlay again? (y/n): ").strip().lower()
                if cont != 'y':
                    break
                
                # Periodic offline learning
                if self.agent.global_step % 10 == 0:
                    print("Agent is reviewing past games...")
                    self.agent.offline_replay(n_batches=5)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        
        finally:
            # Save agent before quitting
            self.save_agent()
            self.print_stats()
            print("\nThanks for playing!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Tic Tac Toe against a learning agent')
    parser.add_argument('--load', type=str, default=None, help='Path to load agent from')
    parser.add_argument('--save', type=str, default='tictactoe_agent.pkl', help='Path to save agent')
    args = parser.parse_args()
    
    ui = TicTacToeUI(agent_path=args.load, save_path=args.save)
    ui.run()


if __name__ == '__main__':
    main()

