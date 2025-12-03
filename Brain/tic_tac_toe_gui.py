#!/usr/bin/env python3
"""
Graphical UI for playing Tic Tac Toe against a saved BrainAgent.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
from pathlib import Path
from tic_tac_toe_env import TicTacToeEnv
from brain import BrainAgent


class TicTacToeGUI:
    """Graphical UI for playing Tic Tac Toe against a BrainAgent."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe - Play Against Brain Agent")
        self.root.geometry("600x700")
        
        self.env = None
        self.agent = None
        self.agent_path = None
        self.current_player = None  # 1 = agent (X), -1 = human (O)
        self.game_active = False
        
        self.stats = {
            'games_played': 0,
            'agent_wins': 0,
            'human_wins': 0,
            'draws': 0,
        }
        
        self.setup_ui()
        self.scan_brain_files()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Top frame for brain selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Select Brain Agent:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        brain_frame = ttk.Frame(top_frame)
        brain_frame.pack(fill=tk.X, pady=5)
        
        self.brain_var = tk.StringVar()
        self.brain_combo = ttk.Combobox(brain_frame, textvariable=self.brain_var, 
                                        state="readonly", width=50)
        self.brain_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.brain_combo.bind("<<ComboboxSelected>>", self.on_brain_selected)
        
        ttk.Button(brain_frame, text="Browse...", command=self.browse_brain_file).pack(side=tk.LEFT)
        
        # Status label
        self.status_label = ttk.Label(top_frame, text="Select a brain agent to start", 
                                      font=("Arial", 10))
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # Game board frame
        board_frame = ttk.Frame(self.root, padding="20")
        board_frame.pack(pady=20)
        
        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(board_frame, text="", font=("Arial", 36, "bold"),
                               width=4, height=2, relief=tk.RAISED,
                               command=lambda r=i, c=j: self.on_cell_click(r, c))
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.buttons.append(row)
        
        # Control buttons frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        self.new_game_btn = ttk.Button(control_frame, text="New Game", 
                                       command=self.start_new_game, state=tk.DISABLED)
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Reset Stats", command=self.reset_stats).pack(side=tk.LEFT, padx=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_label = ttk.Label(stats_frame, text="", font=("Arial", 10))
        self.stats_label.pack()
        self.update_stats_display()
    
    def scan_brain_files(self):
        """Scan for .pkl files in the current directory."""
        pkl_files = []
        current_dir = Path(".")
        
        # Look for .pkl files
        for pkl_file in current_dir.glob("*.pkl"):
            pkl_files.append(str(pkl_file))
        
        # Also check checkpoints directory
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for pkl_file in checkpoints_dir.glob("*.pkl"):
                pkl_files.append(str(pkl_file))
        
        # Sort and update combobox
        pkl_files.sort()
        self.brain_combo['values'] = pkl_files
        
        if pkl_files:
            self.brain_combo.current(0)
            self.on_brain_selected()
    
    def browse_brain_file(self):
        """Open file dialog to browse for brain file."""
        filename = filedialog.askopenfilename(
            title="Select Brain Agent File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.brain_var.set(filename)
            self.on_brain_selected()
    
    def on_brain_selected(self, event=None):
        """Handle brain agent selection."""
        brain_path = self.brain_var.get()
        if not brain_path or not os.path.exists(brain_path):
            self.status_label.config(text="Invalid brain file selected")
            return
        
        try:
            self.status_label.config(text=f"Loading brain from {brain_path}...")
            self.root.update()
            
            self.agent = BrainAgent.load(brain_path)
            self.agent_path = brain_path
            
            # Disable intrinsic motivation for playing (not training)
            self.agent.intrinsic.curiosity_scale = 0.0
            self.agent.intrinsic.learning_progress_scale = 0.0
            
            self.status_label.config(text=f"✓ Brain agent loaded successfully! Training steps: {self.agent.global_step} | Starting game...")
            self.new_game_btn.config(state=tk.NORMAL)
            # Auto-start first game
            self.root.after(500, self.start_new_game)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load brain agent:\n{e}")
            self.status_label.config(text="Failed to load brain agent")
    
    def start_new_game(self):
        """Start a new game."""
        if self.agent is None:
            messagebox.showwarning("Warning", "Please select a brain agent first")
            return
        
        self.env = TicTacToeEnv()
        obs = self.env.reset()
        self.game_active = True
        
        # Randomly choose who goes first
        self.current_player = np.random.choice([1, -1])
        
        self.update_board()
        self.stats['games_played'] += 1
        
        if self.current_player == 1:
            self.status_label.config(text="Agent's turn (X)...")
            self.root.update()
            self.root.after(500, self.agent_move)  # Small delay for visual feedback
        else:
            self.status_label.config(text="Your turn (O)")
    
    def update_board(self):
        """Update the visual board."""
        if self.env is None:
            return
        
        board_2d = self.env.board.reshape(3, 3)
        symbols = {1: 'X', -1: 'O', 0: ''}
        colors = {1: 'blue', -1: 'red', 0: 'lightgray'}
        
        for i in range(3):
            for j in range(3):
                val = board_2d[i, j]
                btn = self.buttons[i][j]
                btn.config(text=symbols[val], 
                          state=tk.NORMAL if val == 0 and self.game_active else tk.DISABLED,
                          bg=colors[val] if val != 0 else 'white',
                          fg='white' if val != 0 else 'black')
    
    def on_cell_click(self, row, col):
        """Handle cell click."""
        if not self.game_active or self.env is None:
            return
        
        if self.current_player != -1:
            return  # Not human's turn
        
        action = row * 3 + col
        
        if not self.env._is_valid_move(action):
            messagebox.showwarning("Invalid Move", "This cell is already occupied!")
            return
        
        # Human's move
        obs, reward, done, info = self.env.make_opponent_move(action)
        self.update_board()
        
        if done:
            self.handle_game_end()
            return
        
        self.current_player = 1
        self.status_label.config(text="Agent's turn (X)...")
        self.root.update()
        self.root.after(500, self.agent_move)  # Small delay for visual feedback
    
    def agent_move(self):
        """Make agent's move."""
        if not self.game_active or self.env is None or self.current_player != 1:
            return
        
        obs = self.env._get_obs()
        
        # Get legal actions mask (convert to numpy array)
        legal_actions = self.env.get_valid_actions()
        legal_mask = np.array([0.0 if i in legal_actions else float('-inf') for i in range(9)], dtype=np.float32)
        
        # Agent selects action
        action, z, value, x = self.agent.act(obs, temperature=0.1, greedy=False, legal_mask=legal_mask)
        
        # Ensure valid move
        if not self.env._is_valid_move(action):
            valid_actions = self.env.get_valid_actions()
            if valid_actions:
                action = self.env.rng.choice(valid_actions)
            else:
                return
        
        # Apply agent's move
        obs, reward, done, info = self.env.step(action)
        self.update_board()
        
        if done:
            self.handle_game_end()
            return
        
        self.current_player = -1
        self.status_label.config(text="Your turn (O)")
    
    def handle_game_end(self):
        """Handle game end."""
        self.game_active = False
        
        winner = self.env.winner
        if winner == 'X':
            self.stats['agent_wins'] += 1
            self.status_label.config(text="Game Over: Agent wins! 🤖 Starting new game...")
        elif winner == 'O':
            self.stats['human_wins'] += 1
            self.status_label.config(text="Game Over: You win! 🎉 Starting new game...")
        else:
            self.stats['draws'] += 1
            self.status_label.config(text="Game Over: It's a draw! 🤝 Starting new game...")
        
        self.update_stats_display()
        
        # Auto-start new game after a short delay
        self.root.after(1500, self.start_new_game)
    
    def update_stats_display(self):
        """Update statistics display."""
        total = self.stats['games_played']
        if total == 0:
            stats_text = "No games played yet"
        else:
            agent_pct = 100 * self.stats['agent_wins'] / total
            human_pct = 100 * self.stats['human_wins'] / total
            draw_pct = 100 * self.stats['draws'] / total
            
            stats_text = (f"Games: {total} | "
                         f"Agent: {self.stats['agent_wins']} ({agent_pct:.1f}%) | "
                         f"You: {self.stats['human_wins']} ({human_pct:.1f}%) | "
                         f"Draws: {self.stats['draws']} ({draw_pct:.1f}%)")
        
        self.stats_label.config(text=stats_text)
    
    def reset_stats(self):
        """Reset statistics."""
        if messagebox.askyesno("Reset Stats", "Are you sure you want to reset statistics?"):
            self.stats = {
                'games_played': 0,
                'agent_wins': 0,
                'human_wins': 0,
                'draws': 0,
            }
            self.update_stats_display()


def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

