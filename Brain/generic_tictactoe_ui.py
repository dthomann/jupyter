import tkinter as tk
from tkinter import ttk, messagebox
import torch
from generic_tictactoe import PolicyNet, train, check_winner, legal_mask, X, O, EMPTY


class TicTacToeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe AI Trainer & Player")
        self.root.geometry("500x600")

        # Initialize model
        self.model = PolicyNet()
        self.model.eval()

        # Game state
        self.board = [EMPTY] * 9
        self.game_over = False
        self.user_player = X  # User plays as X by default
        self.ai_player = O

        # Create UI
        self.create_widgets()
        self.update_board()

    def create_widgets(self):
        # Training section
        train_frame = ttk.LabelFrame(self.root, text="Training", padding=10)
        train_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(train_frame, text="Episodes:").grid(
            row=0, column=0, sticky=tk.W, padx=5)
        self.episodes_var = tk.StringVar(value="800")
        ttk.Entry(train_frame, textvariable=self.episodes_var,
                  width=10).grid(row=0, column=1, padx=5)

        ttk.Label(train_frame, text="Learning Rate:").grid(
            row=0, column=2, sticky=tk.W, padx=5)
        self.lr_var = tk.StringVar(value="0.01")
        ttk.Entry(train_frame, textvariable=self.lr_var,
                  width=10).grid(row=0, column=3, padx=5)

        ttk.Button(train_frame, text="Train Model", command=self.train_model).grid(
            row=1, column=0, columnspan=4, pady=5)

        # Player selection
        player_frame = ttk.LabelFrame(
            self.root, text="Player Selection", padding=10)
        player_frame.pack(fill=tk.X, padx=10, pady=5)

        self.player_var = tk.StringVar(value="X")
        ttk.Radiobutton(player_frame, text="Play as X (First)", variable=self.player_var,
                        value="X", command=self.set_player).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(player_frame, text="Play as O (Second)", variable=self.player_var,
                        value="O", command=self.set_player).pack(side=tk.LEFT, padx=10)

        # Status label
        self.status_label = ttk.Label(self.root, text="Game ready! Click a cell to start.",
                                      font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Board frame
        board_frame = ttk.Frame(self.root)
        board_frame.pack(pady=10)

        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                idx = i * 3 + j
                btn = tk.Button(board_frame, text="", font=("Arial", 24), width=4, height=2,
                                command=lambda idx=idx: self.make_move(idx))
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.buttons.append(row)

        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        ttk.Button(control_frame, text="New Game",
                   command=self.new_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Model",
                   command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Model",
                   command=self.load_model).pack(side=tk.LEFT, padx=5)

    def set_player(self):
        if self.player_var.get() == "X":
            self.user_player = X
            self.ai_player = O
        else:
            self.user_player = O
            self.ai_player = X
        self.new_game()

    def train_model(self):
        try:
            episodes = int(self.episodes_var.get())
            lr = float(self.lr_var.get())

            self.status_label.config(text=f"Training... ({episodes} episodes)")
            self.root.update()

            # Reinitialize model for fresh training
            self.model = PolicyNet()
            train(self.model, episodes=episodes, lr=lr)
            self.model.eval()

            self.status_label.config(text=f"Training complete! Ready to play.")
            messagebox.showinfo(
                "Training", f"Model trained with {episodes} episodes!")
            self.new_game()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def make_move(self, idx):
        if self.game_over or self.board[idx] != EMPTY:
            return

        # User move
        self.board[idx] = self.user_player
        self.update_board()

        winner = check_winner(self.board)
        if winner is not None:
            self.end_game(winner)
            return

        # AI move
        self.root.after(500, self.ai_move)  # Small delay for better UX

    def ai_move(self):
        if self.game_over:
            return

        # Get best move from model
        inp = torch.tensor(self.board, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(inp)
            logits = logits + legal_mask(self.board)
            action = torch.argmax(logits).item()

        if self.board[action] == EMPTY:
            self.board[action] = self.ai_player
            self.update_board()

            winner = check_winner(self.board)
            if winner is not None:
                self.end_game(winner)

    def update_board(self):
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                cell = self.board[idx]
                btn = self.buttons[i][j]

                if cell == X:
                    btn.config(text="X", state=tk.DISABLED,
                               disabledforeground="blue")
                elif cell == O:
                    btn.config(text="O", state=tk.DISABLED,
                               disabledforeground="red")
                else:
                    btn.config(
                        text="", state=tk.NORMAL if not self.game_over else tk.DISABLED)

    def end_game(self, winner):
        self.game_over = True

        if winner == 0:
            self.status_label.config(text="Game Over: Draw!")
        elif winner == self.user_player:
            self.status_label.config(text="Game Over: You Win! 🎉")
        else:
            self.status_label.config(text="Game Over: AI Wins!")

        # Disable all buttons
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.DISABLED)

    def new_game(self):
        self.board = [EMPTY] * 9
        self.game_over = False
        self.update_board()

        if self.user_player == X:
            self.status_label.config(text="Your turn! (You are X)")
        else:
            self.status_label.config(text="AI's turn first... (You are O)")
            self.root.after(500, self.ai_move)  # AI goes first if user is O

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), "generic_tictactoe_model.pth")
            messagebox.showinfo(
                "Success", "Model saved to generic_tictactoe_model.pth")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {e}")

    def load_model(self):
        try:
            self.model.load_state_dict(
                torch.load("generic_tictactoe_model.pth"))
            self.model.eval()
            messagebox.showinfo(
                "Success", "Model loaded from generic_tictactoe_model.pth")
            self.new_game()
        except FileNotFoundError:
            messagebox.showerror(
                "Error", "Model file not found. Train a model first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeUI(root)
    root.mainloop()
