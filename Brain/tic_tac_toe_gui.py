#!/usr/bin/env python3
"""
Graphical UI for playing Tic Tac Toe against a BrainAgent connected to environment server.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
from multiprocessing.connection import Client
from brainprotocol import OBSERVATION, REWARD, ACTION, TERMINAL, SHUTDOWN


class TicTacToeGUI:
    """Graphical UI for playing Tic Tac Toe against a BrainAgent via environment server."""

    def __init__(self, root, host='localhost', port=6000, authkey=b'brain-secret'):
        self.root = root
        self.root.title("Tic Tac Toe - Play Against Brain Agent (Server Mode)")
        self.root.geometry("600x700")

        self.host = host
        self.port = port
        self.authkey = authkey if isinstance(
            authkey, bytes) else authkey.encode()

        self.conn = None
        self.connected = False
        self.connection_thread = None
        self.message_thread = None
        self.running = True

        self.board = np.zeros(9, dtype=np.int32)  # Current board state
        self.current_turn = None  # 'X' or 'O' - whose turn it is
        self.my_player = None  # 'X' or 'O' - which player the GUI is
        self.game_active = False
        self.episode = 0
        self.waiting_for_next_game = False  # Flag to delay next game start
        self.pending_observation = None  # Store observation during delay

        self.stats = {
            'games_played': 0,
            'agent_wins': 0,
            'human_wins': 0,
            'draws': 0,
        }

        self.setup_ui()
        self.connect_to_server()

    def setup_ui(self):
        """Set up the user interface."""
        # Top frame for connection info
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Environment Server Connection:",
                  font=("Arial", 12, "bold")).pack(anchor=tk.W)

        conn_frame = ttk.Frame(top_frame)
        conn_frame.pack(fill=tk.X, pady=5)

        self.conn_status_label = ttk.Label(conn_frame, text=f"Connecting to {self.host}:{self.port}...",
                                           font=("Arial", 10))
        self.conn_status_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(conn_frame, text="Reconnect",
                   command=self.connect_to_server).pack(side=tk.LEFT)

        # Status label
        self.status_label = ttk.Label(top_frame, text="Waiting for connection...",
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

        ttk.Button(control_frame, text="Reset Stats",
                   command=self.reset_stats).pack(side=tk.LEFT, padx=5)

        # Stats frame
        stats_frame = ttk.LabelFrame(
            self.root, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        self.stats_label = ttk.Label(stats_frame, text="", font=("Arial", 10))
        self.stats_label.pack()
        self.update_stats_display()

    def connect_to_server(self):
        """Connect to the environment server."""
        # Don't reconnect if already connecting
        if self.connection_thread and self.connection_thread.is_alive():
            return

        # Clean up old connection if exists
        self.disconnect()

        # Reset game state
        self.board = np.zeros(9, dtype=np.int32)
        self.current_turn = None
        self.my_player = None
        self.game_active = False
        self.episode = 0
        self.waiting_for_next_game = False
        self.pending_observation = None
        self.update_board()

        self.conn_status_label.config(
            text=f"Connecting to {self.host}:{self.port}...")
        self.status_label.config(text="Connecting to environment server...")

        def connect():
            try:
                address = (self.host, self.port)
                conn = Client(address, authkey=self.authkey)
                self.conn = conn
                self.connected = True
                self.root.after(0, lambda: self.conn_status_label.config(
                    text=f"✓ Connected to {self.host}:{self.port}", foreground="green"))
                self.root.after(0, lambda: self.status_label.config(
                    text="Connected! Waiting for second player and game to start..."))

                # Start message receiving loop
                self.receive_messages()
            except Exception as e:
                self.connected = False
                self.conn = None
                error_msg = f"Failed to connect: {e}"
                error_details = (f"Could not connect to environment server at {self.host}:{self.port}\n\n"
                                 f"Error: {e}\n\n"
                                 f"Make sure you have started the server with:\n"
                                 f"  python run_env_server.py")
                self.root.after(0, lambda: self.conn_status_label.config(
                    text=error_msg, foreground="red"))
                self.root.after(0, lambda: self.status_label.config(
                    text="Connection failed. Make sure the environment server is running."))
                self.root.after(0, lambda msg=error_details: messagebox.showerror(
                    "Connection Error", msg))

        self.connection_thread = threading.Thread(target=connect, daemon=True)
        self.connection_thread.start()

    def disconnect(self):
        """Disconnect from server and clean up resources."""
        self.connected = False
        self.running = False

        # Close connection
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None

        # Wait a bit for threads to finish (they're daemon threads so will terminate)
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=0.5)

        if self.message_thread and self.message_thread.is_alive():
            self.message_thread.join(timeout=0.5)

        # Reset running flag for next connection
        self.running = True

    def receive_messages(self):
        """Receive and process messages from the server in a background thread."""
        def message_loop():
            while self.running and self.connected and self.conn:
                try:
                    if self.conn.poll(0.1):  # Poll with timeout
                        msg = self.conn.recv()
                        if not isinstance(msg, dict):
                            continue

                        mtype = msg.get("type")
                        if mtype == OBSERVATION:
                            # Debug: print received observation
                            info = msg.get("info", {})
                            print(
                                f"[GUI] Received OBSERVATION: player={info.get('player')}, current_turn={info.get('current_turn')}, episode={info.get('episode')}")
                            self.root.after(
                                0, lambda m=msg: self.handle_observation(m))
                        elif mtype == REWARD:
                            print(f"[GUI] Received REWARD: {msg.get('value')}")
                            self.root.after(
                                0, lambda m=msg: self.handle_reward(m))
                        elif mtype == TERMINAL:
                            print(
                                f"[GUI] Received TERMINAL: winner={msg.get('info', {}).get('winner')}")
                            self.root.after(
                                0, lambda m=msg: self.handle_terminal(m))
                        elif mtype == SHUTDOWN:
                            print("[GUI] Received SHUTDOWN")
                            self.root.after(0, lambda: self.handle_shutdown())
                            break
                    else:
                        time.sleep(0.01)  # Small sleep to avoid busy-waiting
                except (EOFError, OSError) as e:
                    if self.connected:  # Only update if we were connected
                        self.connected = False
                        print(f"[GUI] Connection error: {e}")
                        self.root.after(0, lambda: self.conn_status_label.config(
                            text=f"Connection lost: {e}", foreground="red"))
                        self.root.after(0, lambda: self.status_label.config(
                            text="Connection lost. Click Reconnect to retry."))
                    break
                except Exception as e:
                    if self.connected:  # Only log if we were connected
                        print(f"[GUI] Error receiving message: {e}")
                        import traceback
                        traceback.print_exc()
                    time.sleep(0.1)

        self.message_thread = threading.Thread(
            target=message_loop, daemon=True)
        self.message_thread.start()

    def handle_observation(self, msg):
        """Handle observation message from server."""
        # If we're waiting for next game (post-game delay), store the observation
        if self.waiting_for_next_game:
            self.pending_observation = msg
            return

        sensors = msg.get("sensors", [])
        info = msg.get("info", {})

        if len(sensors) == 9:
            # Update board state
            self.board = np.array(sensors, dtype=np.int32)

            # Determine which player we are (first observation tells us)
            player_just_set = False
            if self.my_player is None:
                player = info.get("player")
                if player:
                    self.my_player = player
                    player_just_set = True
                    print(f"[GUI] Determined I am player {self.my_player}")

            # Get current turn
            self.current_turn = info.get("current_turn")
            self.episode = info.get("episode", 0)

            print(
                f"[GUI] handle_observation: my_player={self.my_player}, current_turn={self.current_turn}, episode={self.episode}")

            # Update game state and status based on current turn
            if self.current_turn == self.my_player:
                self.game_active = True
                self.status_label.config(text=f"Your turn ({self.my_player})")
                print(f"[GUI] Status: Your turn")
            elif self.current_turn and self.current_turn != self.my_player:
                # It's the opponent's turn
                self.game_active = True
                opponent = 'O' if self.my_player == 'X' else 'X'
                self.status_label.config(text=f"Brain's turn ({opponent})...")
                print(f"[GUI] Status: Brain's turn")
            else:
                # No turn info or turn is None
                self.game_active = False  # Don't allow moves if we don't know whose turn it is
                if player_just_set:
                    opponent = 'O' if self.my_player == 'X' else 'X'
                    self.status_label.config(
                        text=f"You are {self.my_player}, Brain is {opponent}. Waiting for game to start...")
                    print(
                        f"[GUI] Status: Waiting for game to start (no current_turn info)")
                else:
                    # Keep current status if we already know our player
                    print(f"[GUI] Status: No turn info, keeping current status")

            # Update display AFTER setting game_active
            self.update_board()

    def handle_reward(self, msg):
        """Handle reward message (usually at end of game)."""
        # Rewards are handled in handle_terminal
        pass

    def handle_terminal(self, msg):
        """Handle terminal message (game ended)."""
        info = msg.get("info", {})
        winner = info.get("winner")

        self.game_active = False
        self.stats['games_played'] += 1

        # Show game result message
        if winner == self.my_player:
            self.stats['human_wins'] += 1
            result_msg = f"Game Over: You win! 🎉 (Episode {self.episode})"
        elif winner and winner != 'draw':
            self.stats['agent_wins'] += 1
            result_msg = f"Game Over: Brain wins! 🤖 (Episode {self.episode})"
        else:  # draw
            self.stats['draws'] += 1
            result_msg = f"Game Over: It's a draw! 🤝 (Episode {self.episode})"

        self.status_label.config(text=result_msg)
        self.update_stats_display()

        # Set flag to delay next game processing
        self.waiting_for_next_game = True

        # Clear the flag after 2 seconds and process any pending observation
        def clear_delay_and_process():
            self.waiting_for_next_game = False
            if self.pending_observation:
                obs = self.pending_observation
                self.pending_observation = None
                self.handle_observation(obs)

        self.root.after(2000, clear_delay_and_process)

    def handle_shutdown(self):
        """Handle shutdown message from server."""
        self.connected = False
        self.conn_status_label.config(
            text="Server shutdown", foreground="orange")
        self.status_label.config(
            text="Server has shut down. Click Reconnect when server is back.")

    def update_board(self):
        """Update the visual board based on current board state."""
        board_2d = self.board.reshape(3, 3)

        # The observation is from our perspective:
        # - If we're X: X=1, O=-1, empty=0
        # - If we're O: O=1, X=-1, empty=0 (flipped)
        # For display, we want: X always = blue, O always = red

        for i in range(3):
            for j in range(3):
                val = board_2d[i, j]
                btn = self.buttons[i][j]

                # Convert observation value to display
                if val == 0:
                    display_symbol = ''
                    display_color = 'white'
                    text_color = 'black'
                elif self.my_player == 'X':
                    # We're X: 1=X (blue), -1=O (red)
                    display_symbol = 'X' if val == 1 else 'O'
                    display_color = 'blue' if val == 1 else 'red'
                    text_color = 'white'
                else:  # self.my_player == 'O'
                    # We're O: 1=O (red), -1=X (blue)
                    display_symbol = 'O' if val == 1 else 'X'
                    display_color = 'red' if val == 1 else 'blue'
                    text_color = 'white'

                # Determine if this cell is clickable
                is_clickable = (val == 0 and self.game_active and
                                self.connected and
                                self.current_turn == self.my_player)

                # Debug: log clickability for first empty cell
                if val == 0 and i == 0 and j == 0:
                    print(f"[GUI] update_board: cell (0,0) clickable={is_clickable}, "
                          f"val={val}, game_active={self.game_active}, "
                          f"connected={self.connected}, current_turn={self.current_turn}, "
                          f"my_player={self.my_player}")

                btn.config(text=display_symbol,
                           state=tk.NORMAL if is_clickable else tk.DISABLED,
                           bg=display_color,
                           fg=text_color)

    def on_cell_click(self, row, col):
        """Handle cell click - send action to server."""
        if not self.game_active or not self.connected or not self.conn:
            return

        if self.current_turn != self.my_player:
            messagebox.showwarning(
                "Not Your Turn", "Wait for the brain to make its move!")
            return

        action = row * 3 + col

        # Check if move is valid locally (basic check)
        if self.board[action] != 0:
            messagebox.showwarning(
                "Invalid Move", "This cell is already occupied!")
            return

        # Send action to server
        try:
            self.conn.send({
                "type": ACTION,
                "actions": [int(action)],
            })
            # Disable board temporarily while waiting for response
            self.game_active = False
            self.status_label.config(text="Move sent, waiting for response...")
        except (EOFError, OSError) as e:
            self.connected = False
            self.conn_status_label.config(
                text=f"Connection lost: {e}", foreground="red")
            messagebox.showerror("Connection Error",
                                 f"Lost connection to server: {e}")

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
                          f"Brain: {self.stats['agent_wins']} ({agent_pct:.1f}%) | "
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

    def cleanup(self):
        """Clean up resources on exit."""
        self.disconnect()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Tic Tac Toe GUI - Play against a brain connected to environment server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to default server (localhost:6000)
  python tic_tac_toe_gui.py
  
  # Connect to custom server
  python tic_tac_toe_gui.py --host 192.168.1.100 --port 7000
  
  # Custom authkey
  python tic_tac_toe_gui.py --authkey my-secret-key

Note: Make sure the environment server is running:
  python run_env_server.py

And a brain client is connected:
  python run_brain_client.py --load brain.pkl
        """)

    parser.add_argument('--host', type=str, default='localhost',
                        help='Environment server host (default: localhost)')
    parser.add_argument('--port', type=int, default=6000,
                        help='Environment server port (default: 6000)')
    parser.add_argument('--authkey', type=str, default='brain-secret',
                        help='Connection authkey (default: brain-secret)')

    args = parser.parse_args()

    root = tk.Tk()
    app = TicTacToeGUI(root, host=args.host,
                       port=args.port, authkey=args.authkey)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
