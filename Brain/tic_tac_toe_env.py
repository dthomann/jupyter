import numpy as np
import time
from typing import Optional, Dict, Any, Tuple
from multiprocessing.connection import Listener
from brainprotocol import OBSERVATION, REWARD, ACTION, TERMINAL, SHUTDOWN


class TicTacToeEnv:
    """
    Tic Tac Toe environment compatible with BrainAgent.
    
    Observation: 3x3 board flattened to 9 values
    - -1: O (opponent)
    - 0: empty
    - 1: X (agent)
    
    Actions: 0-8 representing positions:
    0 1 2
    3 4 5
    6 7 8
    
    Reward:
    - +10 for win
    - -10 for loss
    - +1 for draw
    - -1 for invalid move
    - 0 otherwise
    """
    
    def __init__(self, agent_symbol='X', opponent_symbol='O', rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.agent_symbol = agent_symbol
        self.opponent_symbol = opponent_symbol
        self.board = None
        self.current_player = None
        self.done = False
        self.winner = None
        
    def reset(self):
        """Reset the board to empty state."""
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = 'X'  # Agent goes first
        self.done = False
        self.winner = None
        return self._get_obs()
    
    def _get_obs(self):
        """Get observation from agent's perspective."""
        # Convert to agent's perspective: X=1, O=-1, empty=0
        obs = self.board.copy()
        return obs
    
    def _check_winner(self):
        """Check if there's a winner. Returns 'X', 'O', 'draw', or None."""
        board_2d = self.board.reshape(3, 3)
        
        # Check rows, columns, diagonals
        for i in range(3):
            # Rows
            if abs(board_2d[i].sum()) == 3:
                return 'X' if board_2d[i].sum() > 0 else 'O'
            # Columns
            if abs(board_2d[:, i].sum()) == 3:
                return 'X' if board_2d[:, i].sum() > 0 else 'O'
        
        # Diagonals
        diag1 = board_2d[0, 0] + board_2d[1, 1] + board_2d[2, 2]
        diag2 = board_2d[0, 2] + board_2d[1, 1] + board_2d[2, 0]
        if abs(diag1) == 3:
            return 'X' if diag1 > 0 else 'O'
        if abs(diag2) == 3:
            return 'X' if diag2 > 0 else 'O'
        
        # Check for draw
        if (self.board != 0).all():
            return 'draw'
        
        return None
    
    def _is_valid_move(self, action):
        """Check if action is valid."""
        if not isinstance(action, (int, np.integer)):
            return False
        if action < 0 or action >= 9:
            return False
        return self.board[action] == 0
    
    def get_valid_actions(self):
        """Get list of valid actions."""
        return [i for i in range(9) if self.board[i] == 0]
    
    def step(self, action):
        """
        Execute action from agent's perspective.
        Returns: (obs, reward, done, info)
        """
        if self.done:
            return self._get_obs(), 0, True, {'message': 'Game already finished'}
        
        # Agent's turn
        if not self._is_valid_move(action):
            return self._get_obs(), -1.0, False, {'message': 'Invalid move'}
        
        # Place agent's move
        self.board[action] = 1  # X = 1
        
        # Check for winner after agent's move
        winner = self._check_winner()
        if winner == 'X':
            self.done = True
            self.winner = 'X'
            return self._get_obs(), 1.0, True, {'winner': 'X', 'message': 'Agent wins!'}
        elif winner == 'draw':
            self.done = True
            self.winner = 'draw'
            return self._get_obs(), 0.0, True, {'winner': 'draw', 'message': 'Draw'}
        
        # Game continues - opponent will move externally
        return self._get_obs(), 0.0, False, {'message': 'Move accepted'}
    
    def make_opponent_move(self, action):
        """
        Make opponent's move (for human or other player).
        Returns: (obs, reward, done, info)
        """
        if self.done:
            return self._get_obs(), 0, True, {'message': 'Game already finished'}
        
        if not self._is_valid_move(action):
            return self._get_obs(), 0, False, {'message': 'Invalid move'}
        
        # Place opponent's move
        self.board[action] = -1  # O = -1
        
        # Check for winner after opponent's move
        winner = self._check_winner()
        if winner == 'O':
            self.done = True
            self.winner = 'O'
            return self._get_obs(), -1.0, True, {'winner': 'O', 'message': 'Opponent wins!'}
        elif winner == 'draw':
            self.done = True
            self.winner = 'draw'
            return self._get_obs(), 0.0, True, {'winner': 'draw', 'message': 'Draw'}
        
        return self._get_obs(), 0.0, False, {'message': 'Opponent move accepted'}
    
    def _blocks_opponent_threat(self, action, player):
        """Check if the move blocks an opponent threat (two in a row)."""
        opponent = -player
        board_2d = self.board.reshape(3, 3)
        
        # Check if opponent had two in a row that this move blocks
        # Check rows
        for i in range(3):
            row = board_2d[i]
            if (row == opponent).sum() == 2 and (row == player).sum() == 1:
                if action in [i*3 + j for j in range(3)]:
                    return True
        # Check columns
        for j in range(3):
            col = board_2d[:, j]
            if (col == opponent).sum() == 2 and (col == player).sum() == 1:
                if action in [i*3 + j for i in range(3)]:
                    return True
        # Check diagonals
        diag1_indices = [0, 4, 8]
        diag1 = [self.board[i] for i in diag1_indices]
        if action in diag1_indices and diag1.count(opponent) == 2 and diag1.count(player) == 1:
            return True
        diag2_indices = [2, 4, 6]
        diag2 = [self.board[i] for i in diag2_indices]
        if action in diag2_indices and diag2.count(opponent) == 2 and diag2.count(player) == 1:
            return True
        return False
    
    def _creates_threat(self, action, player):
        """Check if the move creates a threat (two in a row with empty third)."""
        board_2d = self.board.reshape(3, 3)
        
        # Check if this move creates two in a row
        # Check rows
        for i in range(3):
            row = board_2d[i]
            if (row == player).sum() == 2 and (row == 0).sum() == 1:
                if action in [i*3 + j for j in range(3)]:
                    return True
        # Check columns
        for j in range(3):
            col = board_2d[:, j]
            if (col == player).sum() == 2 and (col == 0).sum() == 1:
                if action in [i*3 + j for i in range(3)]:
                    return True
        # Check diagonals
        diag1_indices = [0, 4, 8]
        diag1 = [self.board[i] for i in diag1_indices]
        if action in diag1_indices and diag1.count(player) == 2 and diag1.count(0) == 1:
            return True
        diag2_indices = [2, 4, 6]
        diag2 = [self.board[i] for i in diag2_indices]
        if action in diag2_indices and diag2.count(player) == 2 and diag2.count(0) == 1:
            return True
        return False
    
    def render(self):
        """Print the current board state."""
        board_2d = self.board.reshape(3, 3)
        symbols = {1: 'X', -1: 'O', 0: ' '}
        
        print("\n  0   1   2")
        for i in range(3):
            row_str = f"{i} "
            for j in range(3):
                val = board_2d[i, j]
                row_str += f" {symbols[val]} "
                if j < 2:
                    row_str += "|"
            print(row_str)
            if i < 2:
                print("  -----------")
        print()


def run_env_server(
    host: str = "localhost",
    port: int = 6000,
    authkey: bytes = b"brain-secret",
    env_dt: float = 0.05,
):
    """
    Environment server that listens for brain client connections.
    Runs tic-tac-toe episodes with self-play (both players use the same brain).
    """
    address = (host, port)
    listener = Listener(address, authkey=authkey)
    print(f"[env] listening on {address}")

    while True:
        print("[env] waiting for brain connection")
        conn = listener.accept()
        print(f"[env] brain connected from {listener.last_accepted}")

        try:
            episode = 0
            while True:
                # Start new episode
                episode += 1
                env = TicTacToeEnv()
                x = env.reset()
                done = False
                current_player = 1  # X starts

                # Episode started (logging removed for cleaner output)

                # Send initial observation
                try:
                    legal_actions = env.get_valid_actions()
                    # Create legal mask: 0 for legal, -inf for illegal
                    legal_mask = [0.0 if i in legal_actions else float('-inf') for i in range(9)]
                    conn.send({
                        "type": OBSERVATION,
                        "sensors": x.tolist(),
                        "info": {"t": 0.0, "episode": episode, "legal_actions": legal_mask},
                    })
                except (EOFError, OSError) as e:
                    print(f"[env] brain disconnected while sending initial observation: {e}")
                    break
                except Exception as e:
                    print(f"[env] Error sending initial observation: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                while not done:
                    # Wait for ACTION message from brain
                    try:
                        msg = conn.recv()
                        if msg.get("type") != ACTION:
                            print(f"[env] unexpected message type: {msg.get('type')}")
                            continue
                        
                        actions = msg.get("actions", [])
                        if not actions:
                            print("[env] no actions in message")
                            continue
                        
                        action = int(actions[0])
                    except (EOFError, OSError) as e:
                        print(f"[env] brain disconnected while receiving action: {e}")
                        done = True
                        break

                    # Apply action to environment
                    if current_player == 1:
                        # X's turn (agent)
                        obs, reward, done, info = env.step(action)
                    else:
                        # O's turn (opponent, also from brain)
                        obs, reward, done, info = env.make_opponent_move(action)
                    
                    # Determine reward based on outcome
                    if done:
                        if env.winner == 'X':
                            final_reward = 1.0
                        elif env.winner == 'O':
                            final_reward = -1.0
                        else:
                            final_reward = 0.0
                    else:
                        final_reward = 0.0

                    try:
                        # Send updated observation after each action
                        legal_actions = env.get_valid_actions()
                        # Create legal mask: 0 for legal, -inf for illegal
                        legal_mask = [0.0 if i in legal_actions else float('-inf') for i in range(9)]
                        conn.send({
                            "type": OBSERVATION,
                            "sensors": obs.tolist(),
                            "info": {"t": info.get("t", 0.0), "episode": episode, "player": current_player, "legal_actions": legal_mask},
                        })
                        
                        # Only send reward when game ends
                        if done:
                            # Send reward (from current player's perspective)
                            # For self-play: player 1 gets +1 for X win, -1 for O win
                            # player -1 gets -1 for X win, +1 for O win
                            player_reward = final_reward * current_player
                            conn.send({
                                "type": REWARD,
                                "value": float(player_reward),
                                "info": {"t": info.get("t", 0.0), "episode": episode, "player": current_player},
                            })
                    except (EOFError, OSError) as e:
                        print(f"[env] brain disconnected while sending obs/reward: {e}")
                        done = True
                        break

                    if done:
                        # Send TERMINAL message when game ends
                        try:
                            conn.send({
                                "type": TERMINAL,
                                "info": {"t": info.get("t", 0.0), "episode": episode, "winner": env.winner},
                            })
                        except (EOFError, OSError):
                            print("[env] brain disconnected while sending terminal")
                            break
                        break

                    # Switch players
                    current_player = -current_player

        except (EOFError, OSError) as e:
            print(f"[env] brain disconnected unexpectedly: {e}")
        except KeyboardInterrupt:
            print("[env] interrupted by user")
            try:
                conn.send({"type": SHUTDOWN})
            except:
                pass
            break
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print("[env] connection closed, waiting for next brain")

    listener.close()

