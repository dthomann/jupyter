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
    
    def __init__(self, agent_symbol='X', opponent_symbol='O', rng=None, training_mode=False):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.agent_symbol = agent_symbol
        self.opponent_symbol = opponent_symbol
        self.training_mode = training_mode
        self.board = None
        self.current_player = None
        self.done = False
        self.winner = None
        
    def reset(self, player_symbol=None):
        """
        Reset the board to empty state.
        Returns observation from specified player's perspective (default: agent_symbol).
        """
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = 'X'  # Agent goes first
        self.done = False
        self.winner = None
        return self._get_obs(player_symbol)
    
    def _get_obs(self, player_symbol=None):
        """
        Get observation from a specific player's perspective.
        If player_symbol is None, uses agent_symbol (default X perspective).
        - For X: X=1, O=-1, empty=0
        - For O: O=1, X=-1, empty=0
        """
        obs = self.board.copy()
        if player_symbol == 'O':
            # Flip perspective: O sees themselves as 1, X as -1
            obs = -obs
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
    
    def get_valid_actions(self, current_player_symbol=None):
        """
        Get list of valid actions.
        
        In training mode, filters actions to:
        - If current player can win (two in a row, third free): return only winning action
        - Else if opponent can win (two in a row, third free): return only blocking action
        - Otherwise: return all valid actions
        
        Args:
            current_player_symbol: 'X' or 'O' - whose turn it is. Required in training mode.
        """
        valid_actions = [i for i in range(9) if self.board[i] == 0]
        
        if not self.training_mode or current_player_symbol is None:
            return valid_actions
        
        # Determine player values: X=1, O=-1
        current_player = 1 if current_player_symbol == 'X' else -1
        opponent = -current_player
        
        # First check: can current player win immediately?
        winning_actions = self._find_winning_actions(current_player)
        if winning_actions:
            return winning_actions
        
        # Second check: must block opponent's immediate win?
        blocking_actions = self._find_blocking_actions(opponent)
        if blocking_actions:
            return blocking_actions
        
        # Otherwise, return all valid actions
        return valid_actions
    
    def _find_winning_actions(self, player):
        """
        Find actions that would result in an immediate win for the given player.
        Returns list of action indices, or empty list if no immediate win possible.
        """
        winning_actions = []
        board_2d = self.board.reshape(3, 3)
        
        # Check rows
        for i in range(3):
            row = board_2d[i]
            if (row == player).sum() == 2 and (row == 0).sum() == 1:
                # Find the empty spot in this row
                for j in range(3):
                    if row[j] == 0:
                        action = i * 3 + j
                        if action not in winning_actions:
                            winning_actions.append(action)
        
        # Check columns
        for j in range(3):
            col = board_2d[:, j]
            if (col == player).sum() == 2 and (col == 0).sum() == 1:
                # Find the empty spot in this column
                for i in range(3):
                    if col[i] == 0:
                        action = i * 3 + j
                        if action not in winning_actions:
                            winning_actions.append(action)
        
        # Check diagonal 1: [0, 4, 8]
        diag1_indices = [0, 4, 8]
        diag1_values = [self.board[i] for i in diag1_indices]
        if diag1_values.count(player) == 2 and diag1_values.count(0) == 1:
            # Find the empty spot
            for idx in diag1_indices:
                if self.board[idx] == 0:
                    if idx not in winning_actions:
                        winning_actions.append(idx)
        
        # Check diagonal 2: [2, 4, 6]
        diag2_indices = [2, 4, 6]
        diag2_values = [self.board[i] for i in diag2_indices]
        if diag2_values.count(player) == 2 and diag2_values.count(0) == 1:
            # Find the empty spot
            for idx in diag2_indices:
                if self.board[idx] == 0:
                    if idx not in winning_actions:
                        winning_actions.append(idx)
        
        return winning_actions
    
    def _find_blocking_actions(self, opponent):
        """
        Find actions that would block an immediate win by the opponent.
        Returns list of action indices, or empty list if no immediate threat.
        """
        # This is the same as finding winning actions for the opponent
        return self._find_winning_actions(opponent)
    
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
    require_two_brains: bool = True,
    training_mode: bool = True,
):
    """
    Environment server that listens for brain client connections.
    
    If require_two_brains=True (default), waits for two brain connections
    and runs matches where brains play against each other.
    
    If require_two_brains=False, runs self-play with a single brain connection
    (backward compatible with old behavior).
    
    If training_mode=True (default), filters legal actions to force immediate
    wins and blocks when available.
    """
    address = (host, port)
    listener = Listener(address, authkey=authkey)
    print(f"[env] listening on {address}")
    if require_two_brains:
        print("[env] mode: two-brain matches (brains play against each other)")
    else:
        print("[env] mode: self-play (single brain plays against itself)")

    while True:
        if require_two_brains:
            # Wait for two brain connections
            print("[env] waiting for first brain (player X)...")
            conn_x = listener.accept()
            print(f"[env] player X connected from {listener.last_accepted}")
            
            print("[env] waiting for second brain (player O)...")
            conn_o = listener.accept()
            print(f"[env] player O connected from {listener.last_accepted}")
            print("[env] both players connected, starting matches...")
            
            try:
                episode = 0
                # Initialize random number generator for random starting player
                rng = np.random.RandomState()
                while True:
                    # Start new episode
                    episode += 1
                    env = TicTacToeEnv(training_mode=training_mode)
                    env.reset()
                    done = False
                    # Randomly choose which player starts (X or O)
                    current_player_symbol = 'X' if rng.random() < 0.5 else 'O'
                    
                    # Log every 100 episodes
                    if episode % 100 == 0:
                        print(f"[env] {episode} games played", end='', flush=True)
                    
                    # Send initial observations to both players
                    # The brain client will only act when it's their turn
                    legal_actions = env.get_valid_actions(current_player_symbol)
                    legal_mask = [0.0 if i in legal_actions else float('-inf') for i in range(9)]
                    
                    try:
                        # Send to X player
                        obs_x = env._get_obs('X')
                        conn_x.send({
                            "type": OBSERVATION,
                            "sensors": obs_x.tolist(),
                            "info": {"t": 0.0, "episode": episode, "player": "X", "current_turn": current_player_symbol, "legal_actions": legal_mask},
                        })
                        
                        # Send to O player
                        obs_o = env._get_obs('O')
                        conn_o.send({
                            "type": OBSERVATION,
                            "sensors": obs_o.tolist(),
                            "info": {"t": 0.0, "episode": episode, "player": "O", "current_turn": current_player_symbol, "legal_actions": legal_mask},
                        })
                    except (EOFError, OSError) as e:
                        print(f"[env] brain disconnected while sending initial observation: {e}")
                        break
                    
                    while not done:
                        # Determine which connection should act
                        conn_current = conn_x if current_player_symbol == 'X' else conn_o
                        player_name = current_player_symbol
                        
                        # Wait for ACTION message from current player
                        # Use blocking receive since we're waiting for the current player's turn
                        try:
                            msg = conn_current.recv()
                            if msg.get("type") != ACTION:
                                print(f"[env] unexpected message type from {player_name}: {msg.get('type')}")
                                continue
                            
                            actions = msg.get("actions", [])
                            if not actions:
                                print(f"[env] no actions in message from {player_name}")
                                continue
                            
                            action = int(actions[0])
                        except (EOFError, OSError) as e:
                            print(f"\n[env] {player_name} disconnected while receiving action: {e}")
                            raise  # Re-raise to trigger reconnection handling
                        
                        # Discard any pending messages from the other player (they may have sent actions out of turn)
                        other_conn = conn_o if current_player_symbol == 'X' else conn_x
                        while other_conn.poll(0.0):
                            try:
                                other_msg = other_conn.recv()
                                # Ignore out-of-turn actions - they'll get updated observation
                            except (EOFError, OSError):
                                break
                            except Exception:
                                break
                        
                        # Apply action to environment
                        if current_player_symbol == 'X':
                            obs, reward, done, info = env.step(action)
                        else:
                            obs, reward, done, info = env.make_opponent_move(action)
                        
                        # Determine rewards for both players
                        if done:
                            if env.winner == 'X':
                                reward_x = 1.0
                                reward_o = -1.0
                            elif env.winner == 'O':
                                reward_x = -1.0
                                reward_o = 1.0
                            else:  # draw
                                reward_x = 0.0
                                reward_o = 0.0
                        else:
                            reward_x = 0.0
                            reward_o = 0.0
                        
                        # Send updated observations to both players
                        # Determine next player (will switch after this move)
                        next_player = 'O' if current_player_symbol == 'X' else 'X'
                        legal_actions = env.get_valid_actions(next_player)
                        legal_mask = [0.0 if i in legal_actions else float('-inf') for i in range(9)]
                        
                        try:
                            
                            # Send to X player
                            obs_x = env._get_obs('X')
                            conn_x.send({
                                "type": OBSERVATION,
                                "sensors": obs_x.tolist(),
                                "info": {"t": info.get("t", 0.0), "episode": episode, "player": "X", "current_turn": next_player, "legal_actions": legal_mask},
                            })
                            
                            # Send to O player
                            obs_o = env._get_obs('O')
                            conn_o.send({
                                "type": OBSERVATION,
                                "sensors": obs_o.tolist(),
                                "info": {"t": info.get("t", 0.0), "episode": episode, "player": "O", "current_turn": next_player, "legal_actions": legal_mask},
                            })
                            
                            # Send rewards when game ends
                            if done:
                                conn_x.send({
                                    "type": REWARD,
                                    "value": float(reward_x),
                                    "info": {"t": info.get("t", 0.0), "episode": episode, "player": "X"},
                                })
                                conn_o.send({
                                    "type": REWARD,
                                    "value": float(reward_o),
                                    "info": {"t": info.get("t", 0.0), "episode": episode, "player": "O"},
                                })
                        except (EOFError, OSError) as e:
                            print(f"\n[env] brain disconnected while sending obs/reward: {e}")
                            raise  # Re-raise to trigger reconnection handling
                        
                        if done:
                            # Send TERMINAL message to both players
                            try:
                                conn_x.send({
                                    "type": TERMINAL,
                                    "info": {"t": info.get("t", 0.0), "episode": episode, "winner": env.winner},
                                })
                                conn_o.send({
                                    "type": TERMINAL,
                                    "info": {"t": info.get("t", 0.0), "episode": episode, "winner": env.winner},
                                })
                            except (EOFError, OSError) as e:
                                print(f"\n[env] brain disconnected while sending terminal: {e}")
                                raise  # Re-raise to trigger reconnection handling
                            break
                        
                        # Switch players
                        current_player_symbol = 'O' if current_player_symbol == 'X' else 'X'
            
            except (EOFError, OSError) as e:
                print(f"\n[env] brain disconnected: {e}")
                print("[env] closing connections and waiting for reconnection...")
            except KeyboardInterrupt:
                print("\n[env] interrupted by user")
                try:
                    conn_x.send({"type": SHUTDOWN})
                    conn_o.send({"type": SHUTDOWN})
                except:
                    pass
                break
            finally:
                try:
                    conn_x.close()
                    conn_o.close()
                except Exception:
                    pass
                # Don't break - loop back to wait for reconnection
                print("[env] waiting for reconnection...")
        
        else:
            # Original self-play mode (single connection)
            print("[env] waiting for brain connection")
            conn = listener.accept()
            print(f"[env] brain connected from {listener.last_accepted}")

            try:
                episode = 0
                while True:
                    # Start new episode
                    episode += 1
                    env = TicTacToeEnv(training_mode=training_mode)
                    x = env.reset()
                    done = False
                    current_player = 1  # X starts

                    # Send initial observation
                    try:
                        current_player_symbol = 'X' if current_player == 1 else 'O'
                        legal_actions = env.get_valid_actions(current_player_symbol)
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
                            obs, reward, done, info = env.step(action)
                        else:
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
                            # Determine next player for legal actions
                            next_player = -current_player
                            next_player_symbol = 'X' if next_player == 1 else 'O'
                            legal_actions = env.get_valid_actions(next_player_symbol)
                            legal_mask = [0.0 if i in legal_actions else float('-inf') for i in range(9)]
                            conn.send({
                                "type": OBSERVATION,
                                "sensors": obs.tolist(),
                                "info": {"t": info.get("t", 0.0), "episode": episode, "player": current_player, "legal_actions": legal_mask},
                            })
                            
                            if done:
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
                            try:
                                conn.send({
                                    "type": TERMINAL,
                                    "info": {"t": info.get("t", 0.0), "episode": episode, "winner": env.winner},
                                })
                            except (EOFError, OSError):
                                print("[env] brain disconnected while sending terminal")
                                break
                            break

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

