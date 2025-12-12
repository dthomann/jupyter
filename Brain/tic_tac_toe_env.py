import numpy as np
import time
import select
import threading
from typing import Optional, Dict, Any, Tuple
from multiprocessing.connection import Listener
from brain.connection_manager import ConnectionManager
from brain.connection_config import BrainConnectionConfig
from brainprotocol import (
    OBSERVATION, REWARD, ACTION, TERMINAL, SHUTDOWN,
    DISCOVERY_STARTUP, DISCOVERY_SHUTDOWN, DISCOVERY_ANNOUNCE,
    PEER_TYPE_ENVIRONMENT, PEER_TYPE_BRAIN
)


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
    env_id = f"env_{host}_{port}"

    # Create ConnectionManager for multicast discovery
    connection_config = BrainConnectionConfig(
        brain_id=env_id,
        listen_address=address,
        enable_listener=True,
        authkey=authkey,
    )
    connection_manager = ConnectionManager(
        connection_config,
        peer_type=PEER_TYPE_ENVIRONMENT,
        max_brains=2 if require_two_brains else 1,
    )

    print(f"[env] Environment {env_id} listening on {address}")
    if require_two_brains:
        print("[env] mode: two-brain matches (brains play against each other)")
    else:
        print("[env] mode: self-play (single brain plays against itself)")

    # Track which brain is X and which is O
    brain_x_id = None
    brain_o_id = None
    # Track if we've already logged waiting messages
    logged_waiting_x = False
    logged_waiting_o = False

    def get_brain_connection(brain_id):
        """Get connection for a brain by ID."""
        return connection_manager.connections.get(brain_id)

    def assign_brains():
        """Assign first two connected brains as X and O."""
        nonlocal brain_x_id, brain_o_id, logged_waiting_x, logged_waiting_o

        connected_brains = [
            peer_id for peer_id, metadata in connection_manager.connection_metadata.items()
            if (peer_id in connection_manager.connections and
                metadata.get("peer_type") == PEER_TYPE_BRAIN)
        ]

        if len(connected_brains) >= 1 and brain_x_id is None:
            brain_x_id = connected_brains[0]
            print(f"[env] Assigned brain {brain_x_id} as player X")
            logged_waiting_x = False  # Reset flag when assigned

        if len(connected_brains) >= 2 and brain_o_id is None:
            # Find a different brain for O
            for brain_id in connected_brains:
                if brain_id != brain_x_id:
                    brain_o_id = brain_id
                    print(f"[env] Assigned brain {brain_o_id} as player O")
                    logged_waiting_o = False  # Reset flag when assigned
                    break

    try:
        # Give connections a moment to establish
        time.sleep(0.5)

        while True:
            if require_two_brains:
                # Process incoming messages (including startup messages from brains)
                # This is needed so we can identify which connections are brains
                events = connection_manager.poll_events()
                for peer_id, msg in events:
                    if isinstance(msg, dict):
                        mtype = msg.get("type")
                        # Handle discovery messages (just log, no relaying)
                        if mtype in (DISCOVERY_STARTUP, DISCOVERY_SHUTDOWN):
                            sender_peer_id = msg.get("peer_id", "unknown")
                            sender_peer_type = msg.get("peer_type", "unknown")
                            print(
                                f"[env] Received {mtype} from {sender_peer_id} (type: {sender_peer_type})")
                            # Startup messages update peer_type, which helps assign_brains() identify brains
                            continue

                # Environments do NOT proactively connect to brains
                # Brains will connect to the environment after discovering it via multicast
                # We only accept incoming connections from brains

                assign_brains()

                if brain_x_id is None:
                    if not logged_waiting_x:
                        print("[env] waiting for first brain (player X)...")
                        logged_waiting_x = True
                    time.sleep(0.1)
                    continue

                if brain_o_id is None:
                    if not logged_waiting_o:
                        print("[env] waiting for second brain (player O)...")
                        logged_waiting_o = True
                    time.sleep(0.1)
                    continue

                conn_x = get_brain_connection(brain_x_id)
                conn_o = get_brain_connection(brain_o_id)

                if conn_x is None or conn_o is None:
                    # One of the connections was lost, reassign
                    if conn_x is None:
                        print(f"[env] Brain X ({brain_x_id}) disconnected")
                        brain_x_id = None
                    if conn_o is None:
                        print(f"[env] Brain O ({brain_o_id}) disconnected")
                        brain_o_id = None
                    time.sleep(0.1)
                    continue

                if conn_x is not None and conn_o is not None:
                    print("[env] both players connected, starting matches...")
                    # Start the game loop
                    try:
                        print("[env] Entering game loop...")
                        episode = 0
                        # Initialize random number generator for random starting player
                        rng = np.random.RandomState()
                        while True:
                            # Start new episode
                            episode += 1
                            print(f"[env] Starting episode {episode}")
                            env = TicTacToeEnv(training_mode=training_mode)
                            env.reset()
                            done = False
                            # Randomly choose which player starts (X or O)
                            current_player_symbol = 'X' if rng.random() < 0.5 else 'O'
                            print(
                                f"[env] Episode {episode}: {current_player_symbol} goes first")

                            # Log every 100 episodes
                            if episode % 100 == 0:
                                print(f"[env] {episode} games played",
                                      end='', flush=True)

                            # Send initial observations to both players
                            # The brain client will only act when it's their turn
                            legal_actions = env.get_valid_actions(
                                current_player_symbol)
                            legal_mask = [0.0 if i in legal_actions else float(
                                '-inf') for i in range(9)]

                            # Send initial observation only to the player who goes first
                            # Verify connections exist before sending
                            if brain_x_id not in connection_manager.connections:
                                print(
                                    f"[env] ERROR: brain_x_id {brain_x_id} not in connections!")
                                raise ConnectionError(
                                    f"Brain X ({brain_x_id}) not connected")
                            if brain_o_id not in connection_manager.connections:
                                print(
                                    f"[env] ERROR: brain_o_id {brain_o_id} not in connections!")
                                raise ConnectionError(
                                    f"Brain O ({brain_o_id}) not connected")

                            # Only send initial observation to the player who goes first
                            if current_player_symbol == 'X':
                                try:
                                    obs_x = env._get_obs('X')
                                    print(
                                        f"[env] Sending initial observation to player X (brain {brain_x_id})")
                                    connection_manager.send(brain_x_id, {
                                        "type": OBSERVATION,
                                        "sensors": obs_x.tolist(),
                                        "info": {"t": 0.0, "episode": episode, "player": "X", "current_turn": current_player_symbol, "legal_actions": legal_mask},
                                    })
                                    print(
                                        f"[env] Initial observation sent to player X")
                                except Exception as e:
                                    print(
                                        f"[env] Player X disconnected while sending initial observation: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise  # Re-raise to trigger reconnection handling
                            else:  # current_player_symbol == 'O'
                                try:
                                    obs_o = env._get_obs('O')
                                    print(
                                        f"[env] Sending initial observation to player O (brain {brain_o_id})")
                                    connection_manager.send(brain_o_id, {
                                        "type": OBSERVATION,
                                        "sensors": obs_o.tolist(),
                                        "info": {"t": 0.0, "episode": episode, "player": "O", "current_turn": current_player_symbol, "legal_actions": legal_mask},
                                    })
                                    print(
                                        f"[env] Initial observation sent to player O")
                                except Exception as e:
                                    print(
                                        f"[env] Player O disconnected while sending initial observation: {e}")
                                    raise  # Re-raise to trigger reconnection handling

                            # Small delay to allow initial observation to be received and any stale actions to arrive
                            # time.sleep(0.2)
                            # Clear any stale messages from previous episode (discard them)
                            # stale_events = connection_manager.poll_events()
                            # Discard any stale actions
                            # for peer_id, msg in stale_events:
                             #   if isinstance(msg, dict) and msg.get("type") == ACTION:
                             #       pass  # Discard stale actions

                            # Wait for actions and process game
                            action_wait_count = 0
                            print(
                                f"[env] Entering game loop for episode {episode}, waiting for actions... (done={done})")
                            while not done:
                                # Debug: check if done changed
                                if action_wait_count == 0:
                                    print(
                                        f"[env] Episode {episode}: Starting action wait loop, done={done}")
                                # Poll for messages from both players
                                events = connection_manager.poll_events()
                                action_received = False

                                # Log periodically if waiting for actions
                                if action_wait_count % 100 == 0 and action_wait_count > 0:
                                    print(
                                        f"[env] Waiting for action from {current_player_symbol} (waited {action_wait_count * 0.01:.1f}s)")
                                action_wait_count += 1

                                for peer_id, msg in events:
                                    if not isinstance(msg, dict):
                                        continue

                                    mtype = msg.get("type")

                                    # Handle discovery messages (just log, no relaying)
                                    if mtype in (DISCOVERY_STARTUP, DISCOVERY_SHUTDOWN):
                                        sender_peer_id = msg.get(
                                            "peer_id", "unknown")
                                        print(
                                            f"[env] Received {mtype} from {sender_peer_id}")
                                        continue

                                    # Only process ACTION messages from the current player
                                    if mtype == ACTION:
                                        # Determine which player this is (should only receive from active player now)
                                        if peer_id == brain_x_id and current_player_symbol == 'X':
                                            action_received = True
                                            player_name = "X"
                                            action_wait_count = 0  # Reset wait counter
                                        elif peer_id == brain_o_id and current_player_symbol == 'O':
                                            action_received = True
                                            player_name = "O"
                                            action_wait_count = 0  # Reset wait counter
                                        else:
                                            # Action from wrong player, ignore (should be rare now that we only send obs to active player)
                                            # Only log if it happens multiple times to avoid spam
                                            if not hasattr(env, '_wrong_action_count'):
                                                env._wrong_action_count = {}
                                            if peer_id not in env._wrong_action_count:
                                                env._wrong_action_count[peer_id] = 0
                                            env._wrong_action_count[peer_id] += 1
                                            if env._wrong_action_count[peer_id] <= 3:
                                                print(
                                                    f"[env] Ignoring action from {peer_id} (not current player {current_player_symbol})")
                                            continue

                                        actions = msg.get("actions", [])
                                        if not actions:
                                            print(
                                                f"[env] no actions in message from {player_name}")
                                            continue

                                        action = int(actions[0])

                                        # Apply action to environment
                                        if current_player_symbol == 'X':
                                            obs, reward, done, info = env.step(
                                                action)
                                        else:
                                            obs, reward, done, info = env.make_opponent_move(
                                                action)

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

                                        # Send updated observation only to the player whose turn it is next
                                        next_player = 'O' if current_player_symbol == 'X' else 'X'
                                        legal_actions = env.get_valid_actions(
                                            next_player)
                                        legal_mask = [0.0 if i in legal_actions else float(
                                            '-inf') for i in range(9)]

                                        # Only send observation to the player whose turn it is
                                        if next_player == 'X':
                                            try:
                                                obs_x = env._get_obs('X')
                                                connection_manager.send(brain_x_id, {
                                                    "type": OBSERVATION,
                                                    "sensors": obs_x.tolist(),
                                                    "info": {"t": info.get("t", 0.0), "episode": episode, "player": "X", "current_turn": next_player, "legal_actions": legal_mask},
                                                })
                                            except Exception as e:
                                                print(
                                                    f"\n[env] Player X disconnected while sending observation: {e}")
                                                raise
                                        else:  # next_player == 'O'
                                            try:
                                                obs_o = env._get_obs('O')
                                                connection_manager.send(brain_o_id, {
                                                    "type": OBSERVATION,
                                                    "sensors": obs_o.tolist(),
                                                    "info": {"t": info.get("t", 0.0), "episode": episode, "player": "O", "current_turn": next_player, "legal_actions": legal_mask},
                                                })
                                            except Exception as e:
                                                print(
                                                    f"\n[env] Player O disconnected while sending observation: {e}")
                                                raise

                                        # Send rewards when game ends
                                        if done:
                                            try:
                                                connection_manager.send(brain_x_id, {
                                                    "type": REWARD,
                                                    "value": float(reward_x),
                                                    "info": {"t": info.get("t", 0.0), "episode": episode, "player": "X"},
                                                })
                                            except Exception as e:
                                                print(
                                                    f"\n[env] Player X disconnected while sending reward: {e}")
                                                raise

                                            try:
                                                connection_manager.send(brain_o_id, {
                                                    "type": REWARD,
                                                    "value": float(reward_o),
                                                    "info": {"t": info.get("t", 0.0), "episode": episode, "player": "O"},
                                                })
                                            except Exception as e:
                                                print(
                                                    f"\n[env] Player O disconnected while sending reward: {e}")
                                                raise

                                            # Send TERMINAL message to both players
                                            try:
                                                connection_manager.send(brain_x_id, {
                                                    "type": TERMINAL,
                                                    "info": {"t": info.get("t", 0.0), "episode": episode, "winner": env.winner},
                                                })
                                            except Exception as e:
                                                print(
                                                    f"\n[env] Player X disconnected while sending terminal: {e}")
                                                raise

                                            try:
                                                connection_manager.send(brain_o_id, {
                                                    "type": TERMINAL,
                                                    "info": {"t": info.get("t", 0.0), "episode": episode, "winner": env.winner},
                                                })
                                            except Exception as e:
                                                print(
                                                    f"\n[env] Player O disconnected while sending terminal: {e}")
                                                raise
                                            break  # Game done

                                        # Switch players
                                        current_player_symbol = 'O' if current_player_symbol == 'X' else 'X'

                                if not action_received:
                                    # No action yet, wait a bit
                                    time.sleep(0.01)
                                    # Safety check: if we've been waiting too long and done is still False, something is wrong
                                    if action_wait_count > 1000 and not done:
                                        print(
                                            f"[env] WARNING: Episode {episode} waiting too long for action from {current_player_symbol} (waited {action_wait_count * 0.01:.1f}s), done={done}")
                                        # Don't break, just log - let it continue waiting
                    except Exception as e:
                        print(f"\n[env] Error in game loop: {e}")
                        import traceback
                        traceback.print_exc()
                        # ConnectionManager handles disconnections automatically
                        # Just reassign brains and continue
                        assign_brains()
                        if brain_x_id is None or brain_o_id is None:
                            print("[env] Waiting for brain reconnection...")
                            break  # Exit game loop, go back to waiting
                        else:
                            # Brains are still connected, continue with next episode
                            print(
                                f"[env] Continuing with next episode after error (episode {episode})...")
                            episode += 1  # Increment episode counter
                            continue  # Continue to next episode in the outer while True loop

                    except KeyboardInterrupt:
                        print("\n[env] interrupted by user")
                        break

        else:
            # Self-play mode (single brain)
            brain_id = None

            try:
                episode = 0
                while True:
                    # Wait for brain connection
                    if brain_id is None:
                        connected_brains = [
                            peer_id for peer_id, metadata in connection_manager.connection_metadata.items()
                            if (peer_id in connection_manager.connections and
                                metadata.get("peer_type") == PEER_TYPE_BRAIN)
                        ]
                        if connected_brains:
                            brain_id = connected_brains[0]
                            print(
                                f"[env] Brain {brain_id} connected for self-play")
                        else:
                            print("[env] waiting for brain connection...")
                            time.sleep(0.1)
                            continue

                    # Check if brain is still connected
                    if brain_id not in connection_manager.connections:
                        print(f"[env] Brain {brain_id} disconnected")
                        brain_id = None
                        continue

                    # Start new episode
                    episode += 1
                    env = TicTacToeEnv(training_mode=training_mode)
                    x = env.reset()
                    done = False
                    current_player = 1  # X starts

                    # Send initial observation
                    try:
                        current_player_symbol = 'X' if current_player == 1 else 'O'
                        legal_actions = env.get_valid_actions(
                            current_player_symbol)
                        legal_mask = [0.0 if i in legal_actions else float(
                            '-inf') for i in range(9)]
                        connection_manager.send(brain_id, {
                            "type": OBSERVATION,
                            "sensors": x.tolist(),
                            "info": {"t": 0.0, "episode": episode, "legal_actions": legal_mask},
                        })
                    except Exception as e:
                        print(f"[env] Error sending initial observation: {e}")
                        brain_id = None
                        continue

                    while not done:
                        # Poll for messages
                        events = connection_manager.poll_events()
                        action_received = False

                        for peer_id, msg in events:
                            if peer_id != brain_id:
                                continue

                            if not isinstance(msg, dict):
                                continue

                            mtype = msg.get("type")

                            # Handle discovery messages
                            if mtype in (DISCOVERY_STARTUP, DISCOVERY_SHUTDOWN):
                                continue

                            if mtype == ACTION:
                                actions = msg.get("actions", [])
                                if not actions:
                                    continue

                                action = int(actions[0])
                                action_received = True

                                # Apply action to environment
                                if current_player == 1:
                                    obs, reward, done, info = env.step(action)
                                else:
                                    obs, reward, done, info = env.make_opponent_move(
                                        action)

                                # Determine reward
                                if done:
                                    if env.winner == 'X':
                                        final_reward = 1.0
                                    elif env.winner == 'O':
                                        final_reward = -1.0
                                    else:
                                        final_reward = 0.0
                                else:
                                    final_reward = 0.0

                                # Send updated observation
                                try:
                                    next_player = -current_player
                                    next_player_symbol = 'X' if next_player == 1 else 'O'
                                    legal_actions = env.get_valid_actions(
                                        next_player_symbol)
                                    legal_mask = [0.0 if i in legal_actions else float(
                                        '-inf') for i in range(9)]
                                    connection_manager.send(brain_id, {
                                        "type": OBSERVATION,
                                        "sensors": obs.tolist(),
                                        "info": {"t": info.get("t", 0.0), "episode": episode, "player": current_player, "legal_actions": legal_mask},
                                    })

                                    if done:
                                        player_reward = final_reward * current_player
                                        connection_manager.send(brain_id, {
                                            "type": REWARD,
                                            "value": float(player_reward),
                                            "info": {"t": info.get("t", 0.0), "episode": episode, "player": current_player},
                                        })

                                        connection_manager.send(brain_id, {
                                            "type": TERMINAL,
                                            "info": {"t": info.get("t", 0.0), "episode": episode, "winner": env.winner},
                                        })
                                except Exception as e:
                                    print(f"[env] Error sending message: {e}")
                                    brain_id = None
                                    break

                                if done:
                                    break

                                current_player = -current_player

                        if not action_received:
                            time.sleep(0.01)
            except KeyboardInterrupt:
                print("[env] interrupted by user")
                raise
    except KeyboardInterrupt:
        print("\n[env] Shutting down...")
    finally:
        # ConnectionManager.close() will send shutdown messages to all known peers
        connection_manager.close()
        print(f"[env] Environment {env_id} shutdown complete")
