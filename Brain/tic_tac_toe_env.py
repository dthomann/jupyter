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
    BOARD_SIZE = 3
    BOARD_CELLS = 9
    WIN_LENGTH = 3

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
        self.board = np.zeros(self.BOARD_CELLS, dtype=np.int32)
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
        board_2d = self.board.reshape(self.BOARD_SIZE, self.BOARD_SIZE)

        # Check rows, columns, diagonals
        for i in range(self.BOARD_SIZE):
            # Rows
            row_sum = board_2d[i].sum()
            if abs(row_sum) == self.WIN_LENGTH:
                return 'X' if row_sum > 0 else 'O'
            # Columns
            col_sum = board_2d[:, i].sum()
            if abs(col_sum) == self.WIN_LENGTH:
                return 'X' if col_sum > 0 else 'O'

        # Diagonals
        diag1 = board_2d[0, 0] + board_2d[1, 1] + board_2d[2, 2]
        diag2 = board_2d[0, 2] + board_2d[1, 1] + board_2d[2, 0]
        if abs(diag1) == self.WIN_LENGTH:
            return 'X' if diag1 > 0 else 'O'
        if abs(diag2) == self.WIN_LENGTH:
            return 'X' if diag2 > 0 else 'O'

        # Check for draw
        if (self.board != 0).all():
            return 'draw'

        return None

    def _is_valid_move(self, action):
        """Check if action is valid."""
        if not isinstance(action, (int, np.integer)):
            return False
        if action < 0 or action >= self.BOARD_CELLS:
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
        valid_actions = [i for i in range(
            self.BOARD_CELLS) if self.board[i] == 0]

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

    def _get_winning_lines(self):
        """
        Get all winning lines (rows, columns, diagonals) as lists of indices.
        Returns list of tuples, each containing indices for a winning line.
        """
        lines = []
        # Rows
        for i in range(self.BOARD_SIZE):
            lines.append(tuple(i * self.BOARD_SIZE +
                         j for j in range(self.BOARD_SIZE)))
        # Columns
        for j in range(self.BOARD_SIZE):
            lines.append(tuple(i * self.BOARD_SIZE +
                         j for i in range(self.BOARD_SIZE)))
        # Diagonals
        lines.append(tuple(i * self.BOARD_SIZE +
                     i for i in range(self.BOARD_SIZE)))  # [0, 4, 8]
        lines.append(tuple(i * self.BOARD_SIZE + (self.BOARD_SIZE - 1 - i)
                     for i in range(self.BOARD_SIZE)))  # [2, 4, 6]
        return lines

    def _find_winning_actions(self, player):
        """
        Find actions that would result in an immediate win for the given player.
        Returns list of action indices, or empty list if no immediate win possible.
        """
        winning_actions = []
        winning_lines = self._get_winning_lines()

        for line_indices in winning_lines:
            line_values = self.board[np.array(line_indices)]
            player_count = np.sum(line_values == player)
            empty_count = np.sum(line_values == 0)

            if player_count == 2 and empty_count == 1:
                # Find the empty spot in this line
                empty_idx = line_indices[np.where(line_values == 0)[0][0]]
                if empty_idx not in winning_actions:
                    winning_actions.append(empty_idx)

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
        winning_lines = self._get_winning_lines()

        for line_indices in winning_lines:
            if action not in line_indices:
                continue
            line_values = self.board[np.array(line_indices)]
            opponent_count = np.sum(line_values == opponent)
            player_count = np.sum(line_values == player)

            if opponent_count == 2 and player_count == 1:
                return True
        return False

    def _creates_threat(self, action, player):
        """Check if the move creates a threat (two in a row with empty third)."""
        winning_lines = self._get_winning_lines()

        for line_indices in winning_lines:
            if action not in line_indices:
                continue
            line_values = self.board[np.array(line_indices)]
            player_count = np.sum(line_values == player)
            empty_count = np.sum(line_values == 0)

            if player_count == 2 and empty_count == 1:
                return True
        return False

    def render(self):
        """Print the current board state."""
        board_2d = self.board.reshape(self.BOARD_SIZE, self.BOARD_SIZE)
        symbols = {1: 'X', -1: 'O', 0: ' '}

        print("\n  " + "   ".join(str(i) for i in range(self.BOARD_SIZE)))
        for i in range(self.BOARD_SIZE):
            row_str = f"{i} "
            for j in range(self.BOARD_SIZE):
                val = board_2d[i, j]
                row_str += f" {symbols[val]} "
                if j < self.BOARD_SIZE - 1:
                    row_str += "|"
            print(row_str)
            if i < self.BOARD_SIZE - 1:
                print("  " + "-" * (self.BOARD_SIZE * 4 - 1))
        print()


def _flush_all_messages(connection_manager):
    """Discard everything currently buffered."""
    while True:
        events = connection_manager.poll_events()
        if not events:
            break


def _wait_for_valid_action(connection_manager, expected_player,
                           brain_x_id, brain_o_id, episode, turn):
    """
    Wait until exactly one valid ACTION message arrives.
    All others are ignored.
    Raises ConnectionError if the expected brain disconnects.
    """
    expected_id = brain_x_id if expected_player == 'X' else brain_o_id

    while True:
        # Check if expected brain is still connected
        with connection_manager.connection_lock:
            if expected_id not in connection_manager.connections:
                raise ConnectionError(
                    f"Brain {expected_id} disconnected while waiting for action")

        events = connection_manager.poll_events()
        if not events:
            time.sleep(0.005)
            continue

        for peer_id, msg in events:
            # Message logging disabled
            # print(f"peer_id: {peer_id}, msg: {msg}")
            # ignore non-dicts
            if not isinstance(msg, dict):
                continue

            # ignore anything not ACTION
            if msg.get("type") != "action":
                # Message logging disabled
                # print(f"not action: {msg.get('type')}")
                continue

            # must come from correct player
            if peer_id != expected_id:
                # Message logging disabled
                # print(f"not expected id: {peer_id}")
                continue

            info = msg.get("info", {})
            # must have correct episode
            if info.get("episode") != episode:
                # Message logging disabled
                # print(f"not expected episode: {info.get('episode')}")
                continue

            # must have correct turn
            if info.get("turn") != turn:
                # Message logging disabled
                # print(
                #     f"not expected turn: {info.get('turn')}", f"turn: {turn}")
                continue

            # extract first action
            actions = msg.get("actions", [])
            if not actions:
                # Message logging disabled
                # print(f"no actions: {actions}")
                continue

            return int(actions[0])


def run_tictactoe_state_machine(env, connection_manager, brain_x_id, brain_o_id, start_episode=1):
    """
    Deterministic, robust two-brain TicTacToe loop.
    Assumes both brains are connected and identified.
    Returns if either brain disconnects.

    env: TicTacToeEnv instance
    connection_manager: ConnectionManager
    brain_x_id, brain_o_id: peer_ids of the two brains
    """
    episode = start_episode
    rng = np.random.RandomState()

    while True:
        # Check if both brains are still connected at the start of each episode
        with connection_manager.connection_lock:
            if (brain_x_id not in connection_manager.connections or
                    brain_o_id not in connection_manager.connections):
                print(f"[env] Brain disconnected during game, exiting state machine")
                return  # Return to outer loop to handle reconnection
        # ------------------------------------------------------------------
        # EPISODE START
        # ------------------------------------------------------------------
        env.reset()
        done = False

        # Choose starting player
        current_player = 'X' if rng.random() < 0.5 else 'O'
        turn = 0

        # FLUSH any stale messages BEFORE sending initial observation
        _flush_all_messages(connection_manager)

        # ------------------------------------------------------------------
        # MAIN TURN LOOP
        # ------------------------------------------------------------------
        while not done:
            legal = env.get_valid_actions(current_player)
            legal_mask = [0.0 if i in legal else float(
                '-inf') for i in range(env.BOARD_CELLS)]

            if current_player == 'X':
                obs_x = env._get_obs('X').tolist()
                connection_manager.send(brain_x_id, {
                    "type": OBSERVATION,
                    "episode": episode,
                    "turn": turn,
                    "sensors": obs_x,
                    "info": {"episode": episode, "player": "X", "current_turn": turn, "legal_actions": legal_mask}
                })
                # Check if connection is still valid after send
                with connection_manager.connection_lock:
                    if brain_x_id not in connection_manager.connections:
                        print(
                            f"[env] Brain X ({brain_x_id}) disconnected, exiting state machine")
                        return
            else:
                obs_o = env._get_obs('O').tolist()
                connection_manager.send(brain_o_id, {
                    "type": OBSERVATION,
                    "episode": episode,
                    "turn": turn,
                    "sensors": obs_o,
                    "info": {"episode": episode, "player": "O", "current_turn": turn, "legal_actions": legal_mask}
                })
                # Check if connection is still valid after send
                with connection_manager.connection_lock:
                    if brain_o_id not in connection_manager.connections:
                        print(
                            f"[env] Brain O ({brain_o_id}) disconnected, exiting state machine")
                        return
            # Wait for correct ACTION
            try:
                action = _wait_for_valid_action(
                    connection_manager=connection_manager,
                    expected_player=current_player,
                    brain_x_id=brain_x_id,
                    brain_o_id=brain_o_id,
                    episode=episode,
                    turn=turn
                )
            except ConnectionError as e:
                print(f"[env] {e}, exiting state machine")
                return  # Return to outer loop to handle reconnection

            # Apply action
            if current_player == 'X':
                obs, reward, done, info = env.step(action)
            else:
                obs, reward, done, info = env.make_opponent_move(action)

            # --------------------------------------------------------------
            # TERMINAL: SEND REWARD + TERMINAL to BOTH
            # --------------------------------------------------------------
            if done:
                if env.winner == 'X':
                    reward_x, reward_o = 1.0, -1.0
                elif env.winner == 'O':
                    reward_x, reward_o = -1.0, 1.0
                else:
                    reward_x, reward_o = 0.0, 0.0

                connection_manager.send(brain_x_id, {
                    "type": REWARD,
                    "value": float(reward_x),
                    "info": {"episode": episode}
                })
                connection_manager.send(brain_o_id, {
                    "type": REWARD,
                    "value": float(reward_o),
                    "info": {"episode": episode}
                })
                connection_manager.send(brain_x_id, {
                    "type": TERMINAL,
                    "info": {"episode": episode, "winner": env.winner}
                })
                connection_manager.send(brain_o_id, {
                    "type": TERMINAL,
                    "info": {"episode": episode, "winner": env.winner}
                })
                # Check if connections are still valid after sending terminal messages
                with connection_manager.connection_lock:
                    if (brain_x_id not in connection_manager.connections or
                            brain_o_id not in connection_manager.connections):
                        print(
                            f"[env] Brain disconnected while sending terminal messages, exiting state machine")
                        return
                break

            # Next player
            current_player = 'O' if current_player == 'X' else 'X'
            turn += 1
        # END WHILE NOT DONE
        episode += 1


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

        # Get all connected brains
        # We check connections first, then filter by metadata to ensure we catch newly connected brains
        connected_brains = []
        with connection_manager.connection_lock:
            # First, get all active connections
            active_connections = set(connection_manager.connections.keys())
            # Then check metadata for each connection
            for peer_id in active_connections:
                metadata = connection_manager.connection_metadata.get(
                    peer_id, {})
                # Include if it's marked as a brain and not disconnected
                if (metadata.get("peer_type") == PEER_TYPE_BRAIN and
                        not metadata.get("disconnected", False)):
                    connected_brains.append(peer_id)

        # Debug: log all connections to help diagnose issues (only when something changes)
        if len(connected_brains) != (2 if brain_x_id and brain_o_id else (1 if brain_x_id or brain_o_id else 0)):
            all_connections = list(connection_manager.connections.keys())
            all_metadata = {pid: metadata.get("peer_type", "unknown")
                            for pid, metadata in connection_manager.connection_metadata.items()}
            print(f"[env] assign_brains: connected_brains={connected_brains}, "
                  f"brain_x_id={brain_x_id}, brain_o_id={brain_o_id}, "
                  f"all_connections={all_connections}, all_metadata={all_metadata}")

        # Check if currently assigned brains are still connected
        if brain_x_id is not None:
            if brain_x_id not in connection_manager.connections:
                print(
                    f"[env] Previously assigned brain X ({brain_x_id}) is no longer connected, resetting")
                brain_x_id = None
                logged_waiting_x = False
            elif brain_x_id not in connected_brains:
                print(
                    f"[env] Previously assigned brain X ({brain_x_id}) is no longer valid, resetting")
                brain_x_id = None
                logged_waiting_x = False

        if brain_o_id is not None:
            if brain_o_id not in connection_manager.connections:
                print(
                    f"[env] Previously assigned brain O ({brain_o_id}) is no longer connected, resetting")
                brain_o_id = None
                logged_waiting_o = False
            elif brain_o_id not in connected_brains:
                print(
                    f"[env] Previously assigned brain O ({brain_o_id}) is no longer valid, resetting")
                brain_o_id = None
                logged_waiting_o = False

        # Role-sticky assignment:
        # - Never move a live brain between X and O.
        # - Only fill empty slots from currently connected brains.
        # This ensures connection order (or restarts) doesn't matter in two-brain mode.

        # Ensure X and O are distinct (should never happen, but be defensive)
        if brain_x_id is not None and brain_o_id is not None and brain_x_id == brain_o_id:
            print(
                f"[env] WARNING: brain {brain_x_id} assigned to both X and O; clearing O and waiting for a distinct second brain")
            brain_o_id = None
            logged_waiting_o = False

        # Assign X if empty: pick any connected brain that isn't currently O
        if brain_x_id is None:
            for candidate in connected_brains:
                if candidate != brain_o_id:
                    brain_x_id = candidate
                    print(f"[env] Assigned brain {brain_x_id} as player X")
                    logged_waiting_x = False
                    break

        # Assign O if empty: pick any connected brain that isn't currently X
        if brain_o_id is None:
            for candidate in connected_brains:
                if candidate != brain_x_id:
                    brain_o_id = candidate
                    print(f"[env] Assigned brain {brain_o_id} as player O")
                    logged_waiting_o = False
                    break

    try:
        # Give connections a moment to establish
        time.sleep(0.5)

        while True:
            if require_two_brains:
                # Process incoming messages (including startup messages from brains)
                # This is needed so we can identify which connections are brains
                # Process events multiple times to ensure all startup messages are handled
                # Keep processing until no more events arrive
                max_iterations = 10
                events_processed = False
                startup_messages_received = False
                for iteration in range(max_iterations):
                    events = connection_manager.poll_events()
                    if not events:
                        # Check if there are any temporary connections waiting for startup messages
                        if iteration == 0:
                            with connection_manager.connection_lock:
                                temp_connections = [pid for pid in connection_manager.connections.keys()
                                                    if pid.startswith("incoming_")]
                            if temp_connections:
                                # Wait a bit for startup messages from temporary connections
                                time.sleep(0.05)
                                continue
                            else:
                                time.sleep(0.01)
                        break
                    events_processed = True
                    for peer_id, msg in events:
                        if isinstance(msg, dict):
                            mtype = msg.get("type")
                            # Handle discovery messages (just log, no relaying)
                            if mtype in (DISCOVERY_STARTUP, DISCOVERY_SHUTDOWN):
                                sender_peer_id = msg.get("peer_id", "unknown")
                                sender_peer_type = msg.get(
                                    "peer_type", "unknown")
                                print(
                                    f"[env] Received {mtype} from {sender_peer_id} (type: {sender_peer_type})")
                                startup_messages_received = True
                                # Startup messages update peer_type, which helps assign_brains() identify brains
                                continue

                # If we received startup messages, give a tiny moment for metadata updates to complete
                if startup_messages_received:
                    time.sleep(0.01)

                # Environments do NOT proactively connect to brains
                # Brains will connect to the environment after discovering it via multicast
                # We only accept incoming connections from brains

                # After processing events, check if there are any temporary connections
                # (incoming_*) that haven't sent startup messages yet, or if there are
                # new connections that we haven't processed startup messages for
                # Keep polling until we've processed all startup messages
                max_additional_polls = 5
                for additional_poll in range(max_additional_polls):
                    # Check for temporary connections that might be waiting for startup messages
                    with connection_manager.connection_lock:
                        temp_connections = [pid for pid in connection_manager.connections.keys()
                                            if pid.startswith("incoming_")]
                        # Also check if there are connections without peer_type set
                        connections_without_type = [
                            pid for pid in connection_manager.connections.keys()
                            if not connection_manager.connection_metadata.get(pid, {}).get("peer_type")
                        ]

                    # If there are temporary connections or connections without type, wait a bit
                    # and poll again to catch their startup messages
                    if temp_connections or connections_without_type:
                        time.sleep(0.05)

                    # Process additional events to catch any startup messages
                    additional_events = connection_manager.poll_events()
                    if additional_events:
                        for peer_id, msg in additional_events:
                            if isinstance(msg, dict):
                                mtype = msg.get("type")
                                if mtype in (DISCOVERY_STARTUP, DISCOVERY_SHUTDOWN):
                                    sender_peer_id = msg.get(
                                        "peer_id", "unknown")
                                    sender_peer_type = msg.get(
                                        "peer_type", "unknown")
                                    print(
                                        f"[env] Received {mtype} from {sender_peer_id} (type: {sender_peer_type})")
                                    startup_messages_received = True
                        if startup_messages_received:
                            time.sleep(0.01)  # Give metadata time to update
                    elif not temp_connections and not connections_without_type:
                        # No more events and no temporary connections, we're done
                        break

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

                # Safety: never allow the same brain to occupy both roles.
                # If this ever happens (shouldn't), clear O and wait for another brain.
                if brain_x_id == brain_o_id:
                    print(
                        f"[env] WARNING: same brain assigned to X and O ({brain_x_id}); clearing O and waiting for a different second brain")
                    brain_o_id = None
                    logged_waiting_o = False
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

                    # Create environment instance
                    env = TicTacToeEnv(training_mode=training_mode)

                    # Run clean state machine (returns if a brain disconnects)
                    run_tictactoe_state_machine(
                        env, connection_manager, brain_x_id, brain_o_id)

                    # State machine returned, check if connections are still valid
                    # This handles the case where a brain disconnected during the game
                    conn_x = get_brain_connection(brain_x_id)
                    conn_o = get_brain_connection(brain_o_id)
                    if conn_x is None:
                        print(
                            f"[env] Brain X ({brain_x_id}) disconnected, will reassign on next iteration")
                        brain_x_id = None
                    if conn_o is None:
                        print(
                            f"[env] Brain O ({brain_o_id}) disconnected, will reassign on next iteration")
                        brain_o_id = None
                    # Continue loop to reassign brains if needed

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
                                metadata.get("peer_type") == PEER_TYPE_BRAIN and
                                not metadata.get("disconnected", False))
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
                            '-inf') for i in range(env.BOARD_CELLS)]
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
                                        '-inf') for i in range(env.BOARD_CELLS)]
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
