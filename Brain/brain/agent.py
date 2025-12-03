from brainprotocol import OBSERVATION, REWARD, ACTION, TERMINAL, SHUTDOWN
import numpy as np
import pickle
import time
import sys
import math
import json
from pathlib import Path
from typing import Optional, Dict, Any
from multiprocessing.connection import Client
from .world_model import HierarchicalWorldModel
from .actor_critic import ActorCritic
from .memory import EpisodicMemory
from .neuromodulators import NeuromodulatorState
from .motivation import IntrinsicMotivation, DriveState

# Import protocol constants
sys.path.insert(0, str(Path(__file__).parent.parent))


class BrainAgent:
    """
    Integrated agent:
    - optional multi-modal encoder
    - hierarchical predictive coding world model
    - actor-critic RL
    - neuromodulators
    - intrinsic motivation (curiosity + learning progress)
    - slow drives (persistent valence)
    - episodic memory + offline replay
    - continuous run with save/resume
    """

    def __init__(
        self,
        obs_dim=None,
        latent_dims=None,
        n_actions=4,
        encoder=None,
        lr_model=1e-3,
        lr_policy=1e-2,
        replay_batch_size=32,
        use_raw_obs_for_policy=False,
        episode_based_learning=False,
        entropy_coeff=0.0,
        reward_shaping=None,
        rng=None,
    ):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.encoder = encoder
        if self.encoder is not None:
            obs_dim_effective = self.encoder.output_dim
        else:
            if obs_dim is None:
                raise ValueError("Either obs_dim or encoder must be provided.")
            obs_dim_effective = obs_dim

        if latent_dims is None:
            latent_dims = [64, 32]

        self.world_model = HierarchicalWorldModel(
            obs_dim_effective, latent_dims, rng=rng)
        top_dim = latent_dims[-1]
        # Use raw observations for policy if requested (more stable for simple tasks)
        self.use_raw_obs_for_policy = use_raw_obs_for_policy
        policy_state_dim = obs_dim_effective if use_raw_obs_for_policy else top_dim
        self.actor_critic = ActorCritic(
            policy_state_dim, n_actions, rng=rng, entropy_coeff=entropy_coeff)
        self.memory = EpisodicMemory(rng=rng)
        self.neuromodulators = NeuromodulatorState()
        self.intrinsic = IntrinsicMotivation()
        self.drives = DriveState()
        self.lr_model = lr_model
        self.lr_policy = lr_policy
        self.replay_batch_size = replay_batch_size
        self.episode_based_learning = episode_based_learning
        self.entropy_coeff = entropy_coeff
        self.reward_shaping = reward_shaping  # Function to shape rewards, or None

        # Episode buffer for episode-based learning
        self.episode_buffer = []

        self.global_step = 0
        self.episode_index = 0

        # Performance tracking
        # List of outcomes: 1.0 (X win), -1.0 (O win), 0.0 (draw)
        self.episode_outcomes = []
        self.stats_window = 100  # Number of recent episodes for statistics
        self.training_metrics_history = []
        self._max_metrics_history = 10000
        self._last_random_eval = None
        self._last_episode_metrics = None

        # Client protocol state (for server communication)
        self.pending_decision = False
        self.last_sensors = None
        # Legal actions from environment (mask format: 0 for legal, -inf for illegal)
        self.last_legal_actions = None
        self.cum_reward = 0.0
        self.last_action = None
        self.dt = 0.02  # Default tick interval

        # Self-play episode state
        self.episode_board = None
        self.episode_player = None
        self.episode_states = []
        self.episode_actions = []
        self.episode_players = []
        self.episode_masks = []

    # ----- input handling -----

    def _prepare_obs(self, obs):
        """
        Convert raw input to flat observation vector.
        """
        if self.encoder is not None:
            if not isinstance(obs, dict):
                raise ValueError(
                    "With encoder set, obs must be dict of modality->array.")
            return self.encoder.encode(obs)
        else:
            return np.asarray(obs).reshape(-1)

    def encode_state(self, obs):
        """
        Map raw observation to (latent z, encoded x).
        """
        x = self._prepare_obs(obs)
        z = self.world_model.encode_state(x)
        return z, x

    # ----- legal actions -----

    def get_legal_actions(self, obs):
        """
        DEPRECATED: Fallback method to infer legal actions from observation.

        This method contains game-specific logic (tic-tac-toe) and should not be used
        in a general-purpose agent. Environments should provide legal_actions in the
        OBSERVATION message info dict. This method is kept only for backward compatibility.

        For tic-tac-toe: empty positions (value == 0) are legal.

        Returns:
            legal_mask: Array of shape (n_actions,), 0 for legal, -inf for illegal
        """
        x = self._prepare_obs(obs)
        # For tic-tac-toe: board positions with value 0 are empty/legal
        # This is game-specific logic and should be avoided
        legal_mask = np.zeros(self.actor_critic.n_actions)
        for i in range(min(len(x), self.actor_critic.n_actions)):
            if abs(x[i]) > 1e-6:  # Position is occupied
                legal_mask[i] = float('-inf')
        return legal_mask

    # ----- acting -----

    def act(self, obs, temperature=1.0, greedy=False, legal_mask=None):
        """
        Encode observation and sample action.

        Args:
            obs: Observation
            temperature: Temperature for action sampling
            greedy: If True, choose best action deterministically
            legal_mask: Optional legal action mask (if None, will try to infer from obs)
        """
        z, x = self.encode_state(obs)
        # Use raw observation for policy if requested, otherwise use latent state
        policy_state = x if self.use_raw_obs_for_policy else z

        # Get legal mask if not provided
        if legal_mask is None:
            try:
                legal_mask = self.get_legal_actions(obs)
            except:
                legal_mask = None  # Fallback if inference fails

        action, probs, value = self.actor_critic.act(
            policy_state, temperature=temperature, greedy=greedy, legal_mask=legal_mask
        )
        return action, z, value, x

    # ----- online learning -----

    def online_update(self, obs, x, z, action, external_reward, next_obs, done, legal_mask=None):
        """
        Single step of learning integrating:
        - world model predictive coding
        - intrinsic motivation
        - drives
        - actor-critic with total valence

        If episode_based_learning is True, buffers transitions instead of updating immediately.
        """
        z_next, x_next = self.encode_state(next_obs)

        # Apply reward shaping if provided
        if self.reward_shaping is not None:
            external_reward = self.reward_shaping(external_reward)

        # Use NE + ACh to gate world model plasticity
        neuromod_factor = (
            0.5 * self.neuromodulators.norepinephrine
            + 0.5 * self.neuromodulators.acetylcholine
        )

        # Update world model and get prediction error
        _, pred_error_norm = self.world_model.learn(
            x=x, neuromod_factor=neuromod_factor, lr_model=self.lr_model
        )

        # Intrinsic reward from prediction error
        intrinsic_reward, components = self.intrinsic.compute(pred_error_norm)

        # Persistent drives from intrinsic components
        drive_vec = self.drives.update(components)

        # Drives scale intrinsic reward
        drive_gain = 1.0 + np.tanh(drive_vec.mean())
        total_reward = float(external_reward) + drive_gain * intrinsic_reward

        if self.episode_based_learning:
            # Buffer transition for episode-based update
            policy_state = x if self.use_raw_obs_for_policy else z
            policy_next_state = x_next if self.use_raw_obs_for_policy else z_next

            # Get value estimate for this state
            value_t, _ = self.actor_critic._forward_value(policy_state)
            value = value_t.item() if hasattr(value_t, 'item') else float(value_t)

            self.episode_buffer.append({
                'obs': obs,
                'x': x,
                'z': z,
                'action': action,
                'external_reward': external_reward,
                'total_reward': total_reward,
                'next_obs': next_obs,
                'x_next': x_next,
                'z_next': z_next,
                'done': done,
                'pred_error_norm': pred_error_norm,
                'intrinsic_reward': intrinsic_reward,
                'drive_vec': drive_vec,
                'value': value,
                'legal_mask': legal_mask,
            })

            # If episode is done, perform episode-based update
            if done:
                self.update_from_episode()

            td_error = 0.0  # Not used in episode-based learning
        else:
            # Standard TD update
            # Use raw observations for policy if requested, otherwise use latent states
            policy_state = x if self.use_raw_obs_for_policy else z
            policy_next_state = x_next if self.use_raw_obs_for_policy else z_next
            td_error, value, next_value = self.actor_critic.update(
                state=policy_state,
                action=action,
                reward=total_reward,
                next_state=policy_next_state,
                done=done,
                neuromodulators=self.neuromodulators,
                base_lr=self.lr_policy,
                legal_mask=legal_mask,
                entropy_coeff=self.entropy_coeff,
            )

            # Final neuromodulator update with prediction error included
            self.neuromodulators.update(
                reward=total_reward,
                value=value,
                next_value=next_value,
                pred_error_norm=pred_error_norm,
            )

        # Store transition
        self.memory.store(x, z, action, total_reward, x_next, z_next, done)

        self.global_step += 1

        return td_error, pred_error_norm, intrinsic_reward, drive_vec

    def update_from_episode(self):
        """
        REINFORCE-style update using all transitions in the episode.
        Each transition gets its own reward (usually outcome * player).
        Uses PyTorch-based batch update for efficiency.
        """
        if len(self.episode_buffer) == 0:
            return

        # Collect states, actions, rewards, and legal masks
        states = []
        actions = []
        rewards = []
        legal_masks = []

        for transition in self.episode_buffer:
            x = transition['x']
            z = transition['z']
            action = transition['action']
            legal_mask = transition.get('legal_mask')

            # Use raw observations for policy if requested
            policy_state = x if self.use_raw_obs_for_policy else z

            # Get per-transition reward (should be outcome * player for self-play)
            reward = transition['external_reward']
            if self.reward_shaping is not None:
                reward = self.reward_shaping(reward)

            states.append(policy_state)
            actions.append(action)
            rewards.append(reward)
            legal_masks.append(legal_mask)

        # Use PyTorch-based REINFORCE update with per-transition rewards
        self.actor_critic.update_reinforce(
            states=states,
            actions=actions,
            rewards=rewards,  # Each transition gets its own reward
            legal_masks=legal_masks if any(
                m is not None for m in legal_masks) else None,
            entropy_coeff=self.entropy_coeff,
            lr=self.lr_policy,  # Use the agent's learning rate
        )

        # Clear episode buffer
        self.episode_buffer = []

    def reset_episode(self):
        """
        Reset episode buffer (useful when starting a new episode).
        """
        if self.episode_based_learning:
            self.episode_buffer = []

    # ----- client protocol methods -----

    def on_observation(self, sensors, info):
        """
        Handle observation message from environment server.
        Stores observation and marks that a decision should be made.
        Extracts legal_actions from info if provided by environment.
        """
        # Check if this is a new episode (episode number in info)
        episode_num = info.get("episode", 0)
        if episode_num > self.episode_index:
            # New episode starting - reset episode state
            # CRITICAL FIX: Increment episode_index immediately to prevent multiple resets
            # (Otherwise every observation in the same episode would trigger a reset)
            self.episode_index = episode_num
            self.episode_board = None
            self.episode_player = None
            self.episode_states = []
            self.episode_actions = []
            self.episode_players = []
            self.episode_masks = []
            self.cum_reward = 0.0
            # Reset random player for new episode
            self._current_episode_random_player = None

        self.last_sensors = list(sensors) if sensors is not None else []

        # Store observation info for use in tick() (e.g., to get current player)
        self._last_observation_info = info

        # Extract legal actions from environment if provided
        if "legal_actions" in info:
            legal_actions = info["legal_actions"]
            # Convert to numpy array if it's a list
            if isinstance(legal_actions, list):
                self.last_legal_actions = np.array(
                    legal_actions, dtype=np.float32)
            else:
                self.last_legal_actions = legal_actions
        else:
            # Environment didn't provide legal actions - will use fallback if available
            self.last_legal_actions = None

        self.pending_decision = True

    def on_reward(self, value: float, info):
        """
        Handle reward message from environment server.
        Accumulates reward and marks that a decision should be made.
        """
        self.cum_reward += float(value)
        self.pending_decision = True

    def tick(self, dt: float) -> Optional[Dict[str, Any]]:
        """
        Called regularly. Returns an ACTION message dict when a new action
        should be sent, otherwise None.
        """
        if not self.pending_decision:
            return None

        # Convert sensors to observation format
        if self.last_sensors is None or len(self.last_sensors) == 0:
            return None

        obs = np.array(self.last_sensors, dtype=np.float32)

        # Update episode board from observation (raw, unflipped board)
        prev_board = self.episode_board.copy() if self.episode_board is not None else None
        self.episode_board = obs.copy()

        # Determine current player by counting moves (like play_episode_simple)
        # X goes first, so if X_count == O_count, it's X's turn (player 1)
        # If X_count > O_count, it's O's turn (player -1)
        x_count = np.sum(self.episode_board == 1)
        o_count = np.sum(self.episode_board == -1)
        self.episode_player = 1 if x_count == o_count else -1

        # SYMMETRY NOTE: We do NOT flip the board observation here.
        # The policy learns on raw states (X=1, O=-1) for both players.
        # Symmetry is achieved through correct reward sign: reward = outcome * player
        # This matches the approach in generic_tictactoe.py and train_brain_tictactoe.py

        # Use legal actions from environment if provided, otherwise fallback to inference
        if self.last_legal_actions is not None:
            legal_mask = self.last_legal_actions
            # Validate that at least one action is legal
            if isinstance(legal_mask, np.ndarray):
                if np.all(np.isinf(legal_mask) & (legal_mask < 0)):
                    # All actions are masked - this shouldn't happen, use fallback
                    legal_mask = None
        else:
            # Fallback: try to infer legal actions (for backward compatibility)
            try:
                legal_mask = self.get_legal_actions(obs)
            except:
                legal_mask = None

        # Check if this is the first move (empty board) - decide which player will be random
        is_first_move = (x_count == 0 and o_count == 0)
        if is_first_move:
            # At start of episode, randomly decide which player (X or O) will play randomly this game
            random_opponent_prob = self._get_random_opponent_probability()
            if self.rng.random() < random_opponent_prob:
                # Randomly choose which player will be random (X=1 or O=-1)
                self._current_episode_random_player = self.rng.choice([1, -1])
            else:
                # No random player this episode
                self._current_episode_random_player = None

        # Check if current player is the random player for this episode
        use_random_action = (self.episode_player == getattr(
            self, '_current_episode_random_player', None))

        if use_random_action:
            # Current player plays randomly this episode - pick a random legal action
            if legal_mask is not None:
                # Get legal actions from mask (actions that are not -inf)
                if isinstance(legal_mask, np.ndarray):
                    legal_actions = np.where(np.isfinite(
                        legal_mask) & (legal_mask >= 0))[0]
                else:
                    legal_actions = [i for i in range(len(legal_mask)) if np.isfinite(
                        legal_mask[i]) and legal_mask[i] >= 0]
            else:
                # Fallback: get legal actions from board state
                legal_actions = np.where(self.episode_board == 0)[0]

            if len(legal_actions) > 0:
                action = int(self.rng.choice(legal_actions))
                # Still encode state for storage (using raw observation)
                z, x = self.encode_state(obs)
                value = None  # No value estimate for random action
            else:
                # Fallback to policy if no legal actions found (shouldn't happen)
                action, z, value, x = self.act(
                    obs, temperature=1.0, greedy=False, legal_mask=legal_mask)
            record_transition = False
        else:
            # Get action from policy (normal case - use policy unless current player is random)
            # CRITICAL: Always use temperature-based exploration during training to allow
            # the agent to discover better strategies. Greedy selection is only for evaluation.
            current_entropy = self._get_current_entropy()
            temperature = self._get_current_temperature()
            # Use greedy mode only when entropy is very low (nearly deterministic)
            greedy = (current_entropy < 0.0001)

            action, z, value, x = self.act(
                obs, temperature=temperature, greedy=greedy, legal_mask=legal_mask)
            record_transition = True

        # Store transition for episode-based learning (self-play)
        if self.episode_based_learning and record_transition:
            self.episode_states.append(
                x.copy() if isinstance(x, np.ndarray) else x)
            self.episode_actions.append(action)
            self.episode_players.append(self.episode_player)
            if legal_mask is not None:
                self.episode_masks.append(legal_mask.copy() if isinstance(
                    legal_mask, np.ndarray) else legal_mask)
            else:
                self.episode_masks.append(None)

        self.last_action = action
        self.pending_decision = False

        return {
            "type": ACTION,
            "actions": [int(action)],
            "info": {"t": time.time(), "cum_reward": self.cum_reward},
        }

    def _check_winner(self, board):
        """Check if there's a winner. Returns 1 for X, -1 for O, 0 for draw, None for ongoing."""
        WIN_LINES = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for a, b, c in WIN_LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        if 0 not in board:
            return 0
        return None

    def _handle_terminal(self, info):
        """
        Handle TERMINAL message - complete episode and trigger training.
        For self-play, the brain manages the full episode internally.
        """
        # If we have an active self-play episode, complete it
        if len(self.episode_states) > 0:
            # Determine outcome from TERMINAL message info (most reliable)
            # In tic-tac-toe: X win = 1.0, O win = -1.0, draw = 0.0
            winner = info.get("winner")
            if winner == 'X':
                outcome = 1.0  # X wins
            elif winner == 'O':
                outcome = -1.0  # O wins
            elif winner == 'draw':
                outcome = 0.0  # Draw
            else:
                # Fallback: use board state if winner not in info
                if self.episode_board is not None:
                    outcome = self._check_winner(self.episode_board)
                    if outcome is None:
                        outcome = 0.0  # Draw
                    else:
                        outcome = float(outcome)
                else:
                    outcome = 0.0

            # REWARD SIGN VERIFICATION: Rewards are from the acting player's perspective
            # outcome: +1.0 (X wins), -1.0 (O wins), 0.0 (draw)
            # player: +1 (X), -1 (O)
            # reward = outcome * player ensures:
            #   - X wins (1.0) * X (1) = +1.0 (X wins from X's perspective) ✓
            #   - X wins (1.0) * O (-1) = -1.0 (X wins = O loses from O's perspective) ✓
            #   - O wins (-1.0) * O (-1) = +1.0 (O wins from O's perspective) ✓
            #   - O wins (-1.0) * X (1) = -1.0 (O wins = X loses from X's perspective) ✓
            # This matches train_brain_tictactoe.py: rewards = [outcome * p for p in players]
            rewards = [outcome * p for p in self.episode_players]

            update_metrics = None

            # Update policy using REINFORCE
            if self.episode_based_learning and len(self.episode_states) > 0:
                # Convert states to numpy arrays if needed
                states_np = []
                for s in self.episode_states:
                    if isinstance(s, np.ndarray):
                        states_np.append(s.astype(np.float32))
                    else:
                        states_np.append(np.array(s, dtype=np.float32))

                # Use decayed entropy coefficient and learning rate
                current_entropy = self._get_current_entropy()
                current_lr = self._get_current_learning_rate()
                update_metrics = self.actor_critic.update_reinforce(
                    states=states_np,
                    actions=self.episode_actions,
                    rewards=rewards,
                    legal_masks=self.episode_masks if any(
                        m is not None for m in self.episode_masks) else None,
                    entropy_coeff=current_entropy,
                    lr=current_lr,  # Use decayed learning rate
                )

            # Track outcome for performance statistics (use outcome computed for training)
            # CRITICAL: Do this BEFORE clearing episode_states!
            self.episode_outcomes.append(outcome)

            # Record diagnostics for this episode
            episode_length = len(self.episode_actions)
            metrics_payload = {
                "episode": int(self.episode_index),
                "outcome": float(outcome),
                "episode_length": episode_length,
                "current_entropy": float(self._get_current_entropy()),
                "current_lr": float(self._get_current_learning_rate()),
                "random_player": getattr(self, '_current_episode_random_player', None),
                "replay_size": len(getattr(self.memory, "buffer", [])),
            }
            if self.episode_players:
                players_np = np.array(self.episode_players, dtype=np.int8)
                metrics_payload["x_moves"] = int((players_np == 1).sum())
                metrics_payload["o_moves"] = int((players_np == -1).sum())
            if update_metrics is not None:
                metrics_payload.update(update_metrics)
            metrics_payload.setdefault(
                "mean_reward", float(np.mean(rewards)) if len(rewards) > 0 else 0.0)
            metrics_payload.setdefault(
                "std_reward", float(np.std(rewards)) if len(rewards) > 1 else 0.0)
            self._record_training_metrics(metrics_payload)

            # Reset episode state
            self.episode_board = None
            self.episode_player = None
            self.episode_states = []
            self.episode_actions = []
            self.episode_players = []
            self.episode_masks = []
        else:
            # No episode states recorded, default to draw
            self.episode_outcomes.append(0.0)
            self._record_training_metrics({
                "episode": int(self.episode_index),
                "outcome": 0.0,
                "episode_length": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "current_entropy": float(self._get_current_entropy()),
                "current_lr": float(self._get_current_learning_rate()),
                "random_player": getattr(self, '_current_episode_random_player', None),
            })

        # Reset cumulative reward for next episode
        self.cum_reward = 0.0
        # NOTE: episode_index is now incremented in on_observation when a new episode starts
        # So we don't need to increment it here

    def _apply_decay(self, start, end, progress, decay_type='linear'):
        """
        Apply decay schedule to interpolate between start and end values.

        Args:
            start: Starting value
            end: Ending value
            progress: Progress from 0.0 to 1.0
            decay_type: 'linear', 'exponential', or 'cosine'

        Returns:
            Decayed value at current progress
        """
        # Early return if no decay (start == end)
        if start == end:
            return start

        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]

        if decay_type == 'linear':
            return start + (end - start) * progress
        elif decay_type == 'exponential':
            # Exponential decay: faster decay early, slower later
            # Formula: end + (start - end) * exp(-k * progress)
            # k is chosen so that at progress=1, we get approximately 'end'
            # For exponential: we want most of decay to happen early
            # Using k such that exp(-k * 1) ≈ 0.01 (99% decay)
            # Scale factor: how much to decay by (1.0 = full decay)
            k = -math.log(0.01)  # k ≈ 4.6 gives ~99% decay at progress=1
            decay_factor = math.exp(-k * progress)
            return end + (start - end) * decay_factor
        elif decay_type == 'cosine':
            # Cosine annealing: smooth S-curve decay
            # Formula: end + (start - end) * 0.5 * (1 + cos(π * progress))
            return end + (start - end) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Unknown decay type, default to linear
            return start + (end - start) * progress

    def _get_current_entropy(self):
        """Calculate current entropy coefficient based on decay schedule."""
        if not hasattr(self, '_entropy_start'):
            # Entropy decay not configured, use constant entropy
            return self.entropy_coeff

        episodes_elapsed = self.episode_index - self._entropy_initial_episode
        if episodes_elapsed >= self._entropy_decay_episodes:
            return self._entropy_end

        progress = episodes_elapsed / self._entropy_decay_episodes
        decay_type = getattr(self, '_entropy_decay_type', 'linear')
        return self._apply_decay(self._entropy_start, self._entropy_end, progress, decay_type)

    def _get_current_temperature(self):
        """Calculate current temperature for action selection based on entropy.
        When entropy is low, temperature should also be low (more deterministic).
        Temperature decays from 1.0 to 0.5 (not 0) to maintain some exploration."""
        current_entropy = self._get_current_entropy()

        # Map entropy to temperature: low entropy -> low temperature (more deterministic)
        # Scale temperature from 1.0 (high entropy) down to 0.5 (low entropy)
        # When entropy=0.001 (max), temp=1.0 (stochastic)
        # When entropy=0, temp=0.5 (still some exploration)
        max_entropy = getattr(self, '_entropy_start',
                              self.entropy_coeff) or 0.001
        if max_entropy > 0:
            # Scale from [0, max_entropy] to [0.5, 1.0]
            temperature = 0.5 + 0.5 * (current_entropy / max_entropy)
            temperature = max(0.5, min(1.0, temperature)
                              )  # Clamp to [0.5, 1.0]
        else:
            temperature = 0.5  # Minimum temperature when entropy is 0

        return temperature

    def _get_current_learning_rate(self):
        """Calculate current learning rate based on decay schedule."""
        if not hasattr(self, '_lr_start'):
            # Learning rate decay not configured, use constant learning rate
            return self.lr_policy

        episodes_elapsed = self.episode_index - self._lr_initial_episode
        if episodes_elapsed >= self._lr_decay_episodes:
            return self._lr_end

        progress = episodes_elapsed / self._lr_decay_episodes
        decay_type = getattr(self, '_lr_decay_type', 'linear')
        return self._apply_decay(self._lr_start, self._lr_end, progress, decay_type)

    def _get_random_opponent_probability(self):
        """Calculate current random opponent probability based on decay schedule."""
        if not hasattr(self, '_random_opponent_prob_start'):
            # Random opponent decay not configured, return 0 (disabled)
            return 0.0

        episodes_elapsed = self.episode_index - self._random_opponent_initial_episode
        if episodes_elapsed >= self._random_opponent_decay_episodes:
            return self._random_opponent_prob_end

        progress = episodes_elapsed / self._random_opponent_decay_episodes
        decay_type = getattr(self, '_random_opponent_decay_type', 'linear')
        return self._apply_decay(self._random_opponent_prob_start,
                                 self._random_opponent_prob_end, progress, decay_type)

    def get_performance_stats(self, window=None):
        """
        Get performance statistics over recent episodes.

        Args:
            window: Number of recent episodes to analyze (None = all episodes)

        Returns:
            dict with keys: wins, losses, draws, win_rate, loss_rate, draw_rate, total
        """
        if window is None:
            window = len(self.episode_outcomes)

        recent = self.episode_outcomes[-window:] if len(
            self.episode_outcomes) >= window else self.episode_outcomes

        if len(recent) == 0:
            return {
                'wins': 0, 'losses': 0, 'draws': 0,
                'win_rate': 0.0, 'loss_rate': 0.0, 'draw_rate': 0.0,
                'total': 0
            }

        wins = sum(1 for o in recent if o > 0)
        losses = sum(1 for o in recent if o < 0)
        draws = sum(1 for o in recent if o == 0)
        total = len(recent)

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / total if total > 0 else 0.0,
            'loss_rate': losses / total if total > 0 else 0.0,
            'draw_rate': draws / total if total > 0 else 0.0,
            'total': total
        }

    def print_performance_stats(self, window=None, prefix="", current_entropy=None, current_lr=None, current_temp=None):
        """Print performance statistics in a readable format."""
        stats = self.get_performance_stats(window)
        entropy_str = f" | Entropy={current_entropy:.4f}" if current_entropy is not None else ""
        lr_str = f" | LR={current_lr:.6f}" if current_lr is not None else ""
        temp_str = f" | Temp={current_temp:.3f}" if current_temp is not None else ""
        print(f"{prefix}Episodes: {self.episode_index} | "
              f"Recent {stats['total']}: "
              f"W={stats['wins']} ({stats['win_rate']*100:.1f}%) | "
              f"L={stats['losses']} ({stats['loss_rate']*100:.1f}%) | "
              f"D={stats['draws']} ({stats['draw_rate']*100:.1f}%){entropy_str}{lr_str}{temp_str}")

    @staticmethod
    def _sanitize_metric_value(value):
        """Convert numpy scalars to native Python types for JSON logging."""
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        return value

    def _record_training_metrics(self, metrics: Dict[str, Any]):
        """Store per-episode diagnostics for later analysis."""
        if metrics is None:
            return

        sanitized = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                sanitized[key] = {
                    k: self._sanitize_metric_value(v) for k, v in value.items()
                }
            else:
                sanitized[key] = self._sanitize_metric_value(value)

        sanitized.setdefault("timestamp", time.time())
        self.training_metrics_history.append(sanitized)
        if len(self.training_metrics_history) > self._max_metrics_history:
            self.training_metrics_history = self.training_metrics_history[-self._max_metrics_history:]
        self._last_episode_metrics = sanitized

    def get_training_metric_summary(self, window=100):
        """Return rolling averages for key training diagnostics."""
        if not self.training_metrics_history:
            return None

        window = max(1, window)
        recent = self.training_metrics_history[-window:]
        summary = {"count": len(recent)}

        def _maybe_avg(field):
            values = [entry[field]
                      for entry in recent if entry.get(field) is not None]
            if values:
                summary[f"{field}_avg"] = float(np.mean(values))

        for field in ("policy_loss", "value_loss", "entropy_term", "mean_reward", "std_reward", "episode_length"):
            _maybe_avg(field)

        draws = sum(1 for entry in recent if float(
            entry.get("outcome", 0.0)) == 0.0)
        summary["draw_rate"] = draws / len(recent)
        summary["latest_random_eval"] = self._last_random_eval
        return summary

    def attach_random_eval(self, episode_idx: int, stats: Dict[str, Any]):
        """Annotate the most recent episode metrics with random-opponent evaluation results."""
        if stats is None:
            return

        eval_record = {
            "episode": int(episode_idx),
            "wins": int(stats.get("wins", 0)),
            "losses": int(stats.get("losses", 0)),
            "draws": int(stats.get("draws", 0)),
            "total_games": int(stats.get("total_games", 0)),
            "win_rate": float(stats.get("win_rate", 0.0)),
            "loss_rate": float(stats.get("loss_rate", 0.0)),
            "draw_rate": float(stats.get("draw_rate", 0.0)),
        }
        self._last_random_eval = eval_record

        for entry in reversed(self.training_metrics_history):
            if entry.get("episode") == episode_idx:
                entry["random_eval_win_rate"] = eval_record["win_rate"]
                entry["random_eval_loss_rate"] = eval_record["loss_rate"]
                entry["random_eval_draw_rate"] = eval_record["draw_rate"]
                entry["random_eval_games"] = eval_record["total_games"]
                break

    def run_brain_client(
        self,
        host: str = "localhost",
        port: int = 6000,
        authkey: bytes = b"brain-secret",
        dt: float = 0.02,
        save_path: Optional[str] = None,
        save_every: Optional[int] = None,
        max_episodes: Optional[int] = None,
        stats_every: int = 100,
        entropy_start: Optional[float] = None,
        entropy_end: float = 0.0,
        entropy_decay_episodes: Optional[int] = None,
        entropy_decay_type: str = 'linear',
        lr_start: Optional[float] = None,
        lr_end: Optional[float] = None,
        lr_decay_episodes: Optional[int] = None,
        lr_decay_type: str = 'linear',
        random_opponent_prob_start: float = 1.0,
        random_opponent_prob_end: float = 0.0,
        random_opponent_decay_episodes: Optional[int] = None,
        random_opponent_decay_type: str = 'linear',
        metrics_log_path: Optional[str] = None,
        metrics_window: int = 200,
        eval_games: int = 200,
        random_loss_patience: Optional[int] = None,
        random_loss_min_delta: float = 0.0,
    ):
        """
        Long-lived brain client process.
        Connects to environment server and handles message protocol.
        """
        address = (host, port)
        self.dt = dt
        t = 0.0
        last_time = time.time()
        metrics_window = max(1, metrics_window)
        if random_loss_min_delta is None:
            random_loss_min_delta = 0.0
        metrics_file = None
        last_logged_episode = 0
        best_random_loss = None
        random_loss_no_improve = 0
        early_stop_reason = None
        if metrics_log_path:
            metrics_file = Path(metrics_log_path).expanduser()
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

        def flush_metrics(last_episode_logged: int) -> int:
            if metrics_file is None:
                return last_episode_logged

            pending = [
                entry for entry in self.training_metrics_history
                if entry.get("episode") is not None and entry["episode"] > last_episode_logged
            ]
            if not pending:
                return last_episode_logged

            with metrics_file.open("a", encoding="utf-8") as fh:
                for entry in pending:
                    fh.write(json.dumps(entry) + "\n")
            return pending[-1]["episode"]

        print(f"[brain] connecting to environment server at {address}")

        try:
            conn = Client(address, authkey=authkey)
            print(f"[brain] connected to environment server")
        except Exception as e:
            print(f"[brain] failed to connect to environment server: {e}")
            return

        try:
            running = True
            initial_episode = self.episode_index

            # Setup entropy decay
            if entropy_start is None:
                entropy_start = self.entropy_coeff
            if entropy_decay_episodes is None:
                entropy_decay_episodes = max_episodes if max_episodes is not None else 5000

            # Store entropy decay parameters as instance variables for access in _handle_terminal
            self._entropy_start = entropy_start
            self._entropy_end = entropy_end
            self._entropy_decay_episodes = entropy_decay_episodes
            self._entropy_initial_episode = initial_episode
            self._entropy_decay_type = entropy_decay_type

            # Print entropy decay settings
            if entropy_start != entropy_end:
                print(
                    f"[brain] Entropy decay ({entropy_decay_type}): {entropy_start:.4f} -> {entropy_end:.4f} over {entropy_decay_episodes} episodes")
            else:
                print(
                    f"[brain] Entropy coefficient: {entropy_start:.4f} (no decay)")

            # Setup learning rate decay
            if lr_start is None:
                lr_start = self.lr_policy
            if lr_end is None:
                lr_end = self.lr_policy  # No decay by default
            if lr_decay_episodes is None:
                lr_decay_episodes = max_episodes if max_episodes is not None else 5000

            # Store learning rate decay parameters as instance variables
            self._lr_start = lr_start
            self._lr_end = lr_end
            self._lr_decay_episodes = lr_decay_episodes
            self._lr_initial_episode = initial_episode
            self._lr_decay_type = lr_decay_type

            # Print learning rate decay settings
            if lr_start != lr_end:
                print(
                    f"[brain] Learning rate decay ({lr_decay_type}): {lr_start:.6f} -> {lr_end:.6f} over {lr_decay_episodes} episodes")
            else:
                print(
                    f"[brain] Learning rate: {lr_start:.6f} (no decay)")

            # Setup random opponent probability decay
            if random_opponent_decay_episodes is None:
                random_opponent_decay_episodes = max_episodes if max_episodes is not None else 5000

            # Store random opponent decay parameters
            self._random_opponent_prob_start = random_opponent_prob_start
            self._random_opponent_prob_end = random_opponent_prob_end
            self._random_opponent_decay_episodes = random_opponent_decay_episodes
            self._random_opponent_initial_episode = initial_episode
            self._random_opponent_decay_type = random_opponent_decay_type

            # Print random opponent decay settings
            if random_opponent_prob_start > 0 or random_opponent_prob_end > 0:
                if random_opponent_prob_start != random_opponent_prob_end:
                    avg_prob = (random_opponent_prob_start +
                                random_opponent_prob_end) / 2.0
                    print(
                        f"[brain] Random player probability ({random_opponent_decay_type}): "
                        f"{random_opponent_prob_start:.2%} -> {random_opponent_prob_end:.2%} "
                        f"over {random_opponent_decay_episodes} episodes "
                        f"(~{avg_prob:.1%} average - randomly choose X or O each episode)")
                else:
                    print(
                        f"[brain] Random player probability: {random_opponent_prob_start:.2%} (constant - randomly choose X or O each episode)")
            # Initialize random player flag (None = no random player, 1 = X random, -1 = O random)
            self._current_episode_random_player = None

            def get_current_entropy(episode_num):
                """Calculate current entropy coefficient based on linear decay."""
                episodes_elapsed = episode_num - initial_episode
                if episodes_elapsed >= entropy_decay_episodes:
                    return entropy_end
                progress = episodes_elapsed / entropy_decay_episodes
                return entropy_start + (entropy_end - entropy_start) * progress

            # Actively wait for initial observation from environment
            # The environment sends an observation immediately after connection
            import time as time_module
            max_wait = 5.0  # Maximum seconds to wait for initial message
            wait_start = time_module.time()
            initial_msg_received = False
            while not conn.poll(0.0):
                if time_module.time() - wait_start > max_wait:
                    print(
                        f"[brain] Timeout waiting for initial observation from environment")
                    return
                time_module.sleep(0.01)  # Small sleep to avoid busy loop

            # Process the initial observation immediately and send first action
            if conn.poll(0.0):
                try:
                    msg = conn.recv()
                    if isinstance(msg, dict) and msg.get("type") == OBSERVATION:
                        sensors = msg.get("sensors", [])
                        info = msg.get("info", {})
                        self.on_observation(sensors, info)
                        initial_msg_received = True

                        # Immediately send an action in response to initial observation
                        try:
                            action_msg = self.tick(0.0)
                            if action_msg is not None:
                                conn.send(action_msg)
                        except Exception as e:
                            print(f"[brain] Error sending initial action: {e}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"[brain] Error receiving initial observation: {e}")
                    import traceback
                    traceback.print_exc()

            while running:
                now = time.time()
                elapsed = now - last_time

                # Check max episodes limit
                if max_episodes is not None and self.episode_index >= initial_episode + max_episodes:
                    print(
                        f"\n[brain] Reached max episodes limit ({max_episodes})")
                    running = False
                    break

                # Process incoming messages FIRST (before sending actions)
                # This ensures we receive observations and can respond immediately
                message_processed = False
                while conn.poll(0.0):
                    try:
                        msg = conn.recv()
                        if not isinstance(msg, dict):
                            continue

                        mtype = msg.get("type")

                        if mtype == OBSERVATION:
                            sensors = msg.get("sensors", [])
                            info = msg.get("info", {})
                            self.on_observation(sensors, info)
                            message_processed = True

                        elif mtype == REWARD:
                            value = float(msg.get("value", 0.0))
                            info = msg.get("info", {})
                            self.on_reward(value, info)
                            message_processed = True

                        elif mtype == TERMINAL:
                            info = msg.get("info", {})
                            self._handle_terminal(info)

                            # Periodic performance statistics
                            if self.episode_index % stats_every == 0:
                                current_entropy = self._get_current_entropy()
                                current_lr = self._get_current_learning_rate()
                                current_temp = self._get_current_temperature()
                                self.print_performance_stats(
                                    window=self.stats_window,
                                    current_entropy=current_entropy,
                                    current_lr=current_lr,
                                    current_temp=current_temp)

                                summary_window = max(
                                    1, min(metrics_window, self.stats_window))
                                metric_summary = self.get_training_metric_summary(
                                    summary_window)
                                if metric_summary:
                                    policy_loss_avg = metric_summary.get(
                                        "policy_loss_avg")
                                    value_loss_avg = metric_summary.get(
                                        "value_loss_avg")
                                    entropy_avg = metric_summary.get(
                                        "entropy_term_avg")
                                    draw_rate_pct = metric_summary.get(
                                        "draw_rate", 0.0) * 100.0
                                    payload = []
                                    if policy_loss_avg is not None:
                                        payload.append(
                                            f"PolicyLoss={policy_loss_avg:.4f}")
                                    if value_loss_avg is not None:
                                        payload.append(
                                            f"ValueLoss={value_loss_avg:.4f}")
                                    if entropy_avg is not None:
                                        payload.append(
                                            f"EntropyTerm={entropy_avg:.4f}")
                                    payload.append(
                                        f"SelfPlayDraws={draw_rate_pct:.1f}% (n={metric_summary['count']})")
                                    print("  | " + " | ".join(payload))

                                # Also test against random player periodically to show improvement
                                # (self-play stats don't show improvement since both players learn)
                                if not hasattr(self, '_last_test_episode'):
                                    self._last_test_episode = 0

                                if self.episode_index - self._last_test_episode >= stats_every:
                                    try:
                                        from train_brain_tictactoe import test_against_random
                                        test_stats = test_against_random(
                                            self, num_games=eval_games)
                                        self.attach_random_eval(
                                            self.episode_index, test_stats)
                                        print(f"  | Test vs Random: W={test_stats['wins']} ({test_stats['win_rate']*100:.1f}%) "
                                              f"L={test_stats['losses']} ({test_stats['loss_rate']*100:.1f}%) "
                                              f"D={test_stats['draws']} ({test_stats['draw_rate']*100:.1f}%) "
                                              f"over {test_stats['total_games']} games")
                                        current_loss = test_stats.get(
                                            'loss_rate')
                                        if random_loss_patience:
                                            if best_random_loss is None or (best_random_loss - current_loss) >= random_loss_min_delta:
                                                if best_random_loss is None or current_loss < best_random_loss:
                                                    best_random_loss = current_loss
                                                random_loss_no_improve = 0
                                            else:
                                                random_loss_no_improve += 1
                                                if random_loss_no_improve >= random_loss_patience:
                                                    early_stop_reason = (
                                                        f"random loss failed to improve by {random_loss_min_delta:.3f} "
                                                        f"over {random_loss_patience} evaluations (loss={current_loss*100:.2f}%)"
                                                    )
                                                    print(
                                                        f"[brain] Early stopping: {early_stop_reason}")
                                                    running = False
                                                    break
                                        self._last_test_episode = self.episode_index
                                    except Exception as e:
                                        # Don't fail if test function isn't available
                                        pass

                                last_logged_episode = flush_metrics(
                                    last_logged_episode)

                            # Periodic saving
                            if save_path and save_every and self.episode_index % save_every == 0:
                                try:
                                    self.save(save_path)
                                except Exception as e:
                                    print(f"[brain] Failed to save: {e}")
                            message_processed = True

                        elif mtype == SHUTDOWN:
                            print("[brain] received shutdown message")
                            conn.close()
                            return

                    except (EOFError, OSError) as e:
                        print(f"[brain] connection lost: {e}")
                        break
                    except Exception as e:
                        import traceback
                        print(f"[brain] Error processing message: {e}")
                        traceback.print_exc()
                        break

                # After processing messages, check if we need to send an action
                if not running:
                    break
                # Check immediately if we just processed a message, or wait for dt interval
                if message_processed or elapsed >= self.dt:
                    if elapsed >= self.dt:
                        last_time = now
                        t += elapsed

                    # Check if we need to send an action
                    try:
                        action_msg = self.tick(
                            elapsed if elapsed >= self.dt else 0.0)
                        if action_msg is not None:
                            try:
                                conn.send(action_msg)
                            except (EOFError, OSError) as e:
                                print(
                                    f"[brain] connection lost while sending action: {e}")
                                break
                    except Exception as e:
                        import traceback
                        print(f"[brain] Error in tick(): {e}")
                        traceback.print_exc()
                        break

                # Small sleep to avoid busy loop
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[brain] interrupted by user")
        finally:
            try:
                conn.close()
            except Exception:
                pass

            last_logged_episode = flush_metrics(last_logged_episode)

            if early_stop_reason:
                print(f"[brain] Training stopped early: {early_stop_reason}")

            # Print final performance summary
            if len(self.episode_outcomes) > 0:
                print("\n" + "=" * 70)
                print("Final Performance Summary")
                print("=" * 70)
                self.print_performance_stats(window=None, prefix="")
                print("=" * 70)

            print("[brain] connection closed")

    # ----- offline replay -----

    def offline_replay(self, n_batches=10):
        """
        Replay stored experiences for slow consolidation.
        """
        if len(self.memory.buffer) == 0:
            return

        for _ in range(n_batches):
            batch = self.memory.sample(self.replay_batch_size)
            for x, z, action, stored_reward, x_next, z_next, done in batch:
                replay_reward = 0.5 * stored_reward

                # Use raw observations for policy if requested
                policy_state = x if self.use_raw_obs_for_policy else z
                policy_next_state = x_next if self.use_raw_obs_for_policy else z_next
                td_error, value, next_value = self.actor_critic.update(
                    state=policy_state,
                    action=action,
                    reward=replay_reward,
                    next_state=policy_next_state,
                    done=done,
                    neuromodulators=self.neuromodulators,
                    base_lr=self.lr_policy * 0.1,
                )

                neuromod_factor = (
                    0.5 * self.neuromodulators.norepinephrine
                    + 0.5 * self.neuromodulators.acetylcholine
                )
                self.world_model.learn(
                    x=x, neuromod_factor=neuromod_factor, lr_model=self.lr_model * 0.1
                )

    # ----- persistence -----

    def save(self, path):
        state = self.get_state()
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        return BrainAgent.from_state(state)

    # ----- state handling -----

    def get_state(self):
        encoder_state = self.encoder.to_state() if self.encoder is not None else None
        config = {
            "obs_dim_effective": self.world_model.layers[0].input_dim,
            "latent_dims": [layer.latent_dim for layer in self.world_model.layers],
            "n_actions": self.actor_critic.n_actions,
            "lr_model": self.lr_model,
            "lr_policy": self.lr_policy,
            "replay_batch_size": self.replay_batch_size,
            "has_encoder": self.encoder is not None,
            "use_raw_obs_for_policy": self.use_raw_obs_for_policy,
            "episode_based_learning": self.episode_based_learning,
            "entropy_coeff": self.entropy_coeff,
            # Note: functions can't be pickled, will be None on load
            "reward_shaping": self.reward_shaping,
        }
        return {
            "config": config,
            "rng_state": self.rng.get_state(),
            "encoder_state": encoder_state,
            "world_model": self.world_model.to_state(),
            "actor_critic": self.actor_critic.to_state(),
            "memory": self.memory.to_state(),
            "neuromodulators": {
                "dopamine": self.neuromodulators.dopamine,
                "norepinephrine": self.neuromodulators.norepinephrine,
                "acetylcholine": self.neuromodulators.acetylcholine,
                "ach_decay": self.neuromodulators.ach_decay,
            },
            "intrinsic": {"prev_pred_error": self.intrinsic.prev_pred_error},
            "drives": {
                "decay": self.drives.decay,
                "curiosity_drive": self.drives.curiosity_drive,
                "competence_drive": self.drives.competence_drive,
            },
            "global_step": self.global_step,
            "episode_index": self.episode_index,
        }

    @staticmethod
    def from_state(state):
        rng = np.random.RandomState()
        rng.set_state(state["rng_state"])

        encoder_state = state["encoder_state"]
        encoder = None
        if encoder_state is not None:
            from .encoder import MultiModalEncoder

            encoder = MultiModalEncoder.from_state(encoder_state)

        config = state["config"]
        obs_dim = None if encoder is not None else config["obs_dim_effective"]

        agent = BrainAgent(
            obs_dim=obs_dim,
            latent_dims=config["latent_dims"],
            n_actions=config["n_actions"],
            encoder=encoder,
            lr_model=config["lr_model"],
            lr_policy=config["lr_policy"],
            replay_batch_size=config["replay_batch_size"],
            use_raw_obs_for_policy=config.get("use_raw_obs_for_policy", False),
            episode_based_learning=config.get("episode_based_learning", False),
            entropy_coeff=config.get("entropy_coeff", 0.0),
            reward_shaping=config.get("reward_shaping", None),
            rng=rng,
        )

        agent.world_model = HierarchicalWorldModel.from_state(
            state["world_model"], rng=rng)
        agent.actor_critic = ActorCritic.from_state(state["actor_critic"])
        agent.memory = EpisodicMemory.from_state(state["memory"])

        nm = state["neuromodulators"]
        agent.neuromodulators.dopamine = nm["dopamine"]
        agent.neuromodulators.norepinephrine = nm["norepinephrine"]
        agent.neuromodulators.acetylcholine = nm["acetylcholine"]
        agent.neuromodulators.ach_decay = nm["ach_decay"]

        agent.intrinsic.prev_pred_error = state["intrinsic"]["prev_pred_error"]

        drives = state["drives"]
        agent.drives.decay = drives["decay"]
        agent.drives.curiosity_drive = drives["curiosity_drive"]
        agent.drives.competence_drive = drives["competence_drive"]

        agent.global_step = state.get("global_step", 0)
        agent.episode_index = state.get("episode_index", 0)

        return agent

    # ----- continuous life loop -----

    def run_continuous(
        self,
        env,
        max_steps=None,
        offline_every=100,
        offline_batches=10,
        save_path=None,
        save_every=None,
        temperature_schedule=None,
        greedy_after=None,
    ):
        """
        Continuous perception-action-learning loop.

        env: must implement reset() and step(action)
        max_steps: stop after this many steps (None = run until interrupted)
        offline_every: how often to call offline_replay
        save_path, save_every: periodic saving
        temperature_schedule(step) -> temperature
        greedy_after: step after which to act greedily
        """
        obs = env.reset()
        self.episode_index += 1
        step_in_episode = 0

        try:
            while True:
                if max_steps is not None and self.global_step >= max_steps:
                    break

                if temperature_schedule is None:
                    temperature = 1.0
                else:
                    temperature = float(temperature_schedule(self.global_step))

                greedy = greedy_after is not None and self.global_step >= greedy_after

                action, z, value, x = self.act(
                    obs, temperature=temperature, greedy=greedy)
                next_obs, external_reward, done, info = env.step(action)

                # Get legal mask for update
                try:
                    legal_mask = self.get_legal_actions(obs)
                except:
                    legal_mask = None

                td_error, pred_err, intrinsic, drives = self.online_update(
                    obs=obs,
                    x=x,
                    z=z,
                    action=action,
                    external_reward=external_reward,
                    next_obs=next_obs,
                    done=done,
                    legal_mask=legal_mask,
                )

                obs = next_obs
                step_in_episode += 1

                if offline_every is not None and self.global_step % offline_every == 0:
                    self.offline_replay(n_batches=offline_batches)

                if save_path is not None and save_every is not None:
                    if self.global_step % save_every == 0:
                        self.save(save_path)

                if done:
                    obs = env.reset()
                    self.episode_index += 1
                    step_in_episode = 0
                    self.reset_episode()

        except KeyboardInterrupt:
            if save_path is not None:
                self.save(save_path)
