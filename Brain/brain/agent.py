from brainprotocol import (
    OBSERVATION, REWARD, ACTION, TERMINAL, SHUTDOWN,
    DISCOVERY_STARTUP, DISCOVERY_SHUTDOWN, DISCOVERY_ANNOUNCE,
    DISCOVERY_LIST_PEERS, DISCOVERY_PEER_LIST,
    PEER_TYPE_BRAIN, PEER_TYPE_ENVIRONMENT
)
import numpy as np
import pickle
import time
import sys
import math
import json
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque
from multiprocessing.connection import Client
from .world_model import HierarchicalWorldModel
from .actor_critic import ActorCritic
from .memory import Hippocampus
from .neuromodulators import NeuromodulatorState
from .motivation import IntrinsicMotivation, DriveState
from .connection_manager import ConnectionManager
from .connection_config import BrainConnectionConfig

# Import protocol constants
sys.path.insert(0, str(Path(__file__).parent.parent))


class BrainAgent:
    """
    Integrated biologically-inspired agent:
    - Hierarchical predictive coding world model (Sensory Cortex)
    - Actor-critic RL (Basal Ganglia / Prefrontal Cortex)
    - Complex Neuromodulators (DA, NE, 5-HT, ACh, Cortisol)
    - Intrinsic motivation (Curiosity, Competence, Autonomy)
    - Homeostatic drives (Energy, Boredom)
    - Hippocampal memory system (STM -> Episodic/LTM)
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
        supervised_loss_coeff=5.0,  # Increased default to make forced action learning stronger
        mode_dim=8,
        z_mode_sigma_base=1.0,
        k_z=0.5,
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

        # Larger default network
        if latent_dims is None:
            latent_dims = [128, 64, 32]

        self.world_model = HierarchicalWorldModel(
            obs_dim_effective, latent_dims, rng=rng)
        top_dim = latent_dims[-1]
        # Use raw observations for policy if requested (more stable for simple tasks)
        self.use_raw_obs_for_policy = use_raw_obs_for_policy
        # Mode variable for behavioral diversity
        self.mode_dim = mode_dim if mode_dim > 0 else 0
        self.z_mode_sigma_base = z_mode_sigma_base
        self.k_z = k_z
        self.current_z_mode = None  # Current episode's mode variable

        # Policy state dimension includes mode if enabled
        base_policy_state_dim = obs_dim_effective if use_raw_obs_for_policy else top_dim
        policy_state_dim = base_policy_state_dim + self.mode_dim

        self.actor_critic = ActorCritic(
            policy_state_dim, n_actions, rng=rng, entropy_coeff=entropy_coeff)

        # Replaces simple Memory
        self.hippocampus = Hippocampus(
            stm_capacity=10, ltm_capacity=100000, rng=rng)

        self.neuromodulators = NeuromodulatorState()
        self.intrinsic = IntrinsicMotivation()
        self.drives = DriveState()

        self.lr_model = lr_model
        self.lr_policy = lr_policy
        self.replay_batch_size = replay_batch_size
        self.episode_based_learning = episode_based_learning
        self.entropy_coeff = entropy_coeff
        self.reward_shaping = reward_shaping
        self.supervised_loss_coeff = supervised_loss_coeff

        # Episode buffer for episode-based learning
        self.episode_buffer = []

        self.global_step = 0
        self.episode_index = 0

        # Performance tracking
        self.episode_outcomes = []
        self.stats_window = 100
        self.training_metrics_history = []
        self._max_metrics_history = 10000
        self._last_random_eval = None
        self._last_episode_metrics = None

        # Client protocol state
        self.pending_decision = False
        self.last_sensors = None
        self.last_legal_actions = None
        self.cum_reward = 0.0
        self.last_action = None
        self.dt = 0.02

        # Self-play episode state
        self.episode_board = None
        self.episode_player = None
        self.episode_states = []
        self.episode_actions = []
        self.episode_players = []
        self.episode_masks = []

        # Two-brain mode: track which player this brain is
        self.my_player_symbol = None  # "X" or "O"

        # Competence tracking (continuous self-regulation)
        self.competence = 0.0  # Initialized to 0 (not competent yet)
        self.competence_alpha = 0.01  # Smoothing factor for competence updates
        self.competence_window_size = 200  # Window for computing moving averages
        # Moving average trackers
        self.da_error_window = deque(maxlen=self.competence_window_size)
        self.ne_error_window = deque(maxlen=self.competence_window_size)
        self.loss_rate_window = deque(maxlen=self.competence_window_size)
        # Competence parameters
        self.competence_w1 = 0.4  # Weight for DA in stability calculation
        self.competence_w2 = 0.3  # Weight for NE in stability calculation
        self.competence_w3 = 0.3  # Weight for loss_rate in stability calculation
        self.competence_sigmoid_a = 10.0  # Sigmoid parameter a
        self.competence_sigmoid_b = -0.5  # Sigmoid parameter b
        self._pending_episode = None
        self._pending_turn = None

    # ----- input handling -----

    def _prepare_obs(self, obs):
        if self.encoder is not None:
            if not isinstance(obs, dict):
                raise ValueError(
                    "With encoder set, obs must be dict of modality->array.")
            return self.encoder.encode(obs)
        else:
            return np.asarray(obs).reshape(-1)

    def encode_state(self, obs):
        x = self._prepare_obs(obs)
        z = self.world_model.encode_state(x)
        return z, x

    # ----- legal actions -----

    def get_legal_actions(self, obs):
        """
        DEPRECATED: Fallback method.
        """
        x = self._prepare_obs(obs)
        legal_mask = np.zeros(self.actor_critic.n_actions)
        for i in range(min(len(x), self.actor_critic.n_actions)):
            if abs(x[i]) > 1e-6:  # Position is occupied
                legal_mask[i] = float('-inf')
        return legal_mask

    # ----- acting -----

    def get_competence_gated_lr(self, base_lr, lr_min=1e-5, lr_plastic=None):
        """
        Get learning rate gated by competence signal.
        When competence is high (C_t → 1), LR → lr_min.
        When competence is low (C_t → 0), LR → lr_plastic * neuromodulator_factor.
        """
        if lr_plastic is None:
            lr_plastic = base_lr
        lr_mod_ne_da = self.neuromodulators.get_learning_rate_factor()
        lr = lr_min + (1.0 - self.competence) * lr_plastic * lr_mod_ne_da
        return np.clip(lr, lr_min, base_lr * 5.0)  # Clamp to reasonable range

    def get_competence_gated_entropy(self, base_entropy, entropy_min=0.0, entropy_plastic=None):
        """
        Get entropy coefficient gated by competence signal.
        When competence is high (C_t → 1), entropy → entropy_min.
        When competence is low (C_t → 0), entropy → entropy_plastic * neuromodulator_factor.
        """
        if entropy_plastic is None:
            entropy_plastic = base_entropy
        entropy_mod_ne_da = self.neuromodulators.get_exploration_entropy()
        entropy = entropy_min + (1.0 - self.competence) * \
            entropy_plastic * entropy_mod_ne_da
        # Clamp to reasonable range
        return np.clip(entropy, entropy_min, base_entropy * 5.0)

    def get_competence_gated_z_sigma(self, z_sigma_base, z_sigma_min=0.0, z_sigma_plastic=None):
        """
        Get mode variable sigma gated by competence signal.
        When competence is high (C_t → 1), z_sigma → z_sigma_min (deterministic).
        When competence is low (C_t → 0), z_sigma → z_sigma_plastic.
        """
        if z_sigma_plastic is None:
            z_sigma_plastic = z_sigma_base
        z_sigma = z_sigma_min + (1.0 - self.competence) * z_sigma_plastic
        return max(z_sigma, z_sigma_min)

    def update_competence(self):
        """
        Update competence signal based on moving averages of DA, NE, and loss rate.
        Uses loss rate against opponents (not random players) as primary signal when available.
        Called at end of each episode.
        """
        if len(self.da_error_window) == 0 or len(self.ne_error_window) == 0 or len(self.loss_rate_window) == 0:
            # Not enough data yet, keep competence at 0
            return

        # Compute moving averages for internal metrics
        bar_abs_da = np.mean(list(self.da_error_window))
        bar_ne = np.mean(list(self.ne_error_window))
        loss_rate = np.mean(list(self.loss_rate_window))

        # Check if we're playing against an opponent (two-brain mode)
        # Only use opponent performance, not random player performance
        opponent_loss_rate = None
        if self.my_player_symbol is not None:
            # Two-brain mode: playing against opponent
            # Use loss rate directly: 0.0 (no losses) -> high competence, 1.0 (all losses) -> low competence
            opponent_loss_rate = loss_rate
        elif len(self.episode_outcomes) > 0:
            # Self-play mode: use loss rate from episode outcomes
            # In self-play, we track draws as "not competent" but this is less reliable
            # So we'll weight it less heavily
            recent_outcomes = self.episode_outcomes[-self.competence_window_size:] if len(
                self.episode_outcomes) >= self.competence_window_size else self.episode_outcomes
            if len(recent_outcomes) > 0:
                # Loss rate in self-play: losses + draws count as "not winning"
                losses = sum(1 for o in recent_outcomes if o < 0)
                draws = sum(1 for o in recent_outcomes if o == 0)
                total = len(recent_outcomes)
                # Count draws as partial losses (0.5 weight)
                opponent_loss_rate = (losses + 0.5 * draws) / \
                    total if total > 0 else 0.0

        if opponent_loss_rate is not None:
            # Use opponent loss rate as primary signal (weighted average with internal metrics)
            # Map loss rate to competence: 0.0 (no losses) -> 1.0, 1.0 (all losses) -> 0.0
            opponent_performance = 1.0 - opponent_loss_rate
            opponent_performance = np.clip(opponent_performance, 0.0, 1.0)

            # Calculate internal error signal
            err = (self.competence_w1 * bar_abs_da +
                   self.competence_w2 * bar_ne +
                   self.competence_w3 * loss_rate)

            # Map internal metrics to stability using sigmoid
            sigmoid_val = 1.0 / \
                (1.0 + np.exp(self.competence_sigmoid_a *
                 err + self.competence_sigmoid_b))
            internal_stability = 1.0 - sigmoid_val
            internal_stability = np.clip(internal_stability, 0.0, 1.0)

            # Combine opponent performance (70% weight) with internal stability (30% weight)
            # Opponent performance is more reliable indicator of actual competence
            stability = 0.7 * opponent_performance + 0.3 * internal_stability
        else:
            # No opponent performance data, use internal metrics only
            # But be more conservative - self-play metrics can be misleading
            # Calculate error signal
            err = (self.competence_w1 * bar_abs_da +
                   self.competence_w2 * bar_ne +
                   self.competence_w3 * loss_rate)

            # Map to stability using sigmoid: S = 1 - sigmoid(a * err + b)
            # When err is low (good performance), S → 1
            # When err is high (poor performance), S → 0
            # Use more conservative sigmoid parameters for self-play only
            sigmoid_val = 1.0 / \
                (1.0 + np.exp(self.competence_sigmoid_a * err +
                 self.competence_sigmoid_b + 1.0))  # +1.0 makes it more conservative
            stability = 1.0 - sigmoid_val
            stability = np.clip(stability, 0.0, 1.0)
            # Further reduce self-play-only competence by 30% to account for self-play bias
            stability = stability * 0.7

        # Update competence with exponential moving average
        self.competence = (1.0 - self.competence_alpha) * \
            self.competence + self.competence_alpha * stability

    def _ensure_mode_initialized(self):
        """
        Ensure current_z_mode is initialized if mode_dim > 0.
        This prevents dimension mismatches when act() is called before on_observation().
        """
        if self.mode_dim > 0 and self.current_z_mode is None:
            # Initialize with competence-gated mode (same logic as on_observation)
            # Mode variance is gated by competence
            # When competent (C_t → 1), z_sigma → 0 (deterministic)
            # When not competent (C_t → 0), z_sigma → base value
            z_sigma = self.get_competence_gated_z_sigma(self.z_mode_sigma_base)
            # Sample z_mode ~ N(0, z_sigma^2 * I)
            self.current_z_mode = self.rng.normal(
                0.0, z_sigma, size=self.mode_dim).astype(np.float32)

    def act(self, obs, temperature=1.0, greedy=False, legal_mask=None):
        z, x = self.encode_state(obs)
        policy_state = x if self.use_raw_obs_for_policy else z

        # Ensure mode is initialized if needed
        self._ensure_mode_initialized()

        # Concatenate mode variable if enabled
        if self.mode_dim > 0:
            policy_state = np.concatenate([policy_state, self.current_z_mode])

        # Hippocampal Retrieval (Context)
        # In a more advanced version, this would be appended to policy_state
        # or used to bias action selection.
        # retrieved_memories = self.hippocampus.recall(policy_state)
        # if retrieved_memories:
        #     pass # Could use this for "one-shot" adaptation

        if legal_mask is None:
            try:
                legal_mask = self.get_legal_actions(obs)
            except:
                legal_mask = None

        action, probs, value = self.actor_critic.act(
            policy_state, temperature=temperature, greedy=greedy, legal_mask=legal_mask
        )
        return action, z, value, x

    # ----- online learning -----

    def online_update(self, obs, x, z, action, external_reward, next_obs, done, legal_mask=None):
        z_next, x_next = self.encode_state(next_obs)

        if self.reward_shaping is not None:
            external_reward = self.reward_shaping(external_reward)

        # 1. World Model Learning
        # Modulated by NE (surprise) and ACh (uncertainty)
        neuromod_factor = (
            0.5 * self.neuromodulators.norepinephrine
            + 0.5 * self.neuromodulators.acetylcholine
        )
        # Clamp factor
        neuromod_factor = np.clip(neuromod_factor, 0.1, 2.0)

        _, pred_error_norm = self.world_model.learn(
            x=x, neuromod_factor=neuromod_factor, lr_model=self.lr_model
        )

        # 2. Intrinsic Motivation
        entropy_of_policy = 0.0  # Could calculate from AC
        intrinsic_reward, components = self.intrinsic.compute(
            pred_error_norm, entropy_of_policy)

        # 3. Update Drives (Homeostasis)
        drive_vec = self.drives.update(components, external_reward)

        # 4. Calculate Total Reward
        # External reward scaled by biological need (Drive Multiplier)
        drive_gain = self.drives.get_drive_multiplier()
        total_reward = float(external_reward) * drive_gain + intrinsic_reward

        # 5. Get Value Estimates for Neuromodulators
        policy_state = x if self.use_raw_obs_for_policy else z
        policy_next_state = x_next if self.use_raw_obs_for_policy else z_next

        # Ensure mode is initialized if needed
        self._ensure_mode_initialized()

        # Concatenate mode variable if enabled
        if self.mode_dim > 0:
            policy_state = np.concatenate([policy_state, self.current_z_mode])
            policy_next_state = np.concatenate(
                [policy_next_state, self.current_z_mode])

        # We need values to compute RPE
        # Note: act() returns value, but we need it here cleanly
        val_curr, _ = self.actor_critic._forward_value(policy_state)
        val_next, _ = self.actor_critic._forward_value(policy_next_state)
        val_curr = val_curr.item() if hasattr(val_curr, 'item') else float(val_curr)
        val_next = val_next.item() if hasattr(val_next, 'item') else float(val_next)
        if done:
            val_next = 0.0

        # 6. Update Neuromodulators
        self.neuromodulators.update(
            reward=total_reward,
            value=val_curr,
            next_value=val_next,
            pred_error_norm=pred_error_norm,
            effort=0.01,  # Action cost
            punishment=0.0  # Could be inferred from negative reward
        )

        # Track DA and NE for competence calculation
        self.da_error_window.append(abs(self.neuromodulators.dopamine))
        self.ne_error_window.append(self.neuromodulators.norepinephrine)

        if self.episode_based_learning:
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
                'value': val_curr,
                'legal_mask': legal_mask,
            })

            if done:
                self.update_from_episode()
            td_error = 0.0
        else:
            # Standard TD update
            # AC uses Neuromodulators for LR and Gamma scaling
            td_error, _, _ = self.actor_critic.update(
                state=policy_state,
                action=action,
                reward=total_reward,
                next_state=policy_next_state,
                done=done,
                neuromodulators=self.neuromodulators,
                base_lr=self.lr_policy,
                legal_mask=legal_mask,
                entropy_coeff=self.entropy_coeff,
                supervised_loss_coeff=self.supervised_loss_coeff,
            )

        # 7. Memory Processing (Hippocampus)
        transition = (x, z, action, total_reward, x_next, z_next, done)
        self.hippocampus.process_experience(transition, self.neuromodulators)

        self.global_step += 1

        return td_error, pred_error_norm, intrinsic_reward, drive_vec

    def update_from_episode(self):
        if len(self.episode_buffer) == 0:
            return

        states = []
        actions = []
        rewards = []
        legal_masks = []

        for transition in self.episode_buffer:
            x = transition['x']
            z = transition['z']
            action = transition['action']
            legal_mask = transition.get('legal_mask')

            policy_state = x if self.use_raw_obs_for_policy else z
            # Ensure mode is initialized if needed
            self._ensure_mode_initialized()
            # Concatenate mode variable if enabled
            if self.mode_dim > 0:
                policy_state = np.concatenate(
                    [policy_state, self.current_z_mode])
            reward = transition['total_reward']  # Use biological reward

            states.append(policy_state)
            actions.append(action)
            rewards.append(reward)
            legal_masks.append(legal_mask)

        # Use competence-gated LR and Entropy for episode update
        current_lr = self.get_competence_gated_lr(self.lr_policy)
        current_entropy = self.get_competence_gated_entropy(self.entropy_coeff)

        self.actor_critic.update_reinforce(
            states=states,
            actions=actions,
            rewards=rewards,
            legal_masks=legal_masks if any(
                m is not None for m in legal_masks) else None,
            entropy_coeff=current_entropy,
            lr=current_lr,
            supervised_loss_coeff=self.supervised_loss_coeff,
        )

        self.episode_buffer = []
        # Consolidate STM to LTM at end of episode
        self.hippocampus.consolidate_stm()

    def reset_episode(self):
        if self.episode_based_learning:
            self.episode_buffer = []
        # Clear STM
        self.hippocampus.stm.clear()

    # ----- client protocol methods -----

    def on_observation(self, sensors, info):
        episode_num = info.get("episode", 0)
        # Update episode_index to match environment's episode number
        # This ensures we track episodes correctly even if brain was loaded with a higher episode_index
        if episode_num > 0 and episode_num != self.episode_index:
            # If this is a new episode (higher number), reset episode state
            if episode_num > self.episode_index:
                self.episode_index = episode_num
                self.episode_board = None
                self.episode_player = None
                self.episode_states = []
                self.episode_actions = []
                self.episode_players = []
                self.episode_masks = []
                self.cum_reward = 0.0
                self._current_episode_random_player = None
                # Also clear STM on new episode if not done already
                self.hippocampus.stm.clear()
            else:
                # Environment episode number is lower (e.g., environment restarted)
                # Update to match environment but don't reset state
                self.episode_index = episode_num
            self.episode_board = None
            self.episode_player = None
            self.episode_states = []
            self.episode_actions = []
            self.episode_players = []
            self.episode_masks = []
            self.cum_reward = 0.0
            self._current_episode_random_player = None
            # Also clear STM on new episode if not done already
            self.hippocampus.stm.clear()

            # Sample new mode variable for this episode
            if self.mode_dim > 0:
                # Mode variance is gated by competence
                # When competent (C_t → 1), z_sigma → 0 (deterministic)
                # When not competent (C_t → 0), z_sigma → base value
                z_sigma = self.get_competence_gated_z_sigma(
                    self.z_mode_sigma_base)
                # Sample z_mode ~ N(0, z_sigma^2 * I)
                self.current_z_mode = self.rng.normal(
                    0.0, z_sigma, size=self.mode_dim).astype(np.float32)
            else:
                self.current_z_mode = None

        self.last_sensors = list(sensors) if sensors is not None else []
        self._last_observation_info = info

        # Store player identity from observation info (for two-brain mode)
        if "player" in info:
            self.my_player_symbol = info["player"]  # "X" or "O"

        if "legal_actions" in info:
            legal_actions = info["legal_actions"]
            if isinstance(legal_actions, list):
                self.last_legal_actions = np.array(
                    legal_actions, dtype=np.float32)
            else:
                self.last_legal_actions = legal_actions
        else:
            self.last_legal_actions = None

        # ---- NEW: strict decision context ----
        episode = info.get("episode")
        turn = info.get("current_turn")

        # Only allow an action if this observation defines a valid decision context
        if episode is not None and turn is not None:
            self._pending_episode = episode
            self._pending_turn = turn
            self.pending_decision = True
        else:
            # Observation without decision semantics -> no action allowed
            self.pending_decision = False

    def on_reward(self, value: float, info):
        """
        Process reward from environment.

        Important: do NOT set pending_decision here.
        The brain should only emit a new ACTION when it receives a new OBSERVATION.
        Rewards update internal state (cum_reward, learning), but do not by themselves
        require an immediate action.

        This guarantees:
        - exactly one action per observation, and
        - no stale actions caused by reward messages arriving between episodes.
        """
        self.cum_reward += float(value)
        # Intentionally do NOT set self.pending_decision here.
        # The next decision will be triggered by the next on_observation().

    def tick(self, dt: float) -> Optional[Dict[str, Any]]:
        if not self.pending_decision:
            return None

        info = getattr(self, "_last_observation_info", {})
        episode = info.get("episode")
        turn = info.get("current_turn")

        # Drop decision if context changed
        if episode != self._pending_episode or turn != self._pending_turn:
            self.pending_decision = False
            return None

        if self.last_sensors is None or len(self.last_sensors) == 0:
            return None

        obs = np.array(self.last_sensors, dtype=np.float32)

        prev_board = self.episode_board.copy() if self.episode_board is not None else None
        self.episode_board = obs.copy()

        x_count = np.sum(self.episode_board == 1)
        o_count = np.sum(self.episode_board == -1)
        self.episode_player = 1 if x_count == o_count else -1

        # In two-brain mode, only act when it's this brain's turn
        if self.my_player_symbol is not None:
            # Check if observation info tells us whose turn it is
            info = getattr(self, '_last_observation_info', {})
            current_turn_from_env = info.get("current_turn")

            if current_turn_from_env is not None:
                # Environment explicitly tells us whose turn it is
                if current_turn_from_env != self.my_player_symbol:
                    # Not this brain's turn, don't send action
                    # Clear pending_decision to avoid repeatedly checking
                    self.pending_decision = False
                    # Debug: log why we're not acting (only once per episode)
                    episode = info.get("episode", 0)
                    log_key = f'_turn_check_logged_ep{episode}'
                    if not hasattr(self, log_key):
                        print(
                            f"[brain] tick() returning None: current_turn={current_turn_from_env} != my_player={self.my_player_symbol}, episode={episode}")
                        setattr(self, log_key, True)
                    return None
                else:
                    # It's our turn - clear the log flag for this episode
                    episode = info.get("episode", 0)
                    log_key = f'_turn_check_logged_ep{episode}'
                    if hasattr(self, log_key):
                        delattr(self, log_key)
            else:
                # No current_turn in info - use fallback logic
                # Fallback: determine from board state
                current_turn_player = "X" if self.episode_player == 1 else "O"
                if current_turn_player != self.my_player_symbol:
                    # Not this brain's turn, don't send action
                    # Clear pending_decision to avoid repeatedly checking
                    self.pending_decision = False
                    return None

        if self.last_legal_actions is not None:
            legal_mask = self.last_legal_actions
            if isinstance(legal_mask, np.ndarray):
                if np.all(np.isinf(legal_mask) & (legal_mask < 0)):
                    legal_mask = None
        else:
            try:
                legal_mask = self.get_legal_actions(obs)
            except:
                legal_mask = None

        is_first_move = (x_count == 0 and o_count == 0)
        if is_first_move:
            random_opponent_prob = self._get_random_opponent_probability()
            if self.rng.random() < random_opponent_prob:
                self._current_episode_random_player = self.rng.choice([1, -1])
            else:
                self._current_episode_random_player = None

        use_random_action = (self.episode_player == getattr(
            self, '_current_episode_random_player', None))

        if use_random_action:
            if legal_mask is not None:
                if isinstance(legal_mask, np.ndarray):
                    legal_actions = np.where(np.isfinite(
                        legal_mask) & (legal_mask >= 0))[0]
                else:
                    legal_actions = [i for i in range(len(legal_mask)) if np.isfinite(
                        legal_mask[i]) and legal_mask[i] >= 0]
            else:
                legal_actions = np.where(self.episode_board == 0)[0]

            if len(legal_actions) > 0:
                action = int(self.rng.choice(legal_actions))
                z, x = self.encode_state(obs)
                value = None
            else:
                action, z, value, x = self.act(
                    obs, temperature=1.0, greedy=False, legal_mask=legal_mask)
            record_transition = False
        else:
            current_entropy = self._get_current_entropy()
            # Modulate temperature with NM
            temperature = self._get_current_temperature()
            # We might also want to use NM exploration factor
            if hasattr(self.neuromodulators, 'get_exploration_entropy'):
                # Higher entropy -> needs higher temp?
                # Currently temp is derived from entropy schedule.
                # We will trust the NM modulation in update_reinforce/update.
                pass

            greedy = (current_entropy < 0.0001)

            action, z, value, x = self.act(
                obs, temperature=temperature, greedy=greedy, legal_mask=legal_mask)
            record_transition = True

        if self.episode_based_learning and record_transition:
            # Store policy state with mode variable included
            policy_state = x if self.use_raw_obs_for_policy else z
            # Ensure mode is initialized if needed
            self._ensure_mode_initialized()
            if self.mode_dim > 0:
                policy_state_with_mode = np.concatenate(
                    [policy_state, self.current_z_mode])
            else:
                policy_state_with_mode = policy_state
            self.episode_states.append(
                policy_state_with_mode.copy() if isinstance(policy_state_with_mode, np.ndarray) else policy_state_with_mode)
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
            "info": {
                "t": time.time(),
                "episode": self._pending_episode,
                "turn": self._pending_turn,
                "cum_reward": self.cum_reward,
            },
        }

    def _check_winner(self, board):
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
        if len(self.episode_states) > 0:
            winner = info.get("winner")
            if winner == 'X':
                outcome = 1.0
            elif winner == 'O':
                outcome = -1.0
            elif winner == 'draw':
                outcome = 0.0
            else:
                if self.episode_board is not None:
                    outcome = self._check_winner(self.episode_board)
                    if outcome is None:
                        outcome = 0.0
                    else:
                        outcome = float(outcome)
                else:
                    outcome = 0.0

            # Calculate rewards using raw outcome (from environment perspective)
            # The rewards calculation (outcome * p) works correctly because:
            # - Agent X: episode_players contains only [1, 1, ...], so rewards = outcome * 1
            # - Agent O: episode_players contains only [-1, -1, ...], so rewards = outcome * (-1)
            # This correctly gives positive rewards for wins and negative for losses
            rewards = [outcome * p for p in self.episode_players]

            # In two-brain mode, adjust logged outcome to be from this agent's perspective
            # Raw outcome: 1.0 = X wins, -1.0 = O wins, 0.0 = draw
            # For logging/statistics, we want the outcome from the agent's perspective:
            #   Agent X: 1.0 = X wins (agent won), -1.0 = O wins (agent lost)
            #   Agent O: 1.0 = O wins (agent won), -1.0 = X wins (agent lost)
            logged_outcome = outcome
            if self.my_player_symbol == 'O' and outcome != 0.0:
                logged_outcome = -outcome  # Flip for agent O's perspective

            # This triggers update_from_episode logic if buffer has items
            # But for BrainClient, we accumulate transitions in episode_buffer during play?
            # BrainClient logic is mixed here. `tick` calls `act`.
            # But `online_update` is NOT called in `tick`.
            # `online_update` is usually called in a training loop.
            # `BrainClient` (run_brain_client.py) just uses `tick` and `_handle_terminal`.
            # Wait! `BrainAgent.tick` just returns the action. It does NOT do learning.
            # `run_brain_client.py` relies on `_handle_terminal` to trigger training.
            # But `_handle_terminal` in original code did:
            # self.actor_critic.update_reinforce(...)

            # The `online_update` method is for `run_continuous` loop, not for the `BrainClient` protocol used in `run_brain_client.py`.
            # `run_brain_client.py` uses `tick` (inference) and `_handle_terminal` (update).
            # So I need to put my "Biological Learning" logic into `_handle_terminal` or `tick` as well!

            # Actually, for "Brain Client" (which is what the user asked to modify), I should ensure `tick` pushes to a buffer, and `_handle_terminal` processes it.
            # `tick` currently appends to `self.episode_states`.
            # I should also calculate intrinsic rewards, drives, neuromodulators during the episode?
            # Or just at the end?
            # Biological systems learn continuously.
            # If I want `BrainClient` to be "biologically plausible", it should process rewards/observations as they come in.

            # `on_observation` -> `tick` -> Action.
            # `on_reward` -> Reward.

            # I will enable "online" processing within `tick` if possible, or at least simulate it.
            # But `BrainClient` is request/response.

            # Let's stick to `_handle_terminal` doing the heavy lifting for the Episode-based TTT,
            # BUT I must update Neuromodulators/Drives during the episode simulation in `_handle_terminal`.

            # The problem: `_handle_terminal` does a batch update.
            # I should iterate through the episode and update NM/Drives step-by-step to be accurate.

            # Let's rewrite `_handle_terminal` to iterate through the episode history, calculate NM/Drive updates, and THEN do the policy update.

            update_metrics = None

            if self.episode_based_learning and len(self.episode_states) > 0:
                states_np = [np.array(s, dtype=np.float32)
                             for s in self.episode_states]

                # Replay the episode to update internal biological state (NM, Drives, Hippocampus)
                # This is "fast replay" or "consolidation" of the just-finished episode.
                # We need the full trajectory.

                final_biological_rewards = []

                for i in range(len(states_np)):
                    s = states_np[i]
                    a = self.episode_actions[i]
                    ext_r = rewards[i]

                    # Extract base state (without mode) for world model learning
                    if self.mode_dim > 0:
                        base_state_dim = len(s) - self.mode_dim
                        s_base = s[:base_state_dim]
                    else:
                        s_base = s

                    # 1. World Model (for Intrinsic)
                    # We don't have next_state easily for all steps unless we tracked it.
                    # episode_states stores state at t.
                    # We need s_{t+1}.
                    if i < len(states_np) - 1:
                        s_next = states_np[i+1]
                        done = False
                    else:
                        # Create terminal state with same shape as s (including mode if present)
                        s_next = np.zeros_like(s)
                        done = True

                    # Note: 's_base' here is the base policy_state (x or z) without mode.
                    # If x, we can use it for WM learning.
                    # If z, we can't easily learn WM (WM learns x->z).
                    # Assuming use_raw_obs_for_policy=True (default for TTT), s_base is x.

                    neuromod_factor = 1.0  # Simplified for batch replay
                    pred_error_norm = 0.0
                    if self.use_raw_obs_for_policy:
                        # If s_base is x, we can learn
                        _, pred_error_norm = self.world_model.learn(
                            s_base, neuromod_factor, self.lr_model)

                    # 2. Intrinsic
                    intr_r, comps = self.intrinsic.compute(pred_error_norm)

                    # 3. Drives
                    self.drives.update(comps, ext_r)
                    drive_gain = self.drives.get_drive_multiplier()

                    # 4. Total Reward
                    total_r = ext_r * drive_gain + intr_r
                    final_biological_rewards.append(total_r)

                    # 5. Neuromodulators
                    # Estimate values
                    # Note: s already includes mode variable if mode_dim > 0 (stored in tick)
                    v_curr, _ = self.actor_critic._forward_value(s)
                    v_next, _ = self.actor_critic._forward_value(s_next)
                    v_curr = v_curr.item()
                    v_next = v_next.item() if not done else 0.0

                    self.neuromodulators.update(
                        total_r, v_curr, v_next, pred_error_norm)

                    # Track DA and NE for competence calculation
                    self.da_error_window.append(
                        abs(self.neuromodulators.dopamine))
                    self.ne_error_window.append(
                        self.neuromodulators.norepinephrine)

                    # 6. Hippocampus
                    # Store transition
                    # We need 'obs' but we only have 'policy_state'.
                    # Approximate
                    trans = (s, s, a, total_r, s_next, s_next, done)
                    self.hippocampus.process_experience(
                        trans, self.neuromodulators)

                # Track loss rate for competence calculation
                # Determine if agent lost this episode
                # For self-play: agent plays both sides, so we track if outcome != 0 (someone won)
                # For two-brain mode: track if agent's player lost
                # Use logged_outcome for this check since it's from the agent's perspective
                agent_lost = False
                if self.my_player_symbol is not None:
                    # Two-brain mode: agent has a specific player symbol
                    # logged_outcome is already from agent's perspective: < 0 = loss, > 0 = win
                    agent_lost = (logged_outcome < 0)
                else:
                    # Self-play mode: agent plays both sides
                    # In self-play, we can't really say the agent "lost" since it plays both sides
                    # Instead, track draws as "not winning" - if outcome == 0, it's a draw
                    # For competence, we want to track when the agent fails to win (draws or losses)
                    # Since in self-play the agent is both players, we track draws as "not successful"
                    # But actually, in self-play, we should track based on whether there was a clear winner
                    # For now, let's track: loss = 1 if outcome == 0 (draw), 0 if outcome != 0 (someone won)
                    # Actually, better: track based on whether the game ended in a draw
                    # Draws indicate the agent hasn't learned optimal play yet
                    agent_lost = (outcome == 0.0)  # Draw = not competent
                self.loss_rate_window.append(1.0 if agent_lost else 0.0)

                # Now update Policy with biological rewards using competence-gated LR and entropy
                current_lr = self.get_competence_gated_lr(self.lr_policy)
                current_entropy = self.get_competence_gated_entropy(
                    self.entropy_coeff)

                # Update competence signal at end of episode
                self.update_competence()

                update_metrics = self.actor_critic.update_reinforce(
                    states=states_np,
                    actions=self.episode_actions,
                    rewards=final_biological_rewards,  # Use calculated bio rewards
                    legal_masks=self.episode_masks if any(
                        m is not None for m in self.episode_masks) else None,
                    entropy_coeff=current_entropy,
                    lr=current_lr,
                    supervised_loss_coeff=self.supervised_loss_coeff,
                )

                self.hippocampus.consolidate_stm()

            # Use logged_outcome for statistics (from agent's perspective)
            self.episode_outcomes.append(logged_outcome)

            metrics_payload = {
                "episode": int(self.episode_index),
                "outcome": float(logged_outcome),
                "episode_length": len(self.episode_actions),
                "current_entropy": float(self._get_current_entropy()),
                "current_lr": float(self._get_current_learning_rate()),
                "competence": float(self.competence),
                "norepinephrine": float(self.neuromodulators.norepinephrine),
                "tonic_dopamine": float(self.neuromodulators.tonic_dopamine),
                "tonic_acetylcholine": float(self.neuromodulators.tonic_acetylcholine),
                "boredom": float(self.neuromodulators.boredom) if hasattr(self.neuromodulators, 'boredom') else 0.0,
                "random_player": getattr(self, '_current_episode_random_player', None),
            }
            if self.episode_players:
                players_np = np.array(self.episode_players, dtype=np.int8)
                metrics_payload["x_moves"] = int((players_np == 1).sum())
                metrics_payload["o_moves"] = int((players_np == -1).sum())
            if update_metrics is not None:
                metrics_payload.update(update_metrics)
            metrics_payload.setdefault(
                "mean_reward", float(np.mean(rewards)) if len(rewards) > 0 else 0.0)

            self._record_training_metrics(metrics_payload)

            self.episode_board = None
            self.episode_player = None
            self.episode_states = []
            self.episode_actions = []
            self.episode_players = []
            self.episode_masks = []
        else:
            self.episode_outcomes.append(0.0)
            self._record_training_metrics({
                "episode": int(self.episode_index),
                "outcome": 0.0
            })

        self.cum_reward = 0.0

    def _apply_decay(self, start, end, progress, decay_type='linear'):
        if start == end:
            return start
        progress = max(0.0, min(1.0, progress))
        if decay_type == 'linear':
            return start + (end - start) * progress
        elif decay_type == 'exponential':
            k = -math.log(0.01)
            decay_factor = math.exp(-k * progress)
            return end + (start - end) * decay_factor
        elif decay_type == 'cosine':
            return end + (start - end) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return start + (end - start) * progress

    def _get_current_entropy(self):
        """
        Get current entropy coefficient.
        Returns the actual competence-gated entropy being used, not the scheduled value.
        """
        # Use competence-gated entropy (the actual value used in training)
        return self.get_competence_gated_entropy(self.entropy_coeff)

    def _get_current_temperature(self):
        current_entropy = self._get_current_entropy()
        max_entropy = getattr(self, '_entropy_start',
                              self.entropy_coeff) or 0.001
        if max_entropy > 0:
            temperature = 0.5 + 0.5 * (current_entropy / max_entropy)
            temperature = max(0.5, min(1.0, temperature))
        else:
            temperature = 0.5
        return temperature

    def _get_current_learning_rate(self):
        """
        Get current learning rate.
        Returns the actual competence-gated learning rate being used, not the scheduled value.
        """
        # Use competence-gated learning rate (the actual value used in training)
        return self.get_competence_gated_lr(self.lr_policy)

    def _get_random_opponent_probability(self):
        if not hasattr(self, '_random_opponent_prob_start'):
            return 0.0
        episodes_elapsed = self.episode_index - self._random_opponent_initial_episode
        if episodes_elapsed >= self._random_opponent_decay_episodes:
            return self._random_opponent_prob_end
        progress = episodes_elapsed / self._random_opponent_decay_episodes
        return self._apply_decay(self._random_opponent_prob_start, self._random_opponent_prob_end, progress, getattr(self, '_random_opponent_decay_type', 'linear'))

    def get_performance_stats(self, window=None):
        if window is None:
            window = len(self.episode_outcomes)
        recent = self.episode_outcomes[-window:] if len(
            self.episode_outcomes) >= window else self.episode_outcomes
        if len(recent) == 0:
            return {'wins': 0, 'losses': 0, 'draws': 0, 'win_rate': 0.0, 'loss_rate': 0.0, 'draw_rate': 0.0, 'total': 0}
        wins = sum(1 for o in recent if o > 0)
        losses = sum(1 for o in recent if o < 0)
        draws = sum(1 for o in recent if o == 0)
        total = len(recent)
        return {
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins / total if total > 0 else 0.0,
            'loss_rate': losses / total if total > 0 else 0.0,
            'draw_rate': draws / total if total > 0 else 0.0,
            'total': total
        }

    def print_performance_stats(self, window=None, prefix="", current_entropy=None, current_lr=None, current_temp=None):
        stats = self.get_performance_stats(window)
        entropy_str = f" | EntCoeff={current_entropy:.4f}" if current_entropy is not None else ""
        lr_str = f" | LR={current_lr:.5f}" if current_lr is not None else ""
        temp_str = f" | T={current_temp:.2f}" if current_temp is not None else ""
        competence_str = f" | C={self.competence:.3f}" if hasattr(
            self, 'competence') else ""
        nm_str = f" | NE={self.neuromodulators.norepinephrine:.2f} DA={self.neuromodulators.tonic_dopamine:.2f}"
        boredom_str = f" | Boredom={self.neuromodulators.boredom:.3f}" if hasattr(
            self.neuromodulators, 'boredom') else ""
        print(f"{prefix}Ep: {self.episode_index} | "
              f"{stats['total']}: "
              f"W={stats['wins']} "
              f"L={stats['losses']} "
              f"D={stats['draws']} ({stats['draw_rate']*100:.0f}%){entropy_str}{lr_str}{competence_str}{nm_str}{boredom_str}")

    @staticmethod
    def _sanitize_metric_value(value):
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        return value

    def _record_training_metrics(self, metrics: Dict[str, Any]):
        if metrics is None:
            return
        sanitized = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                sanitized[key] = {k: self._sanitize_metric_value(
                    v) for k, v in value.items()}
            else:
                sanitized[key] = self._sanitize_metric_value(value)
        sanitized.setdefault("timestamp", time.time())
        self.training_metrics_history.append(sanitized)
        if len(self.training_metrics_history) > self._max_metrics_history:
            self.training_metrics_history = self.training_metrics_history[-self._max_metrics_history:]
        self._last_episode_metrics = sanitized

    def get_training_metric_summary(self, window=100):
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
        for field in ("policy_loss", "value_loss", "entropy_term", "mean_reward", "std_reward", "episode_length",
                      "competence", "current_entropy", "current_lr",
                      "norepinephrine", "tonic_dopamine", "tonic_acetylcholine", "boredom"):
            _maybe_avg(field)
        draws = sum(1 for entry in recent if float(
            entry.get("outcome", 0.0)) == 0.0)
        summary["draw_rate"] = draws / len(recent)
        summary["latest_random_eval"] = self._last_random_eval
        return summary

    def attach_random_eval(self, episode_idx: int, stats: Dict[str, Any]):
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
        play_against_opponent: bool = False,
        metrics_log_path: Optional[str] = None,
        metrics_window: int = 200,
        eval_games: int = 200,
        random_loss_patience: Optional[int] = None,
        random_loss_min_delta: float = 0.0,
        connection_config: Optional[BrainConnectionConfig] = None,
    ):
        """
        Run brain client with ConnectionManager for bi-directional communication.

        Args:
            connection_config: Optional BrainConnectionConfig. If None, creates default
                config from host/port args for backwards compatibility.
        """
        # Create connection config if not provided (backwards compatibility)
        if connection_config is None:
            connection_config = BrainConnectionConfig.from_args(
                host=host,
                port=port,
                authkey=authkey,
            )

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

        print(
            f"[brain] Initializing ConnectionManager (brain_id: {connection_config.brain_id})")
        if connection_config.listen_address:
            print(
                f"[brain] Listener enabled on {connection_config.listen_address}")

        # Create ConnectionManager (uses multicast discovery)
        print(
            f"[brain] Creating ConnectionManager for {connection_config.brain_id}...")
        connection_manager = ConnectionManager(
            connection_config, peer_type=PEER_TYPE_BRAIN)
        print(f"[brain] ConnectionManager created, starting discovery...")

        # Helper to find environment connection
        def get_environment_peer_id():
            """Find the connected environment peer."""
            for peer_id, metadata in connection_manager.connection_metadata.items():
                if (peer_id in connection_manager.connections and
                        metadata.get("peer_type") == PEER_TYPE_ENVIRONMENT):
                    return peer_id
            return None

        try:
            running = True
            initial_episode = self.episode_index
            # Track environment's starting episode for correct "trained for X episodes" calculation
            self._environment_starting_episode = None

            # Setup decay parameters (same as before)
            if entropy_start is None:
                entropy_start = self.entropy_coeff
            if entropy_decay_episodes is None:
                entropy_decay_episodes = max_episodes if max_episodes else 5000
            self._entropy_start = entropy_start
            self._entropy_end = entropy_end
            self._entropy_decay_episodes = entropy_decay_episodes
            self._entropy_initial_episode = initial_episode
            self._entropy_decay_type = entropy_decay_type

            if lr_start is None:
                lr_start = self.lr_policy
            if lr_end is None:
                lr_end = self.lr_policy
            if lr_decay_episodes is None:
                lr_decay_episodes = max_episodes if max_episodes else 5000
            self._lr_start = lr_start
            self._lr_end = lr_end
            self._lr_decay_episodes = lr_decay_episodes
            self._lr_initial_episode = initial_episode
            self._lr_decay_type = lr_decay_type

            # If playing against opponent, disable random moves (always use learned policy)
            if play_against_opponent:
                random_opponent_prob_start = 0.0
                random_opponent_prob_end = 0.0
                random_opponent_decay_episodes = 1  # Immediate, no decay needed

            if random_opponent_decay_episodes is None:
                random_opponent_decay_episodes = max_episodes if max_episodes else 5000
            self._random_opponent_prob_start = random_opponent_prob_start
            self._random_opponent_prob_end = random_opponent_prob_end
            self._random_opponent_decay_episodes = random_opponent_decay_episodes
            self._random_opponent_initial_episode = initial_episode
            self._random_opponent_decay_type = random_opponent_decay_type

            # Wait for environment connection and initial observation
            import time as time_module
            print("[brain] Discovering environment via multicast...")
            initial_received = False
            env_discovered_logged = False
            env_connected_logged = False
            while not initial_received:
                try:
                    # First, check if we've discovered the environment via multicast
                    known_peers = connection_manager.get_all_known_peers()
                    env_peers = [pid for pid, info in known_peers.items()
                                 if info.kind == PEER_TYPE_ENVIRONMENT]

                    if env_peers and not env_discovered_logged:
                        env_count = len(env_peers)
                        if env_count == 1:
                            print(
                                f"[brain] Discovered environment: {env_peers[0]}")
                            env_discovered_logged = True
                        else:
                            print(
                                f"[brain] Discovered {env_count} environments, waiting for single environment...")
                            env_discovered_logged = True

                    # Only check for connection after discovery
                    if env_peers:
                        env_peer_id = get_environment_peer_id()
                        if env_peer_id and not env_connected_logged:
                            print(
                                f"[brain] Connected to environment: {env_peer_id}")
                            env_connected_logged = True
                    else:
                        # No environment discovered yet, wait a bit
                        time_module.sleep(0.1)
                        continue

                    events = connection_manager.poll_events()
                    for peer_id, msg in events:
                        # Process initial observation from environment
                        if isinstance(msg, dict) and msg.get("type") == OBSERVATION:
                            peer_type = connection_manager.get_peer_type(
                                peer_id)
                            if peer_type == PEER_TYPE_ENVIRONMENT:
                                info = msg.get("info", {})
                                player = info.get("player", "?")
                                episode = info.get("episode", 0)
                                # Track environment's starting episode
                                if self._environment_starting_episode is None and episode > 0:
                                    self._environment_starting_episode = episode - 1
                                print(
                                    f"[brain] Training started! I am player {player}, Episode {episode}")
                                self.on_observation(
                                    msg.get("sensors", []), info)
                                # Ensure pending_decision is set (on_observation should set it, but double-check)
                                if not self.pending_decision:
                                    print(
                                        f"[brain] WARNING: pending_decision is False after on_observation(), setting to True")
                                    self.pending_decision = True
                                initial_received = True
                                break

                    if not initial_received:
                        env_peer_id = get_environment_peer_id()
                        if not env_peer_id:
                            # No environment connected yet
                            time_module.sleep(0.1)
                        else:
                            # Connected but no message yet
                            time_module.sleep(0.01)
                except KeyboardInterrupt:
                    print(
                        "\n[brain] Interrupted by user while waiting for initial observation")
                    connection_manager.close()
                    return
                except Exception as e:
                    # Other errors - log but continue waiting
                    print(f"[brain] Unexpected error while waiting: {e}")
                    time_module.sleep(0.1)

            while running:
                now = time.time()
                elapsed = now - last_time

                if max_episodes is not None and self.episode_index >= initial_episode + max_episodes:
                    print(
                        f"\n[brain] Reached max episodes limit ({max_episodes})")
                    running = False
                    break

                message_processed = False
                # Poll for messages from all peers
                events = connection_manager.poll_events()
                for peer_id, msg in events:
                    try:
                        if not isinstance(msg, dict):
                            continue
                        mtype = msg.get("type")

                        # Handle discovery messages (startup/shutdown are now handled via multicast)
                        if mtype == DISCOVERY_STARTUP:
                            # Peer sent startup message over TCP - just acknowledge
                            sender_peer_id = msg.get("peer_id") or peer_id
                            sender_peer_type = msg.get("peer_type", "unknown")
                            print(
                                f"[brain] Received startup from {sender_peer_id} ({sender_peer_type})")
                            message_processed = True

                        elif mtype == DISCOVERY_SHUTDOWN:
                            # Peer is shutting down
                            sender_peer_id = msg.get("peer_id")
                            sender_peer_type = msg.get("peer_type", "unknown")
                            print(
                                f"[brain] Peer {sender_peer_id} ({sender_peer_type}) is shutting down")
                            if sender_peer_type == PEER_TYPE_ENVIRONMENT:
                                print(
                                    "[brain] Environment shutting down - waiting for reconnection...")
                                connection_manager._mark_disconnected(
                                    sender_peer_id)
                                self._was_waiting_for_reconnect = True
                            message_processed = True

                        # Process application messages from environment
                        elif mtype == OBSERVATION:
                            # OBSERVATION messages only come from environments
                            peer_type = connection_manager.get_peer_type(
                                peer_id)
                            if peer_type == PEER_TYPE_ENVIRONMENT:
                                # Check if we just reconnected
                                if hasattr(self, '_was_waiting_for_reconnect'):
                                    print(
                                        f"[brain] Received observation from reconnected environment, resuming training...")
                                    delattr(self, '_was_waiting_for_reconnect')

                            # Process observation from environment
                            self.on_observation(
                                msg.get("sensors", []), msg.get("info", {}))
                            message_processed = True
                        elif mtype == REWARD:
                            # REWARD messages only come from environments
                            # Process reward from environment
                            self.on_reward(
                                float(msg.get("value", 0.0)), msg.get("info", {}))
                            message_processed = True
                        elif mtype == TERMINAL:
                            # TERMINAL messages only come from environments
                            # Process terminal from environment
                            # Always process TERMINAL messages (they only come from environments)
                            info = msg.get("info", {})
                            episode_num = info.get("episode", 0)
                            # Update episode_index to match environment's episode number
                            # This ensures we track episodes correctly for the "trained for X episodes" message
                            if episode_num > 0:
                                self.episode_index = episode_num

                            self._handle_terminal(info)

                            if self.episode_index % stats_every == 0:
                                self.print_performance_stats(
                                    window=self.stats_window,
                                    current_entropy=self._get_current_entropy(),
                                    current_lr=self._get_current_learning_rate(),
                                    current_temp=self._get_current_temperature()
                                )

                                # Add detailed metrics
                                summary_window = max(
                                    1, min(metrics_window, self.stats_window))
                                metric_summary = self.get_training_metric_summary(
                                    summary_window)
                                if metric_summary:
                                    payload = []
                                    policy_loss_avg = metric_summary.get(
                                        "policy_loss_avg")
                                    value_loss_avg = metric_summary.get(
                                        "value_loss_avg")
                                    entropy_term_avg = metric_summary.get(
                                        "entropy_term_avg")
                                    ach_avg = metric_summary.get(
                                        "tonic_acetylcholine_avg")

                                    if policy_loss_avg is not None:
                                        payload.append(
                                            f"PolicyLoss={policy_loss_avg:.4f}")
                                    if value_loss_avg is not None:
                                        payload.append(
                                            f"ValueLoss={value_loss_avg:.4f}")
                                    if entropy_term_avg is not None:
                                        payload.append(
                                            f"EntropyTerm={entropy_term_avg:.4f}")
                                    if ach_avg is not None:
                                        payload.append(
                                            f"ACh={ach_avg:.3f}")

                                    if payload:
                                        print("  | " + " | ".join(payload))

                                # Test against random player periodically
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

                            if save_path and save_every and self.episode_index % save_every == 0:
                                self.save(save_path)
                            message_processed = True
                        elif mtype == SHUTDOWN:
                            # Environment is shutting down - treat as disconnection and wait for reconnection
                            # Don't exit the brain, just mark the connection as disconnected
                            peer_type = connection_manager.get_peer_type(
                                peer_id)
                            if peer_type == PEER_TYPE_ENVIRONMENT:
                                print(
                                    "[brain] Environment sent shutdown message - waiting for reconnection...")
                                connection_manager._mark_disconnected(peer_id)
                                # Mark that we're waiting for reconnect
                                self._was_waiting_for_reconnect = True
                            message_processed = True
                    except Exception as e:
                        print(
                            f"[brain] Error processing message from {peer_id}: {e}")
                        # Continue - don't break on other errors
                        continue

                if not running:
                    break

                # Check if environment is connected
                env_peer_id = get_environment_peer_id()
                if not env_peer_id:
                    # Environment is disconnected, wait for reconnection
                    self._was_waiting_for_reconnect = True
                    # Log waiting status periodically (every 5 seconds)
                    if not hasattr(self, '_last_reconnect_log') or (time.time() - self._last_reconnect_log) > 5.0:
                        print(
                            "[brain] Environment disconnected, waiting for reconnection...")
                    self._last_reconnect_log = time.time()
                    # Sleep longer when disconnected to reduce CPU usage
                    time.sleep(0.1)
                    continue  # Skip processing actions until reconnected
                else:
                    # Clear reconnect log time when connected
                    if hasattr(self, '_last_reconnect_log'):
                        delattr(self, '_last_reconnect_log')

                if message_processed or elapsed >= self.dt:
                    if elapsed >= self.dt:
                        last_time = now
                        t += elapsed
                    try:
                        action_msg = self.tick(
                            elapsed if elapsed >= self.dt else 0.0)
                        if action_msg is not None:
                            # print(f"[brain] Sending action to environment: {action_msg}")
                            # Send to environment
                            env_peer_id = get_environment_peer_id()
                            if env_peer_id:
                                # print(
                                #     f"[brain] Sending action to environment: {action_msg.get('actions', [])}")
                                connection_manager.send(
                                    env_peer_id, action_msg)
                        elif self.pending_decision:
                            # We have a pending decision but tick() returned None
                            # This might be because it's not our turn
                            info = getattr(self, '_last_observation_info', {})
                            current_turn = info.get("current_turn", "unknown")
                            if current_turn != self.my_player_symbol:
                                # Not our turn, that's expected
                                pass
                    except Exception as e:
                        print(f"[brain] Error in tick(): {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue - don't break on other errors
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[brain] interrupted by user")
        finally:
            # Process any remaining messages to ensure episode_index is up to date
            try:
                for _ in range(10):  # Poll a few times to catch any pending messages
                    events = connection_manager.poll_events()
                    if not events:
                        break
                    for peer_id, msg in events:
                        if isinstance(msg, dict):
                            mtype = msg.get("type")
                            if mtype == TERMINAL:
                                info = msg.get("info", {})
                                episode_num = info.get("episode", 0)
                                if episode_num > 0:
                                    self.episode_index = episode_num
                                self._handle_terminal(info)
            except Exception:
                pass  # Don't fail on cleanup

            connection_manager.close()
            last_logged_episode = flush_metrics(last_logged_episode)

        # If we exit normally, break outer loop
        if early_stop_reason:
            print(f"[brain] Training stopped early: {early_stop_reason}")
        if len(self.episode_outcomes) > 0:
            print("\n" + "=" * 70)
            print("Final Performance Summary")
            self.print_performance_stats(window=None, prefix="")
            print("=" * 70)

    # ----- offline replay -----

    def offline_replay(self, n_batches=10):
        """
        Replay stored experiences (dreaming/consolidation).
        """
        # Use Hippocampus to sample
        # batch = self.hippocampus.sample_replay(self.replay_batch_size)
        # Since we changed the structure, let's implement a simple version
        pass

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
            "reward_shaping": self.reward_shaping,
            "supervised_loss_coeff": self.supervised_loss_coeff,
            "mode_dim": self.mode_dim,
            "z_mode_sigma_base": self.z_mode_sigma_base,
            "k_z": self.k_z,
        }
        # Serialize new components
        nm_state = {
            "dopamine": self.neuromodulators.dopamine,
            "tonic_dopamine": self.neuromodulators.tonic_dopamine,
            "norepinephrine": self.neuromodulators.norepinephrine,
            "tonic_norepinephrine": self.neuromodulators.tonic_norepinephrine,
            "serotonin": self.neuromodulators.serotonin,
            "tonic_serotonin": self.neuromodulators.tonic_serotonin,
            "acetylcholine": self.neuromodulators.acetylcholine,
            "tonic_acetylcholine": self.neuromodulators.tonic_acetylcholine,
            "cortisol": self.neuromodulators.cortisol,
            "tonic_cortisol": self.neuromodulators.tonic_cortisol,
            "decay": self.neuromodulators.decay,
            "boredom": self.neuromodulators.boredom,
            "da_window": list(self.neuromodulators.da_window),
            "ne_window": list(self.neuromodulators.ne_window),
        }

        return {
            "config": config,
            "rng_state": self.rng.get_state(),
            "encoder_state": encoder_state,
            "world_model": self.world_model.to_state(),
            "actor_critic": self.actor_critic.to_state(),
            "hippocampus": self.hippocampus.to_state(),
            "neuromodulators": nm_state,  # Simplified, or implement to_state in NM
            "intrinsic": {"prev_pred_error": self.intrinsic.prev_pred_error},
            "drives": self.drives.get_drive_state_vector(),
            "competence": self.competence,
            "da_error_window": list(self.da_error_window),
            "ne_error_window": list(self.ne_error_window),
            "loss_rate_window": list(self.loss_rate_window),
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
            supervised_loss_coeff=config.get("supervised_loss_coeff", 5.0),
            # Default to 0 for backward compatibility
            mode_dim=config.get("mode_dim", 0),
            z_mode_sigma_base=config.get("z_mode_sigma_base", 1.0),
            k_z=config.get("k_z", 0.5),
            rng=rng,
        )

        agent.world_model = HierarchicalWorldModel.from_state(
            state["world_model"], rng=rng)
        agent.actor_critic = ActorCritic.from_state(state["actor_critic"])

        # Handle legacy memory state if loading old brain
        if "hippocampus" in state:
            agent.hippocampus = Hippocampus.from_state(state["hippocampus"])
        elif "memory" in state:
            # Migration from old EpisodicMemory to Hippocampus
            # For now, just create empty hippocampus
            pass

        # Load neuromodulators
        if "neuromodulators" in state:
            nm = state["neuromodulators"]
            agent.neuromodulators.dopamine = nm.get("dopamine", 0.0)
            agent.neuromodulators.tonic_dopamine = nm.get(
                "tonic_dopamine", 0.5)
            agent.neuromodulators.norepinephrine = nm.get(
                "norepinephrine", 0.0)
            agent.neuromodulators.tonic_norepinephrine = nm.get(
                "tonic_norepinephrine", 0.1)
            agent.neuromodulators.serotonin = nm.get("serotonin", 0.0)
            agent.neuromodulators.tonic_serotonin = nm.get(
                "tonic_serotonin", 0.5)
            agent.neuromodulators.acetylcholine = nm.get("acetylcholine", 0.0)
            agent.neuromodulators.tonic_acetylcholine = nm.get(
                "tonic_acetylcholine", 0.5)
            agent.neuromodulators.cortisol = nm.get("cortisol", 0.0)
            agent.neuromodulators.tonic_cortisol = nm.get(
                "tonic_cortisol", 0.1)
            agent.neuromodulators.decay = nm.get("decay", 0.99)
            # Load boredom state if available
            agent.neuromodulators.boredom = nm.get("boredom", 0.0)
            if "da_window" in nm:
                agent.neuromodulators.da_window = deque(
                    nm["da_window"], maxlen=agent.neuromodulators.window_size)
            if "ne_window" in nm:
                agent.neuromodulators.ne_window = deque(
                    nm["ne_window"], maxlen=agent.neuromodulators.window_size)

        # Load intrinsic motivation
        if "intrinsic" in state:
            agent.intrinsic.prev_pred_error = state["intrinsic"].get(
                "prev_pred_error", 0.0)

        # Load drives
        if "drives" in state:
            drive_vec = state["drives"]
            if isinstance(drive_vec, (list, np.ndarray)) and len(drive_vec) >= 3:
                agent.drives.energy = float(drive_vec[0])
                agent.drives.curiosity_drive = float(drive_vec[1])
                agent.drives.competence_drive = float(drive_vec[2])

        # Load competence state (backward compatible - defaults to 0.0)
        agent.competence = state.get("competence", 0.0)
        if "da_error_window" in state:
            agent.da_error_window = deque(
                state["da_error_window"], maxlen=agent.competence_window_size)
        if "ne_error_window" in state:
            agent.ne_error_window = deque(
                state["ne_error_window"], maxlen=agent.competence_window_size)
        if "loss_rate_window" in state:
            agent.loss_rate_window = deque(
                state["loss_rate_window"], maxlen=agent.competence_window_size)

        # Initialize mode variable if agent has mode_dim > 0
        # Use competence-gated sigma (deterministic if competence is high)
        if agent.mode_dim > 0:
            if agent.competence > 0.9:
                # High competence -> deterministic (zero mode)
                agent.current_z_mode = np.zeros(
                    agent.mode_dim, dtype=np.float32)
            else:
                # Low competence -> use competence-gated sigma
                z_sigma = agent.get_competence_gated_z_sigma(
                    agent.z_mode_sigma_base)
                agent.current_z_mode = agent.rng.normal(
                    0.0, z_sigma, size=agent.mode_dim).astype(np.float32)

        agent.global_step = state.get("global_step", 0)
        agent.episode_index = state.get("episode_index", 0)

        return agent

    def run_continuous(self, env, max_steps=None, offline_every=100, offline_batches=10, save_path=None, save_every=None, temperature_schedule=None, greedy_after=None):
        # Similar logic as before, but calling online_update
        obs = env.reset()
        self.episode_index += 1
        try:
            while True:
                if max_steps is not None and self.global_step >= max_steps:
                    break

                temp = 1.0
                if temperature_schedule:
                    temp = float(temperature_schedule(self.global_step))
                greedy = greedy_after is not None and self.global_step >= greedy_after

                action, z, value, x = self.act(
                    obs, temperature=temp, greedy=greedy)
                next_obs, ext_reward, done, info = env.step(action)

                try:
                    legal_mask = self.get_legal_actions(obs)
                except:
                    legal_mask = None

                self.online_update(obs, x, z, action, ext_reward,
                                   next_obs, done, legal_mask=legal_mask)

                obs = next_obs
                if save_path and save_every and self.global_step % save_every == 0:
                    self.save(save_path)

                if done:
                    obs = env.reset()
                    self.episode_index += 1
                    self.reset_episode()
        except KeyboardInterrupt:
            pass
