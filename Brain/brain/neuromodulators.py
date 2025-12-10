import numpy as np
from collections import deque


class NeuromodulatorState:
    """
    Expanded biologically-inspired neuromodulators.

    - Dopamine (DA): Reward prediction error. Signals value, drives policy learning (plasticity).
    - Norepinephrine (NE): Surprise / Unexpected uncertainty. Signals arousal, increases learning rate & attention.
    - Serotonin (5-HT): Patience / Punishment. Regulates discount factor and impulsivity (inhibits action).
    - Acetylcholine (ACh): Expected uncertainty / Attention. Gates bottom-up vs top-down processing (plasticity in sensory cortex).
    - Cortisol: Stress. Accumulated punishment/effort. Can inhibit retrieval or boost consolidation of negative events.
    """

    def __init__(self, decay=0.99, target_stim=0.3, k_b=0.01, window_size=100, w_da=0.5, w_ne=0.5, c_L=0.2, c_B=0.5):
        # Phasic levels (short-term spikes)
        self.dopamine = 0.0
        self.norepinephrine = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0
        self.cortisol = 0.0

        # Tonic levels (baseline/long-term average)
        self.tonic_dopamine = 0.5
        self.tonic_norepinephrine = 0.1
        self.tonic_serotonin = 0.5
        self.tonic_acetylcholine = 0.5
        self.tonic_cortisol = 0.1

        self.decay = decay

        # Boredom tracking
        self.boredom = 0.0  # Initialized to 0.0 (not bored initially)
        self.target_stim = target_stim  # Comfortable stimulation level
        self.k_b = k_b  # Boredom update rate
        self.window_size = window_size  # Window size for averaging DA/NE
        self.w_da = w_da  # Weight for DA in stimulation calculation
        self.w_ne = w_ne  # Weight for NE in stimulation calculation
        self.c_L = c_L  # Boredom multiplier for learning rate
        self.c_B = c_B  # Boredom multiplier for entropy
        self.da_window = deque(maxlen=window_size)  # Rolling window of |DA|
        self.ne_window = deque(maxlen=window_size)  # Rolling window of NE

    def update(self, reward, value, next_value, pred_error_norm, effort=0.0, punishment=0.0):
        """
        Update neuromodulators based on agent's experience.

        Args:
            reward: External reward received
            value: Estimated value of current state
            next_value: Estimated value of next state
            pred_error_norm: Magnitude of world model prediction error
            effort: Cost of action (metabolic cost)
            punishment: Negative reward signal
        """
        gamma = 0.99  # Could be modulated by serotonin
        td_error = reward + gamma * next_value - value

        # --- Dopamine (DA) ---
        # Classic RPE. Positive = better than expected.
        self.dopamine = td_error
        # Integrate into tonic (slowly)
        self.tonic_dopamine = 0.99 * self.tonic_dopamine + 0.01 * (0.5 + np.tanh(td_error))

        # --- Norepinephrine (NE) ---
        # Driven by surprise (prediction error) and sudden changes.
        # High NE = High plasticity, high attention (learning rate).
        self.norepinephrine = float(pred_error_norm)
        self.tonic_norepinephrine = 0.99 * self.tonic_norepinephrine + 0.01 * self.norepinephrine

        # --- Serotonin (5-HT) ---
        # Linked to punishment and patience.
        # High 5-HT = increased patience (higher gamma), reduced impulsivity.
        # Low 5-HT = impulsive.
        # We model it as reacting to negative reward/punishment.
        if reward < 0 or punishment > 0:
             # Dip in serotonin on punishment (or spike? Biological literature varies, but commonly low 5-HT -> impulsivity/depression)
             # Here we model: High Serotonin = "Everything is okay/stable", Low Serotonin = "Bad things happening"
             self.serotonin = -1.0 * abs(reward) - punishment
        else:
             self.serotonin = 0.1 * reward # Small boost for small rewards

        self.tonic_serotonin = 0.995 * self.tonic_serotonin + 0.005 * (0.5 + np.tanh(self.serotonin))

        # --- Acetylcholine (ACh) ---
        # Expected uncertainty. Tracks average surprise.
        # High ACh = high environmental volatility -> rely on external input (high learning rate).
        # Low ACh = stable environment -> rely on internal model.
        self.acetylcholine = float(pred_error_norm)
        self.tonic_acetylcholine = self.decay * self.tonic_acetylcholine + (1.0 - self.decay) * self.acetylcholine

        # --- Cortisol ---
        # Stress/Effort.
        self.cortisol = 0.9 * self.cortisol + 0.1 * (effort + punishment)
        self.tonic_cortisol = 0.999 * self.tonic_cortisol + 0.001 * self.cortisol

        # Update boredom after updating DA/NE
        self.update_boredom()

        return td_error

    def get_learning_rate_factor(self):
        """
        Modulate learning rate.
        High NE (surprise) and High ACh (uncertainty) -> Boost Learning.
        Boredom also increases learning rate when stimulation is low.
        """
        # Base factor 1.0.
        # Add NE contribution (surprise boosts learning).
        # Add ACh contribution (uncertainty boosts learning).
        base_factor = 1.0 + 2.0 * self.norepinephrine + 1.0 * self.tonic_acetylcholine
        # Multiply by boredom factor
        factor = base_factor * (1.0 + self.c_L * self.boredom)
        return np.clip(factor, 0.1, 5.0)

    def get_discount_factor(self):
        """
        Modulate discount factor (gamma).
        High Serotonin -> High Gamma (Patience).
        Low Serotonin -> Low Gamma (Impulsivity).
        """
        base_gamma = 0.9
        # Map tonic serotonin [0, 1] to gamma modulation
        # If serotonin is high (0.8), gamma -> 0.99
        # If serotonin is low (0.2), gamma -> 0.5
        gamma = base_gamma + 0.09 * (self.tonic_serotonin * 2 - 1)
        return np.clip(gamma, 0.1, 0.999)

    def get_exploration_entropy(self):
        """
        Modulate exploration (entropy).
        High Dopamine (doing well) -> Exploit (Lower entropy).
        Low Dopamine (doing poorly) -> Explore (Higher entropy).
        High NE (Surprise) -> Explore (Higher entropy).
        Boredom increases exploration when stimulation is low.
        """
        # Inverse relation with tonic dopamine?
        # Actually, often High DA = Exploit (Go for reward), Low DA = Apathy or Random.
        # Let's say: High NE = Confusion/Surprise -> High Entropy (Randomness)
        base_factor = 1.0 + 1.0 * self.norepinephrine - 0.5 * (self.tonic_dopamine - 0.5)
        # Multiply by boredom factor
        entropy_factor = base_factor * (1.0 + self.c_B * self.boredom)
        return np.clip(entropy_factor, 0.1, 5.0)

    def update_boredom(self):
        """
        Update boredom based on stimulation levels.
        Boredom increases when stimulation (DA/NE) is below target.
        """
        # Add current DA and NE to windows
        self.da_window.append(abs(self.dopamine))
        self.ne_window.append(self.norepinephrine)

        # Calculate average stimulation if we have enough data
        if len(self.da_window) > 0 and len(self.ne_window) > 0:
            avg_abs_da = np.mean(list(self.da_window))
            avg_ne = np.mean(list(self.ne_window))
            stim_t = self.w_da * avg_abs_da + self.w_ne * avg_ne

            # Update boredom: increases when stim_t < target_stim
            boredom_delta = self.k_b * (self.target_stim - stim_t)
            self.boredom = np.clip(self.boredom + boredom_delta, 0.0, 1.0)
