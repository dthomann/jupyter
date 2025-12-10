import numpy as np


class IntrinsicMotivation:
    """
    Generates intrinsic rewards based on psychological drives.
    """

    def __init__(self, curiosity_scale=0.1, competence_scale=0.1, autonomy_scale=0.05):
        # Reward for prediction error (Novelty)
        self.curiosity_scale = curiosity_scale
        # Reward for learning progress (Improvement)
        self.competence_scale = competence_scale
        # Reward for entropy (Freedom of choice) - simplified
        self.autonomy_scale = autonomy_scale

        self.prev_pred_error = 0.0

    def compute(self, pred_error_norm, entropy_of_policy=0.0):
        """
        Compute intrinsic reward.
        """
        pred_error_norm = float(pred_error_norm)

        # 1. Curiosity: driven by novelty (prediction error)
        # NOTE: Too much curiosity can be bad (noisy TV problem).
        # We use a simple linear scale here.
        curiosity = self.curiosity_scale * pred_error_norm

        # 2. Competence / Learning Progress: reduction in prediction error
        # "I am getting better at understanding this"
        improvement = max(0.0, self.prev_pred_error - pred_error_norm)
        competence = self.competence_scale * improvement

        self.prev_pred_error = pred_error_norm

        # 3. Autonomy: Preference for states with options (High entropy)
        # We approximate this by the entropy of the current policy distribution
        autonomy = self.autonomy_scale * entropy_of_policy

        total_intrinsic = curiosity + competence + autonomy

        components = {
            "curiosity": curiosity,
            "competence": competence,
            "autonomy": autonomy
        }
        return total_intrinsic, components


class DriveState:
    """
    Homeostatic drives (Needs).
    - Energy: Depletes with action, restored by Reward (Food).
    - Boredom: Increases with time, reduced by Curiosity/Novelty.
    - Rest: Depletes with time/effort, restored by Inactivity (not implemented in TTT).
    """

    def __init__(self, decay=0.999):
        self.decay = decay

        # Physiological State (0 = depleted/suffering, 1 = satisfied)
        self.energy = 1.0
        self.comfort = 1.0

        # Psychological State (accumulated drives)
        self.curiosity_drive = 0.0  # Accumulated curiosity satisfaction
        self.competence_drive = 0.0  # Accumulated competence satisfaction

    def update(self, intrinsic_components, external_reward, effort=0.01):
        """
        Update internal drives.
        """
        # --- Energy Dynamics ---
        # Action costs energy. Reward restores it.
        self.energy -= effort
        if external_reward > 0:
            self.energy += external_reward  # Eat
        self.energy = np.clip(self.energy, 0.0, 1.0)

        # --- Boredom / Curiosity ---
        c = intrinsic_components.get("curiosity", 0.0)
        self.curiosity_drive = self.decay * \
            self.curiosity_drive + (1.0 - self.decay) * c

        # --- Competence ---
        lp = intrinsic_components.get("competence", 0.0)
        self.competence_drive = self.decay * \
            self.competence_drive + (1.0 - self.decay) * lp

        # Calculate "Drive Reduction Reward" (Homeostatic Reward)
        # If Energy is low, getting energy (reward) is MORE valuable.
        # If Bored, getting curiosity is MORE valuable.

        return self.get_drive_state_vector()

    def get_drive_multiplier(self):
        """
        Return multiplier for external rewards based on need.
        Hungry (Low Energy) -> High Multiplier.
        """
        # Simple linear hunger: 1.0 when full, 2.0 when empty
        return 2.0 - self.energy

    def get_drive_state_vector(self):
        return np.array([self.energy, self.curiosity_drive, self.competence_drive], dtype=float)
