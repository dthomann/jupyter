import numpy as np


class IntrinsicMotivation:
    """
    Intrinsic rewards from internal signals:
      - curiosity: large prediction errors
      - learning progress: reduction in prediction error
    """

    def __init__(self, curiosity_scale=0.1, learning_progress_scale=0.1):
        self.curiosity_scale = curiosity_scale
        self.learning_progress_scale = learning_progress_scale
        self.prev_pred_error = 0.0

    def compute(self, pred_error_norm):
        pred_error_norm = float(pred_error_norm)

        curiosity = self.curiosity_scale * pred_error_norm

        improvement = max(0.0, self.prev_pred_error - pred_error_norm)
        learning_progress = self.learning_progress_scale * improvement

        self.prev_pred_error = pred_error_norm

        total_intrinsic = curiosity + learning_progress
        components = {
            "curiosity": curiosity,
            "learning_progress": learning_progress,
        }
        return total_intrinsic, components


class DriveState:
    """
    Slow internal drives accumulated from intrinsic rewards.
    """

    def __init__(self, decay=0.999):
        self.decay = decay
        self.curiosity_drive = 0.0
        self.competence_drive = 0.0

    def update(self, intrinsic_components):
        c = intrinsic_components.get("curiosity", 0.0)
        lp = intrinsic_components.get("learning_progress", 0.0)

        self.curiosity_drive = self.decay * \
            self.curiosity_drive + (1.0 - self.decay) * c
        self.competence_drive = self.decay * \
            self.competence_drive + (1.0 - self.decay) * lp

        return np.array([self.curiosity_drive, self.competence_drive], dtype=float)
