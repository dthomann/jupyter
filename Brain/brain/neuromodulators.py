import numpy as np


class NeuromodulatorState:
    """
    Simplified neuromodulators.
    - dopamine: reward prediction error
    - norepinephrine: surprise (prediction error magnitude)
    - acetylcholine: expected uncertainty (running average of surprise)
    """

    def __init__(self, ach_decay=0.99):
        self.dopamine = 0.0
        self.norepinephrine = 0.0
        self.acetylcholine = 0.5
        self.ach_decay = ach_decay

    def update(self, reward, value, next_value, pred_error_norm):
        """
        Update neuromodulators for a transition.
        pred_error_norm is the world model's prediction error magnitude.
        """
        gamma = 0.99
        td_error = reward + gamma * next_value - value

        # Signed value error
        self.dopamine = td_error

        # Surprise
        self.norepinephrine = pred_error_norm

        # Expected uncertainty
        self.acetylcholine = (
            self.ach_decay * self.acetylcholine
            + (1.0 - self.ach_decay) * pred_error_norm
        )

        return td_error


