import numpy as np


class MultiModalDummyEnv:
    """
    Toy multi-modal environment.
    Observations: dict with small vectors for "vision", "audio", "text".
    Actions slightly move the state; reward prefers small norm.
    """

    def __init__(self, modality_dims, n_actions, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.modality_dims = modality_dims
        self.n_actions = n_actions
        self.state = {m: rng.randn(d) for m, d in modality_dims.items()}
        self.action_effects = {
            m: rng.randn(n_actions, d) * 0.05 for m, d in modality_dims.items()
        }

    def reset(self):
        self.state = {m: self.rng.randn(d) for m, d in self.modality_dims.items()}
        return {m: v.copy() for m, v in self.state.items()}

    def step(self, action):
        total_reward = 0.0
        for m, d in self.modality_dims.items():
            effect = self.action_effects[m][action]
            noise = 0.01 * self.rng.randn(d)
            self.state[m] = self.state[m] + effect + noise
            total_reward -= float(np.linalg.norm(self.state[m]))
        done = False
        obs = {m: v.copy() for m, v in self.state.items()}
        info = {}
        return obs, total_reward, done, info


