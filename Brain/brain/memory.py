import numpy as np


class EpisodicMemory:
    """
    Replay buffer (hippocampal-like).
    """

    def __init__(self, capacity=100000, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.rng = rng

    def store(self, obs, state, action, reward, next_obs, next_state, done):
        transition = (
            np.asarray(obs, dtype=np.float32).copy(),
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32).copy(),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        batch_size = min(batch_size, len(self.buffer))
        idx = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def to_state(self):
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
            "position": self.position,
            "rng_state": self.rng.get_state(),
        }

    @staticmethod
    def from_state(state):
        rng = np.random.RandomState()
        rng.set_state(state["rng_state"])
        mem = EpisodicMemory(capacity=state["capacity"], rng=rng)
        mem.buffer = list(state["buffer"])
        mem.position = state["position"]
        return mem
