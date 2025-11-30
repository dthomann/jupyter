import numpy as np


class EpisodicMemory:
    """
    Replay buffer (hippocampal-like).
    """

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

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
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]


