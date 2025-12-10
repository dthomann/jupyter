import numpy as np
from collections import deque

class ShortTermMemory:
    """
    Working memory / Short-term memory.
    Limited capacity (e.g., 7 items). Stores recent transitions.
    """
    def __init__(self, capacity=7):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, item):
        self.buffer.append(item)

    def get_all(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()


class EpisodicMemory:
    """
    Long-term storage of specific episodes (experiences).
    """
    def __init__(self, capacity=100000, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.rng = rng
        # Priority/Importance weights could be added here
        self.priorities = []

    def store(self, transition, importance=1.0):
        """
        Store transition with importance weight.
        Transition: (obs, state, action, reward, next_obs, next_state, done)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(importance)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = importance
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        
        # Simple uniform sampling for now, could be priority-based
        batch_size = min(batch_size, len(self.buffer))
        idx = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def retrieve_similar(self, current_state, k=1):
        """
        Retrieve k episodes most similar to current_state.
        Simple Euclidean distance on 'state' (latent z).
        """
        if not self.buffer:
            return []
        
        # Extract all stored states (z) - simplistic implementation (slow for large buffers)
        # In production, use a KD-tree or FAISS.
        # buffer item index 1 is 'state' (z)
        states = np.array([item[1] for item in self.buffer])
        dists = np.linalg.norm(states - current_state, axis=1)
        
        # Get indices of k smallest distances
        if len(dists) < k:
            k = len(dists)
            
        idx = np.argpartition(dists, k)[:k]
        return [self.buffer[i] for i in idx]

    def to_state(self):
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
            "priorities": list(self.priorities),
            "position": self.position,
            "rng_state": self.rng.get_state(),
        }

    @staticmethod
    def from_state(state):
        rng = np.random.RandomState()
        rng.set_state(state["rng_state"])
        mem = EpisodicMemory(capacity=state["capacity"], rng=rng)
        mem.buffer = list(state["buffer"])
        mem.priorities = list(state.get("priorities", [1.0]*len(mem.buffer)))
        mem.position = state["position"]
        return mem


class Hippocampus:
    """
    Orchestrates memory consolidation and retrieval.
    Manages flow from STM -> LTM (Episodic).
    Decides what to consolidate based on Neuromodulators (Salience).
    """
    def __init__(self, stm_capacity=10, ltm_capacity=100000, rng=None):
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.episodic = EpisodicMemory(capacity=ltm_capacity, rng=rng)
        self.rng = rng

    def process_experience(self, transition, neuromodulators):
        """
        Process a new experience.
        1. Add to STM.
        2. Decide whether to consolidate to Episodic immediately (if salient).
        """
        self.stm.add(transition)
        
        # Calculate salience based on neuromodulators
        # High Dopamine (Reward) or High Norepinephrine (Surprise) -> High Importance
        salience = abs(neuromodulators.dopamine) + neuromodulators.norepinephrine
        
        # Threshold for consolidation (simplified)
        # If highly salient, store in Episodic memory immediately
        if salience > 0.5:
            self.episodic.store(transition, importance=salience)

    def consolidate_stm(self):
        """
        Move everything from STM to Episodic (e.g., at end of episode or during sleep).
        """
        for item in self.stm.get_all():
            self.episodic.store(item, importance=0.5) # Default importance
        self.stm.clear()

    def recall(self, context_state):
        """
        Retrieve relevant memories for the current context.
        """
        return self.episodic.retrieve_similar(context_state)

    def sample_replay(self, batch_size):
        """
        Sample memories for replay (learning).
        """
        return self.episodic.sample(batch_size)

    def to_state(self):
        return {
            "episodic": self.episodic.to_state()
            # STM is transient, usually not saved, or we can save it if needed.
        }

    @staticmethod
    def from_state(state):
        # We need to pass RNG somehow or recreate it
        mem = Hippocampus()
        if "episodic" in state:
            mem.episodic = EpisodicMemory.from_state(state["episodic"])
        return mem
