import numpy as np
from .utils import softmax


class ActorCritic:
    """
    Linear actor-critic on latent state z.
    """

    def __init__(self, state_dim, n_actions, scale=0.1, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.W_policy = scale * rng.randn(state_dim, n_actions)
        self.W_value = scale * rng.randn(state_dim)

    def act(self, state, temperature=1.0, greedy=False):
        """
        Sample or choose action given state.
        """
        logits = state @ self.W_policy
        if temperature <= 0:
            probs = np.zeros_like(logits)
            probs[np.argmax(logits)] = 1.0
        else:
            logits = logits / temperature
            probs = softmax(logits)
        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.n_actions, p=probs))
        value = state @ self.W_value
        return action, probs, value

    def update(self, state, action, reward, next_state, done, neuromodulators, base_lr=1e-2):
        """
        TD update of critic and policy.
        NeuromodulatorState carries TD error as dopamine.
        """
        value = state @ self.W_value
        next_value = 0.0 if done else next_state @ self.W_value

        # First pass: dopaminergic TD error, pred_error_norm=0 here
        td_error = neuromodulators.update(
            reward=reward,
            value=value,
            next_value=next_value,
            pred_error_norm=0.0,
        )

        lr_value = base_lr * (1.0 + abs(neuromodulators.norepinephrine))
        lr_policy = base_lr * (1.0 + abs(neuromodulators.dopamine))

        # Critic
        self.W_value += lr_value * td_error * state

        # Actor
        logits = state @ self.W_policy
        probs = softmax(logits)
        grad_log_policy = -probs
        grad_log_policy[action] += 1.0
        self.W_policy += lr_policy * td_error * np.outer(state, grad_log_policy)

        return td_error, value, next_value


