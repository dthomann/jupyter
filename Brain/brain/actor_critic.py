import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions


class ActorCritic(nn.Module):
    """
    Actor-critic with PyTorch, similar to generic_tictactoe.py.
    Policy and value networks with configurable hidden layers.
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        scale=0.1,
        rng=None,
        policy_hidden_dims=(64, 32),
        value_hidden_dims=(64, 32),
        activation="relu",
        entropy_coeff=0.0,
    ):
        super().__init__()

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.activation = activation
        self.entropy_coeff = entropy_coeff

        # Policy network
        policy_layers = []
        prev_dim = state_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                policy_layers.append(nn.ReLU())
            elif activation == "tanh":
                policy_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, n_actions))
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network
        value_layers = []
        prev_dim = state_dim
        for hidden_dim in value_hidden_dims:
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                value_layers.append(nn.ReLU())
            elif activation == "tanh":
                value_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)

        # Use default PyTorch initialization (like generic_tictactoe.py)
        # Don't use custom init - it can hurt performance

        # Optimizer (will be set by BrainAgent if needed)
        self.optimizer = None

    def _to_tensor(self, x):
        """Convert numpy array to torch tensor."""
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        return x

    def _to_numpy(self, x):
        """Convert torch tensor to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def forward(self, x):
        """Forward pass for policy network (for compatibility)."""
        return self.policy_net(x)

    def policy_logits(self, state, legal_mask=None):
        """
        Get policy logits, optionally masked for legal actions.

        Args:
            state: State vector (numpy array or torch tensor)
            legal_mask: Optional mask array (0 for legal, -inf for illegal)
        """
        state_t = self._to_tensor(state)
        logits = self.policy_net(state_t)

        if legal_mask is not None:
            mask_t = self._to_tensor(legal_mask)
            logits = logits + mask_t

        return self._to_numpy(logits) if isinstance(state, np.ndarray) else logits

    def act(self, state, temperature=1.0, greedy=False, legal_mask=None):
        """
        Sample or choose action given state.

        Args:
            state: State vector (numpy array)
            temperature: Temperature for softmax (0 = deterministic)
            greedy: If True, choose best action deterministically
            legal_mask: Optional mask array (0 for legal, -inf for illegal)

        Returns:
            action: Selected action index (int)
            probs: Action probabilities (numpy array)
            value: State value estimate (float)
        """
        state_t = self._to_tensor(state)

        with torch.no_grad():
            logits = self.policy_net(state_t)

            # Apply legal action mask if provided
            if legal_mask is not None:
                mask_t = self._to_tensor(legal_mask)
                logits = logits + mask_t

                # Check if all actions are masked (all logits are -inf)
                # This can happen if the legal mask is invalid or all actions are illegal
                if torch.all(torch.isinf(logits) & (logits < 0)):
                    # Fallback: use uniform distribution over all actions
                    # This shouldn't happen in normal operation, but prevents NaN errors
                    probs = torch.ones_like(logits) / len(logits)
                    if greedy:
                        action = int(torch.argmax(probs).item())
                    else:
                        dist = distributions.Categorical(probs)
                        action = int(dist.sample().item())
                    value = self.value_net(state_t).item()
                    return action, self._to_numpy(probs), value

            if temperature <= 0:
                probs = torch.zeros_like(logits)
                probs[torch.argmax(logits)] = 1.0
            else:
                logits_scaled = logits / temperature
                probs = torch.softmax(logits_scaled, dim=0)

                # Check for NaN values in probs (shouldn't happen, but safety check)
                if torch.any(torch.isnan(probs)):
                    # Fallback to uniform distribution if softmax produces NaN
                    probs = torch.ones_like(logits) / len(logits)

            if greedy:
                action = int(torch.argmax(probs).item())
            else:
                dist = distributions.Categorical(probs)
                action = int(dist.sample().item())

            value = self.value_net(state_t).item()

        return action, self._to_numpy(probs), value

    def _forward_value(self, state):
        """Forward pass for value network. Returns value and activations for backprop."""
        state_t = self._to_tensor(state)
        value = self.value_net(state_t)
        return value, None  # PyTorch handles gradients automatically

    def update(self, state, action, reward, next_state, done, neuromodulators, base_lr=1e-2, legal_mask=None, entropy_coeff=None):
        """
        TD update of critic and policy using PyTorch autograd.
        NeuromodulatorState carries TD error as dopamine.

        Args:
            state: Current state (numpy array)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (numpy array)
            done: Whether episode is done (bool)
            neuromodulators: NeuromodulatorState instance
            base_lr: Base learning rate
            legal_mask: Optional legal action mask
            entropy_coeff: Optional entropy coefficient override

        Returns:
            td_error: TD error (float)
            value: Current state value (float)
            next_value: Next state value (float)
        """
        state_t = self._to_tensor(state)
        next_state_t = self._to_tensor(next_state)

        # Get value estimates
        value = self.value_net(state_t).squeeze()
        next_value = self.value_net(next_state_t).squeeze()
        if done:
            next_value = torch.tensor(0.0)

        # Convert to numpy for neuromodulator update
        value_np = value.item() if isinstance(value, torch.Tensor) else float(value)
        next_value_np = next_value.item() if isinstance(
            next_value, torch.Tensor) else float(next_value)

        td_error = neuromodulators.update(
            reward=reward,
            value=value_np,
            next_value=next_value_np,
            pred_error_norm=0.0,
        )

        # Compute learning rates with neuromodulator scaling
        lr_value = base_lr * (1.0 + abs(neuromodulators.norepinephrine))
        lr_policy = base_lr * (1.0 + abs(neuromodulators.dopamine))

        # Policy update
        logits = self.policy_net(state_t)

        # Apply legal mask if provided
        if legal_mask is not None:
            mask_t = self._to_tensor(legal_mask)
            logits = logits + mask_t

            # Check if all actions are masked
            if torch.all(torch.isinf(logits) & (logits < 0)):
                # All actions masked - use uniform distribution
                logits = torch.zeros_like(logits)

        probs = torch.softmax(logits, dim=0)

        # Safety check for NaN values
        if torch.any(torch.isnan(probs)):
            probs = torch.ones_like(logits) / len(logits)

        log_probs = torch.log(probs + 1e-10)

        # Policy gradient: -log_prob(action) * td_error
        policy_loss = -log_probs[action] * td_error

        # Add entropy bonus if enabled
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff
        if entropy_coeff > 0:
            entropy = -(probs * log_probs).sum()
            policy_loss = policy_loss - entropy_coeff * entropy

        # Value update
        value_target = reward + \
            (0.99 * next_value if not done else torch.tensor(0.0))
        value_loss = (value - value_target) ** 2

        # Combined loss
        total_loss = policy_loss + lr_value / lr_policy * value_loss

        # Backward pass
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=base_lr)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return td_error, value_np, next_value_np

    def update_reinforce(self, states, actions, rewards, legal_masks=None, entropy_coeff=None, lr=None):
        """
        REINFORCE-style update for episode-based learning.
        All transitions get the same final reward.

        Args:
            states: List of state arrays
            actions: List of action indices
            rewards: Final reward (scalar) or list of rewards
            legal_masks: Optional list of legal masks
            entropy_coeff: Optional entropy coefficient
            lr: Optional learning rate (if provided, creates/updates optimizer with this rate)
        """
        if len(states) == 0:
            return

        # Convert to tensors
        states_t = torch.stack([self._to_tensor(s) for s in states])
        actions_t = torch.tensor(actions, dtype=torch.long)

        # If rewards is scalar, use for all transitions
        if isinstance(rewards, (int, float)):
            rewards_t = torch.full((len(states),), float(rewards))
        else:
            rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # Forward pass
        logits = self.policy_net(states_t)

        # Apply legal masks if provided
        if legal_masks is not None:
            # Filter out None masks and convert to tensors
            valid_masks = []
            for m in legal_masks:
                if m is not None:
                    valid_masks.append(self._to_tensor(m))
                else:
                    # If mask is None, use all-zeros mask (all actions legal)
                    valid_masks.append(torch.zeros(self.n_actions))

            if valid_masks:
                masks_t = torch.stack(valid_masks)
                logits = logits + masks_t

            # Check for all-masked actions (all -inf) and handle gracefully
            # This shouldn't happen, but prevents NaN errors during training
            for i in range(logits.shape[0]):
                if torch.all(torch.isinf(logits[i]) & (logits[i] < 0)):
                    # All actions masked for this state - use uniform distribution
                    logits[i] = torch.zeros_like(logits[i])

        probs = torch.softmax(logits, dim=1)

        # Safety check for NaN values
        if torch.any(torch.isnan(probs)):
            # Replace NaN with uniform distribution
            probs = torch.where(torch.isnan(probs),
                                torch.ones_like(probs) / probs.shape[1],
                                probs)

        dist = distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)

        # REINFORCE: -log_prob(action) * reward (like generic_tictactoe.py)
        # In generic_tictactoe.py: loss -= logp * r (sums, doesn't mean!)
        # CRITICAL FIX: Use sum() to match generic_tictactoe.py exactly
        # Using mean() divides by batch size, weakening gradients for shorter episodes
        policy_loss = -(log_probs * rewards_t).sum()

        # Add entropy bonus (like generic_tictactoe.py: loss -= entropy_coeff * (-logp.exp() * logp))
        # In generic_tictactoe.py: loss -= entropy_coeff * (-logp.exp() * logp)
        # where logp is log_prob of selected action
        # -logp.exp() * logp = -exp(log_prob) * log_prob = -prob * log_prob
        # So: loss -= entropy_coeff * (-prob * log_prob) = loss += entropy_coeff * prob * log_prob
        # This adds entropy to the loss (since we minimize, this maximizes entropy = exploration)
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff
        entropy_summary = None
        if entropy_coeff > 0:
            # Match generic_tictactoe.py exactly: -entropy_coeff * (-logp.exp() * logp)
            # For each selected action: -entropy_coeff * (-exp(log_prob) * log_prob)
            # = entropy_coeff * prob * log_prob
            selected_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            # -logp.exp() * logp = -prob * log_prob for selected action
            # We want: loss -= entropy_coeff * (-prob * log_prob)
            entropy_term = -(selected_probs * log_probs)  # -prob * log_prob
            # Use sum() to match generic_tictactoe.py: loss -= entropy_coeff * (-logp.exp() * logp)
            entropy_summary = entropy_term.sum().detach().cpu().item()
            policy_loss = policy_loss - entropy_coeff * entropy_term.sum()

        # Backward pass
        # Create or update optimizer with correct learning rate
        if self.optimizer is None:
            # Use provided lr, or default to 0.001 if not provided
            optimizer_lr = lr if lr is not None else 0.001
            self.optimizer = optim.Adam(self.parameters(), lr=optimizer_lr)
        elif lr is not None:
            # Update existing optimizer's learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.optimizer.zero_grad()
        policy_loss.backward()

        # Check for gradient issues
        total_grad_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_grad_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5

        # Clip gradients if they're too large
        if total_grad_norm > 10.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Collect diagnostics for logging/monitoring
        with torch.no_grad():
            value_predictions = self.value_net(states_t).squeeze()
            value_loss_metric = torch.mean(
                (value_predictions - rewards_t) ** 2).item()
            reward_mean = rewards_t.mean().item()
            reward_std = rewards_t.std(
                unbiased=False).item() if len(rewards_t) > 1 else 0.0

        metrics = {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss_metric),
            "mean_reward": float(reward_mean),
            "std_reward": float(reward_std),
            "entropy_term": float(entropy_summary) if entropy_summary is not None else 0.0,
            "batch_size": len(states),
        }

        return metrics

    def to_state(self):
        """Save state for persistence."""
        return {
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "policy_hidden_dims": [self.policy_net[i].out_features
                                   for i in range(0, len(self.policy_net)-2, 2)],
            "value_hidden_dims": [self.value_net[i].out_features
                                  for i in range(0, len(self.value_net)-2, 2)],
            "activation": self.activation,
            "entropy_coeff": self.entropy_coeff,
            "rng_state": self.rng.get_state(),
        }

    @staticmethod
    def from_state(state):
        """Load state from saved dict."""
        rng = np.random.RandomState()
        rng.set_state(state["rng_state"])

        ac = ActorCritic(
            state_dim=state["state_dim"],
            n_actions=state["n_actions"],
            rng=rng,
            policy_hidden_dims=tuple(
                state.get("policy_hidden_dims", [64, 32])),
            value_hidden_dims=tuple(state.get("value_hidden_dims", [64, 32])),
            activation=state.get("activation", "relu"),
            entropy_coeff=state.get("entropy_coeff", 0.0),
        )

        ac.load_state_dict(state["state_dict"])
        return ac
