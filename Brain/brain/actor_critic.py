import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions


class ActorCritic(nn.Module):
    """
    Actor-critic with PyTorch.
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

        self.optimizer = None

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        return x

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def forward(self, x):
        return self.policy_net(x)

    def policy_logits(self, state, legal_mask=None):
        state_t = self._to_tensor(state)
        logits = self.policy_net(state_t)
        if legal_mask is not None:
            mask_t = self._to_tensor(legal_mask)
            logits = logits + mask_t
        return self._to_numpy(logits) if isinstance(state, np.ndarray) else logits

    def act(self, state, temperature=1.0, greedy=False, legal_mask=None):
        state_t = self._to_tensor(state)

        with torch.no_grad():
            logits = self.policy_net(state_t)

            if legal_mask is not None:
                mask_t = self._to_tensor(legal_mask)
                logits = logits + mask_t
                if torch.all(torch.isinf(logits) & (logits < 0)):
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
                if torch.any(torch.isnan(probs)):
                    probs = torch.ones_like(logits) / len(logits)

            if greedy:
                action = int(torch.argmax(probs).item())
            else:
                dist = distributions.Categorical(probs)
                action = int(dist.sample().item())

            value = self.value_net(state_t).item()

        return action, self._to_numpy(probs), value

    def _forward_value(self, state):
        state_t = self._to_tensor(state)
        value = self.value_net(state_t)
        return value, None

    def update(self, state, action, reward, next_state, done, neuromodulators, base_lr=1e-2, legal_mask=None, entropy_coeff=None, supervised_loss_coeff=5.0):
        """
        TD update of critic and policy.
        Does NOT update neuromodulator state (pure function w.r.t NM).
        
        Args:
            supervised_loss_coeff: Weight for supervised loss when only one action is available.
                                  Set to 0 to disable supervised loss.
        """
        state_t = self._to_tensor(state)
        next_state_t = self._to_tensor(next_state)

        value = self.value_net(state_t).squeeze()
        next_value = self.value_net(next_state_t).squeeze()
        if done:
            next_value = torch.tensor(0.0)

        value_np = value.item() if isinstance(value, torch.Tensor) else float(value)
        next_value_np = next_value.item() if isinstance(
            next_value, torch.Tensor) else float(next_value)

        # Use NM-modulated gamma
        gamma = neuromodulators.get_discount_factor() if hasattr(
            neuromodulators, 'get_discount_factor') else 0.99
        td_error = reward + gamma * next_value_np - value_np

        # Compute learning rates with neuromodulator scaling
        # NE (surprise) -> boosts global plasticity
        # DA (reward) -> boosts policy plasticity?
        # Using get_learning_rate_factor from new NM class
        lr_factor = 1.0
        if hasattr(neuromodulators, 'get_learning_rate_factor'):
            lr_factor = neuromodulators.get_learning_rate_factor()

        lr_value = base_lr * lr_factor
        lr_policy = base_lr * lr_factor

        # Policy update
        logits = self.policy_net(state_t)
        
        # Check if this is a forced action (only one valid action)
        # IMPORTANT: Detect BEFORE masking so we can compute supervised loss on unmasked probs
        is_forced_action = False
        forced_action_target = None
        if legal_mask is not None:
            mask_t = self._to_tensor(legal_mask)
            
            # Check if only one action is valid (BEFORE masking)
            valid_actions = torch.isfinite(mask_t) & (mask_t >= 0)
            num_valid = valid_actions.sum().item()
            
            if num_valid == 1:
                is_forced_action = True
                forced_action_target = torch.where(valid_actions)[0][0].item()
            
            # Now apply mask for action selection
            logits = logits + mask_t
            if torch.all(torch.isinf(logits) & (logits < 0)):
                logits = torch.zeros_like(logits)

        probs = torch.softmax(logits, dim=0)
        if torch.any(torch.isnan(probs)):
            probs = torch.ones_like(logits) / len(logits)
        log_probs = torch.log(probs + 1e-10)

        # Policy gradient
        policy_loss = -log_probs[action] * td_error
        
        # Add supervised loss for forced actions
        # CRITICAL: Compute on UNMASKED logits to teach policy the pattern
        if supervised_loss_coeff > 0 and is_forced_action and forced_action_target is not None:
            # Get unmasked logits and probabilities
            unmasked_logits = self.policy_net(state_t)
            unmasked_probs = torch.softmax(unmasked_logits, dim=0)
            # Cross-entropy loss: push policy towards the forced action (unmasked)
            supervised_loss = -torch.log(unmasked_probs[forced_action_target] + 1e-10)
            policy_loss = policy_loss + supervised_loss_coeff * supervised_loss

        # Entropy
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff

        # Modulate entropy with NM
        if hasattr(neuromodulators, 'get_exploration_entropy'):
            entropy_coeff *= neuromodulators.get_exploration_entropy()

        if entropy_coeff > 0:
            entropy = -(probs * log_probs).sum()
            policy_loss = policy_loss - entropy_coeff * entropy

        # Value update
        value_target = reward + \
            (gamma * next_value if not done else torch.tensor(0.0))
        value_loss = (value - value_target) ** 2

        total_loss = policy_loss + value_loss

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=base_lr)

        # Update optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_value  # Use modulated LR

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return td_error, value_np, next_value_np

    def update_reinforce(self, states, actions, rewards, legal_masks=None, entropy_coeff=None, lr=None, supervised_loss_coeff=5.0):
        """
        REINFORCE update with optional supervised loss for forced actions.
        
        Args:
            supervised_loss_coeff: Weight for supervised loss when only one action is available.
                                  Set to 0 to disable supervised loss.
        """
        if len(states) == 0:
            return

        states_t = torch.stack([self._to_tensor(s) for s in states])
        actions_t = torch.tensor(actions, dtype=torch.long)

        if isinstance(rewards, (int, float)):
            rewards_t = torch.full((len(states),), float(rewards))
        else:
            rewards_t = torch.tensor(rewards, dtype=torch.float32)

        logits = self.policy_net(states_t)

        # Track which states have forced actions (only one valid action)
        # IMPORTANT: Detect forced actions BEFORE masking, so we can compute
        # supervised loss on unmasked probabilities to teach the policy pattern
        forced_action_indices = []
        forced_action_targets = []
        
        if legal_masks is not None:
            valid_masks = []
            for idx, m in enumerate(legal_masks):
                if m is not None:
                    mask_t = self._to_tensor(m)
                    valid_masks.append(mask_t)
                    
                    # Check if this is a forced action (only one valid action)
                    # Valid actions have mask value >= 0 (typically 0.0)
                    valid_actions = torch.isfinite(mask_t) & (mask_t >= 0)
                    num_valid = valid_actions.sum().item()
                    
                    if num_valid == 1:
                        # Only one valid action - this is a forced action
                        forced_action_idx = torch.where(valid_actions)[0][0].item()
                        forced_action_indices.append(idx)
                        forced_action_targets.append(forced_action_idx)
                else:
                    valid_masks.append(torch.zeros(self.n_actions))
            if valid_masks:
                masks_t = torch.stack(valid_masks)
                logits = logits + masks_t
            for i in range(logits.shape[0]):
                if torch.all(torch.isinf(logits[i]) & (logits[i] < 0)):
                    logits[i] = torch.zeros_like(logits[i])

        probs = torch.softmax(logits, dim=1)
        if torch.any(torch.isnan(probs)):
            probs = torch.where(torch.isnan(probs), torch.ones_like(
                probs) / probs.shape[1], probs)

        dist = distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)

        policy_loss = -(log_probs * rewards_t).sum()

        # Add supervised loss for forced actions
        # CRITICAL: Compute on UNMASKED logits to teach policy the pattern
        supervised_loss_summary = 0.0
        if supervised_loss_coeff > 0 and len(forced_action_indices) > 0:
            # Get unmasked logits for forced action states
            unmasked_logits = self.policy_net(states_t[forced_action_indices])
            unmasked_probs = torch.softmax(unmasked_logits, dim=1)
            forced_targets_t = torch.tensor(forced_action_targets, dtype=torch.long, device=unmasked_probs.device)
            
            # Cross-entropy loss on UNMASKED probabilities: -log(prob of forced action)
            # This teaches the policy to recognize the board state and assign high prob
            # to the forced action even without masking
            forced_action_probs = unmasked_probs.gather(1, forced_targets_t.unsqueeze(1)).squeeze(1)
            supervised_loss = -torch.log(forced_action_probs + 1e-10).sum()
            supervised_loss_summary = supervised_loss.detach().cpu().item()
            policy_loss = policy_loss + supervised_loss_coeff * supervised_loss

        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff

        entropy_summary = None
        if entropy_coeff > 0:
            selected_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            entropy_term = -(selected_probs * log_probs)
            entropy_summary = entropy_term.sum().detach().cpu().item()
            policy_loss = policy_loss - entropy_coeff * entropy_term.sum()

        if self.optimizer is None:
            optimizer_lr = lr if lr is not None else 0.001
            self.optimizer = optim.Adam(self.parameters(), lr=optimizer_lr)
        elif lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.optimizer.zero_grad()
        policy_loss.backward()

        total_grad_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_grad_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5

        if total_grad_norm > 10.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)

        self.optimizer.step()

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
            "supervised_loss": float(supervised_loss_summary),
            "forced_actions_count": len(forced_action_indices),
            "batch_size": len(states),
        }
        return metrics

    def to_state(self):
        return {
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "policy_hidden_dims": [self.policy_net[i].out_features for i in range(0, len(self.policy_net)-2, 2)],
            "value_hidden_dims": [self.value_net[i].out_features for i in range(0, len(self.value_net)-2, 2)],
            "activation": self.activation,
            "entropy_coeff": self.entropy_coeff,
            "rng_state": self.rng.get_state(),
        }

    @staticmethod
    def from_state(state):
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
