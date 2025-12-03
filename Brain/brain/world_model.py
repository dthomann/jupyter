import numpy as np


class PredictiveCodingLayer:
    """
    One layer in a predictive coding hierarchy.
    """

    def __init__(self, input_dim, latent_dim, scale=0.1, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.W_down = scale * rng.randn(latent_dim, input_dim)

    def predict(self, z):
        """
        Predict input x_hat from latent z.
        """
        return np.tanh(z @ self.W_down)

    def infer_latent(self, x, z_init, n_steps=5, step_size=0.1):
        """
        Adjust z to reduce prediction error for fixed x.
        """
        z = z_init.copy()
        for _ in range(n_steps):
            x_hat = self.predict(z)
            error = x - x_hat
            d_tanh = 1.0 - x_hat * x_hat
            grad_z = -(error * d_tanh) @ self.W_down.T
            z = z - step_size * grad_z
        return z

    def update_weights_three_factor(self, x, z, neuromod_factor, lr):
        """
        Three-factor Hebbian-like update of W_down.
        """
        x_hat = self.predict(z)
        error = x - x_hat
        d_tanh = 1.0 - x_hat * x_hat
        local_signal = error * d_tanh
        delta_W = lr * neuromod_factor * np.outer(z, local_signal)
        self.W_down += delta_W
        return np.linalg.norm(error)

    def to_state(self):
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "W_down": self.W_down.copy(),
        }

    @staticmethod
    def from_state(state, rng=None):
        layer = PredictiveCodingLayer(
            input_dim=state["input_dim"],
            latent_dim=state["latent_dim"],
            rng=rng,
        )
        layer.W_down = np.array(state["W_down"], copy=True)
        return layer


class HierarchicalWorldModel:
    """
    Stack of predictive coding layers (cortical hierarchy).
    """

    def __init__(self, input_dim, latent_dims, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        dims = [input_dim] + latent_dims
        self.layers = []
        for l in range(len(latent_dims)):
            self.layers.append(
                PredictiveCodingLayer(
                    input_dim=dims[l], latent_dim=dims[l + 1], rng=rng
                )
            )

    def infer(self, x, n_steps_per_layer=5, step_size=0.1):
        """
        Infer latents bottom-up.
        """
        z_states = []
        current_input = x
        for layer in self.layers:
            z_init = np.zeros(layer.latent_dim)
            z = layer.infer_latent(
                current_input,
                z_init,
                n_steps=n_steps_per_layer,
                step_size=step_size,
            )
            z_states.append(z)
            current_input = z
        return z_states

    def learn(self, x, neuromod_factor, lr_model):
        """
        Run inference and update generative weights.
        Return mean prediction error norm.
        """
        z_states = self.infer(x)
        errors = []
        current_input = x
        for layer, z in zip(self.layers, z_states):
            err_norm = layer.update_weights_three_factor(
                current_input,
                z,
                neuromod_factor,
                lr_model,
            )
            errors.append(err_norm)
            current_input = z
        pred_error_norm = float(np.mean(errors)) if errors else 0.0
        return z_states, pred_error_norm

    def encode_state(self, x):
        """
        Return top-level latent as abstract state.
        """
        z_states = self.infer(x)
        return z_states[-1]

    def to_state(self):
        return [layer.to_state() for layer in self.layers]

    @staticmethod
    def from_state(state, rng=None):
        if len(state) == 0:
            return HierarchicalWorldModel(0, [], rng=rng)
        first = state[0]
        latent_dims = [layer_state["latent_dim"] for layer_state in state]
        model = HierarchicalWorldModel(first["input_dim"], latent_dims, rng=rng)
        for layer, layer_state in zip(model.layers, state):
            layer.W_down = np.array(layer_state["W_down"], copy=True)
        return model

