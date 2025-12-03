import numpy as np


class MultiModalEncoder:
    """
    Early fusion of multiple modalities into a single feature vector.
    Each modality: linear + tanh, then concatenation.
    """

    def __init__(self, modality_dims, hidden_dim_per_modality=32, scale=0.1, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.modalities = sorted(modality_dims.keys())
        self.hidden_dim_per_modality = hidden_dim_per_modality
        self.W = {}
        self.b = {}
        for name in self.modalities:
            in_dim = modality_dims[name]
            self.W[name] = scale * rng.randn(in_dim, hidden_dim_per_modality)
            self.b[name] = np.zeros(hidden_dim_per_modality)
        self.output_dim = hidden_dim_per_modality * len(self.modalities)

    def encode(self, inputs):
        """
        inputs: dict name -> np.array
        Missing modalities are treated as zeros.
        """
        h_list = []
        for name in self.modalities:
            x = inputs.get(name, None)
            if x is None:
                x = np.zeros(self.W[name].shape[0])
            x = np.asarray(x).reshape(-1)
            if x.shape[0] != self.W[name].shape[0]:
                raise ValueError(
                    f"Modality {name} expected dim {self.W[name].shape[0]}, got {x.shape[0]}"
                )
            h = np.tanh(x @ self.W[name] + self.b[name])
            h_list.append(h)
        return np.concatenate(h_list, axis=0)

    def to_state(self):
        return {
            "modalities": list(self.modalities),
            "hidden_dim_per_modality": self.hidden_dim_per_modality,
            "W": {k: v.copy() for k, v in self.W.items()},
            "b": {k: v.copy() for k, v in self.b.items()},
            "rng_state": self.rng.get_state(),
        }

    @staticmethod
    def from_state(state):
        modality_dims = {name: state["W"][name].shape[0] for name in state["modalities"]}
        encoder = MultiModalEncoder(
            modality_dims=modality_dims,
            hidden_dim_per_modality=state["hidden_dim_per_modality"],
        )
        encoder.W = {k: np.array(v, copy=True) for k, v in state["W"].items()}
        encoder.b = {k: np.array(v, copy=True) for k, v in state["b"].items()}
        encoder.rng.set_state(state["rng_state"])
        encoder.output_dim = state["hidden_dim_per_modality"] * len(state["modalities"])
        return encoder

