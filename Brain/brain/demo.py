import numpy as np
from .encoder import MultiModalEncoder
from .agent import BrainAgent
from .environment import MultiModalDummyEnv


def build_and_run_demo():
    modality_dims = {"vision": 16, "audio": 8, "text": 12}
    encoder = MultiModalEncoder(modality_dims, hidden_dim_per_modality=16)
    rng = np.random.RandomState(0)

    env = MultiModalDummyEnv(modality_dims, n_actions=4, rng=rng)
    agent = BrainAgent(
        encoder=encoder,
        latent_dims=[64, 32],
        n_actions=4,
        lr_model=1e-3,
        lr_policy=1e-2,
        replay_batch_size=32,
        rng=rng,
    )

    def temp_schedule(step):
        return max(0.5, 1.0 - step / 10000.0)

    agent.run_continuous(
        env,
        max_steps=500,
        offline_every=100,
        offline_batches=5,
        save_path=None,      # e.g. "brain_agent.pkl"
        save_every=None,
        temperature_schedule=temp_schedule,
        greedy_after=20000,
    )

    print("Global steps:", agent.global_step)
    print("Memory size:", len(agent.memory.buffer))
    print("Drives (curiosity, competence):", agent.drives.curiosity_drive, agent.drives.competence_drive)
    return agent


if __name__ == "__main__":
    build_and_run_demo()


