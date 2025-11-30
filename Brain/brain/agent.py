import numpy as np
import pickle
from .world_model import HierarchicalWorldModel
from .actor_critic import ActorCritic
from .memory import EpisodicMemory
from .neuromodulators import NeuromodulatorState
from .motivation import IntrinsicMotivation, DriveState


class BrainAgent:
    """
    Integrated agent:
    - optional multi-modal encoder
    - hierarchical predictive coding world model
    - actor-critic RL
    - neuromodulators
    - intrinsic motivation (curiosity + learning progress)
    - slow drives (persistent valence)
    - episodic memory + offline replay
    - continuous run with save/resume
    """

    def __init__(
        self,
        obs_dim=None,
        latent_dims=None,
        n_actions=4,
        encoder=None,
        lr_model=1e-3,
        lr_policy=1e-2,
        replay_batch_size=32,
        rng=None,
    ):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.encoder = encoder
        if self.encoder is not None:
            obs_dim_effective = self.encoder.output_dim
        else:
            if obs_dim is None:
                raise ValueError("Either obs_dim or encoder must be provided.")
            obs_dim_effective = obs_dim

        if latent_dims is None:
            latent_dims = [64, 32]

        self.world_model = HierarchicalWorldModel(obs_dim_effective, latent_dims, rng=rng)
        top_dim = latent_dims[-1]
        self.actor_critic = ActorCritic(top_dim, n_actions, rng=rng)
        self.memory = EpisodicMemory()
        self.neuromodulators = NeuromodulatorState()
        self.intrinsic = IntrinsicMotivation()
        self.drives = DriveState()
        self.lr_model = lr_model
        self.lr_policy = lr_policy
        self.replay_batch_size = replay_batch_size

        self.global_step = 0
        self.episode_index = 0

    # ----- input handling -----

    def _prepare_obs(self, obs):
        """
        Convert raw input to flat observation vector.
        """
        if self.encoder is not None:
            if not isinstance(obs, dict):
                raise ValueError("With encoder set, obs must be dict of modality->array.")
            return self.encoder.encode(obs)
        else:
            return np.asarray(obs).reshape(-1)

    def encode_state(self, obs):
        """
        Map raw observation to (latent z, encoded x).
        """
        x = self._prepare_obs(obs)
        z = self.world_model.encode_state(x)
        return z, x

    # ----- acting -----

    def act(self, obs, temperature=1.0, greedy=False):
        """
        Encode observation and sample action.
        """
        z, x = self.encode_state(obs)
        action, probs, value = self.actor_critic.act(z, temperature=temperature, greedy=greedy)
        return action, z, value, x

    # ----- online learning -----

    def online_update(self, obs, x, z, action, external_reward, next_obs, done):
        """
        Single step of learning integrating:
        - world model predictive coding
        - intrinsic motivation
        - drives
        - actor-critic with total valence
        """
        z_next, x_next = self.encode_state(next_obs)

        # Use NE + ACh to gate world model plasticity
        neuromod_factor = (
            0.5 * self.neuromodulators.norepinephrine
            + 0.5 * self.neuromodulators.acetylcholine
        )

        # Update world model and get prediction error
        _, pred_error_norm = self.world_model.learn(
            x=x, neuromod_factor=neuromod_factor, lr_model=self.lr_model
        )

        # Intrinsic reward from prediction error
        intrinsic_reward, components = self.intrinsic.compute(pred_error_norm)

        # Persistent drives from intrinsic components
        drive_vec = self.drives.update(components)

        # Drives scale intrinsic reward
        drive_gain = 1.0 + np.tanh(drive_vec.mean())
        total_reward = float(external_reward) + drive_gain * intrinsic_reward

        # Actor-critic update using total reward
        td_error, value, next_value = self.actor_critic.update(
            state=z,
            action=action,
            reward=total_reward,
            next_state=z_next,
            done=done,
            neuromodulators=self.neuromodulators,
            base_lr=self.lr_policy,
        )

        # Final neuromodulator update with prediction error included
        self.neuromodulators.update(
            reward=total_reward,
            value=value,
            next_value=next_value,
            pred_error_norm=pred_error_norm,
        )

        # Store transition
        self.memory.store(x, z, action, total_reward, x_next, z_next, done)

        self.global_step += 1

        return td_error, pred_error_norm, intrinsic_reward, drive_vec

    # ----- offline replay -----

    def offline_replay(self, n_batches=10):
        """
        Replay stored experiences for slow consolidation.
        """
        if len(self.memory.buffer) == 0:
            return

        for _ in range(n_batches):
            batch = self.memory.sample(self.replay_batch_size)
            for x, z, action, stored_reward, x_next, z_next, done in batch:
                replay_reward = 0.5 * stored_reward

                td_error, value, next_value = self.actor_critic.update(
                    state=z,
                    action=action,
                    reward=replay_reward,
                    next_state=z_next,
                    done=done,
                    neuromodulators=self.neuromodulators,
                    base_lr=self.lr_policy * 0.1,
                )

                neuromod_factor = (
                    0.5 * self.neuromodulators.norepinephrine
                    + 0.5 * self.neuromodulators.acetylcholine
                )
                self.world_model.learn(
                    x=x, neuromod_factor=neuromod_factor, lr_model=self.lr_model * 0.1
                )

    # ----- persistence -----

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ----- continuous life loop -----

    def run_continuous(
        self,
        env,
        max_steps=None,
        offline_every=100,
        offline_batches=10,
        save_path=None,
        save_every=None,
        temperature_schedule=None,
        greedy_after=None,
    ):
        """
        Continuous perception-action-learning loop.

        env: must implement reset() and step(action)
        max_steps: stop after this many steps (None = run until interrupted)
        offline_every: how often to call offline_replay
        save_path, save_every: periodic saving
        temperature_schedule(step) -> temperature
        greedy_after: step after which to act greedily
        """
        obs = env.reset()
        self.episode_index += 1
        step_in_episode = 0

        try:
            while True:
                if max_steps is not None and self.global_step >= max_steps:
                    break

                if temperature_schedule is None:
                    temperature = 1.0
                else:
                    temperature = float(temperature_schedule(self.global_step))

                greedy = greedy_after is not None and self.global_step >= greedy_after

                action, z, value, x = self.act(obs, temperature=temperature, greedy=greedy)
                next_obs, external_reward, done, info = env.step(action)

                td_error, pred_err, intrinsic, drives = self.online_update(
                    obs=obs,
                    x=x,
                    z=z,
                    action=action,
                    external_reward=external_reward,
                    next_obs=next_obs,
                    done=done,
                )

                obs = next_obs
                step_in_episode += 1

                if offline_every is not None and self.global_step % offline_every == 0:
                    self.offline_replay(n_batches=offline_batches)

                if save_path is not None and save_every is not None:
                    if self.global_step % save_every == 0:
                        self.save(save_path)

                if done:
                    obs = env.reset()
                    self.episode_index += 1
                    step_in_episode = 0

        except KeyboardInterrupt:
            if save_path is not None:
                self.save(save_path)


