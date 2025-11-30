"""
Brain Agent - A comprehensive neural agent implementation with:
- Multi-modal encoding
- Hierarchical predictive coding world model
- Actor-critic reinforcement learning
- Neuromodulators (dopamine, norepinephrine, acetylcholine)
- Intrinsic motivation (curiosity and learning progress)
- Episodic memory with replay
"""

from .utils import softmax
from .neuromodulators import NeuromodulatorState
from .world_model import PredictiveCodingLayer, HierarchicalWorldModel
from .encoder import MultiModalEncoder
from .actor_critic import ActorCritic
from .memory import EpisodicMemory
from .motivation import IntrinsicMotivation, DriveState
from .agent import BrainAgent
from .environment import MultiModalDummyEnv
from .demo import build_and_run_demo

__all__ = [
    "softmax",
    "NeuromodulatorState",
    "PredictiveCodingLayer",
    "HierarchicalWorldModel",
    "MultiModalEncoder",
    "ActorCritic",
    "EpisodicMemory",
    "IntrinsicMotivation",
    "DriveState",
    "BrainAgent",
    "MultiModalDummyEnv",
    "build_and_run_demo",
]


