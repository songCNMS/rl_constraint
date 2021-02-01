from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict
import numpy as np
import os
import ray
from ray.rllib.env import BaseEnv
from agents.inventory import SKUStoreUnit


class InventoryMetricCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.custom_metrics['env_episode_reward'] = 0

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # for env in base_env.get_unwrapped():
        env = base_env.get_unwrapped()[env_index]
        for f in env.world.facilities.values():
            if isinstance(f, SKUStoreUnit):
                episode.custom_metrics['env_episode_reward'] += f.step_reward
