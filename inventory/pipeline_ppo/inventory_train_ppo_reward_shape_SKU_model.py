from pickle import FALSE
import random
import numpy as np
import time
import os

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_torch
import ray.rllib.agents.trainer_template as tt
# from ray.rllib.models.tf.tf_action_dist import MultiCategorical
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical as MultiCategorical
from ray.rllib.models import ModelCatalog
from functools import partial

import ray.rllib.agents.ppo.ppo as ppo
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
import ray.rllib.agents.qmix.qmix as qmix
from ray.rllib.agents.qmix.qmix_policy import QMixTorchPolicy
import ray.rllib.env.multi_agent_env
import ray.rllib.models as models
from gym.spaces import Box, Tuple, MultiDiscrete, Discrete

import ray.rllib.agents.trainer_template as tt
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from env.inventory_env import InventoryManageEnv
from env.inventory_utils import Utils
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy
# from scheduler.inventory_random_policy import ConsumerBaselinePolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
# from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
# from scheduler.inventory_tf_model import FacilityNet
# from scheduler.inventory_torch_model import SKUStoreGRU, SKUWarehouseNet, SKUStoreDNN
# from scheduler.inventory_torch_model import SKUStoreBatchNormModel as SKUStoreDNN
from scheduler.inventory_torch_model import ConsumerRewardShapeModel
from config.inventory_config import env_config
from explorer.stochastic_sampling import StochasticSampling


# Configuration ===============================================================================


env_config_for_rendering = env_config.copy()
episod_duration = env_config_for_rendering['episod_duration']
env = InventoryManageEnv(env_config_for_rendering)


ppo_policy_config_store_consumer = {
    "model": {
        "fcnet_hiddens": [16, 16],
        "custom_model": "sku_store_net",
        # == LSTM ==
        "use_lstm": False,
        "max_seq_len": 14,
        "lstm_cell_size": 8, 
        "lstm_use_prev_action_reward": False
    }
}


# Model Configuration ===============================================================================
# models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreDNN)
# models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreGRU)
models.ModelCatalog.register_custom_model("sku_store_net", ConsumerRewardShapeModel)

MyTFPolicy = PPOTorchPolicy

policies = {}
policies_to_train = []

for agent_id in env.agent_ids():
    if Utils.is_producer_agent(agent_id):
        policies[agent_id] = (ProducerBaselinePolicy, env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env))
    elif agent_id.startswith('SKUStoreUnit') or agent_id.startswith('OuterSKUStoreUnit'):
        policies[agent_id] = (MyTFPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_store_consumer)
        policies_to_train.append(agent_id)
    else:
        policies[agent_id] = (ConsumerBaselinePolicy, env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env))

# Training Routines ===============================================================================

def print_training_results(result):
    keys = ['date', 'episode_len_mean', 'episodes_total', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        if isinstance(result[k], dict):
            result_to_print = {p: result[k][p] for p in policies_to_train}
            print(f"- {k}: {result_to_print}")
        else:
            print(f"- {k}: {result[k]}")


def policy_map_fn(agent_id):
    return agent_id
    
def train_ppo(args):
    ext_conf = ppo.DEFAULT_CONFIG.copy()
    ext_conf.update({
            "env": InventoryManageEnv,
            "framework": "torch",
            "num_workers": 4,
            "vf_share_layers": True,
            "vf_loss_coeff": 1.00,   
            # estimated max value of vf, used to normalization   
            "vf_clip_param": 100.0,
            "clip_param": 0.2, 
            "use_critic": True,
            "use_gae": True,
            "lambda": 1.0,
            "gamma": 0.9,
            'env_config': env_config_for_rendering,
            # Number of steps after which the episode is forced to terminate. Defaults
            # to `env.spec.max_episode_steps` (if present) for Gym envs.
            "horizon": args.episod,
            # Calculate rewards but don't reset the environment when the horizon is
            # hit. This allows value estimation and RNN state to span across logical
            # episodes denoted by horizon. This only has an effect if horizon != inf.
            "soft_horizon": False,
            # Minimum env steps to optimize for per train call. This value does
            # not affect learning, only the length of train iterations.
            'timesteps_per_iteration': 1000,
            'batch_mode': 'complete_episodes',
            # Size of batches collected from each worker
            "rollout_fragment_length": args.rollout_fragment_length,
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            "train_batch_size": args.rollout_fragment_length*args.batch_size,
            # Whether to shuffle sequences in the batch when training (recommended).
            "shuffle_sequences": True,
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            "sgd_minibatch_size": args.rollout_fragment_length*args.min_batch_size,
            # Number of SGD iterations in each outer loop (i.e., number of epochs to
            # execute per train batch).
            "num_sgd_iter": 20,
            "lr": 5e-4,
            "_fake_gpus": False,
            "num_gpus": 0.2,
            "explore": True,
            "exploration_config": {
                "type": StochasticSampling,
                "random_timesteps": 0, #args.rollout_fragment_length*args.batch_size*5,
            },
            "multiagent": {
                "policies":policies,
                "policy_mapping_fn": policy_map_fn,
                "policies_to_train": policies_to_train
            }
        })

    # print(f"Environment: action space producer {env.action_space_producer}, action space consumer {env.action_space_consumer}, observation space {env.observation_space}")

    ppo_trainer = ppo.PPOTrainer(
            env = InventoryManageEnv,
            config = ext_conf)
    
    # ppo_trainer.restore('/home/lesong/ray_results/PPO_InventoryManageEnv_2020-11-27_14-16-06_t0epjl_/checkpoint_52/checkpoint-52')

    max_reward_mean = -np.inf
    max_reward_mean_checkpoint = None
    for i in range(args.stop_iters):
        print("== Iteration", i, "==")
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_iteration(i, args.stop_iters)))
        
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_discount_training_mode(True)))
        for agent_id in env.agent_ids():
            policy = ppo_trainer.get_policy(policy_map_fn(agent_id))
            if policy_map_fn(agent_id) in ext_conf['multiagent']['policies_to_train']:
                policy.model.discount_training = True

        
        print_training_results(ppo_trainer.train())

        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_discount_training_mode(False)))
        
        for agent_id in env.agent_ids():
            policy = ppo_trainer.get_policy(policy_map_fn(agent_id))
            if policy_map_fn(agent_id) in ext_conf['multiagent']['policies_to_train']:
                policy.model.discount_training = False

        result = ppo_trainer.train()
        print_training_results(result)

        reward_mean = np.mean([result['policy_reward_mean'][p] for p in ext_conf['multiagent']['policies_to_train']])
        # if (i+1) % 10 == 0 or i == args.stop_iters - 1:
        print(reward_mean, max_reward_mean)
        if reward_mean > max_reward_mean:
            max_reward_mean = reward_mean
            checkpoint = ppo_trainer.save()
            max_reward_mean_checkpoint = checkpoint
            print("checkpoint saved at", checkpoint)
            if max_reward_mean > 0.0:
                render(ppo_trainer, args)
        # else:
        #     if max_reward_mean_checkpoint is not None:
        #         ppo_trainer.restore(max_reward_mean_checkpoint)
    return ppo_trainer


def load_policy(trainer, agent_id):
    trainer_config = trainer.config
    policy = trainer.get_policy(policy_map_fn(agent_id))
    if policy_map_fn(agent_id) in trainer_config['multiagent']['policies_to_train']:
        policy.model.discount_training = False
    return policy


def render(trainer, args):
# Create the environment
    evaluation_epoch_len = 60
    env.set_iteration(1, 1)
    env.discount_reward_training = False
    env.env_config.update({'episod_duration': evaluation_epoch_len})
    print(f"Environment: Producer action space {env.action_space_producer}, Consumer action space {env.action_space_consumer}, Observation space {env.observation_space}")
    
    policies = {}
    for agent_id in env.agent_ids():
        policies[agent_id] = load_policy(trainer, agent_id)

    # Simulation loop
    tracker = SimulationTracker(evaluation_epoch_len, 1, env, policies)
    
    if args.pt:
        loc_path = f"{os.environ['PT_OUTPUT_DIR']}/{policy_mode}/"
    else:
        loc_path = 'output/%s/' % policy_mode
    tracker.run_and_render(loc_path)



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--episod", type=int, default=episod_duration)
parser.add_argument("--rollout-fragment-length", type=int, default=14)
parser.add_argument("--batch-size", type=int, default=2560)
parser.add_argument("--min-batch-size", type=int, default=128)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--stop-iters", type=int, default=20)
parser.add_argument("--stop-reward", type=float, default=100.0)
parser.add_argument("--pt", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    # Parameters of the tracing simulation
    # 'baseline' or 'trained'
    
    policy_mode = 'reward_shape_serial'
    trainer = train_ppo(args)

    # model = trainer.get_policy(policy_map_fn('SKUStoreUnit_8c')).model
    # torch.save(model.state_dict(), 'SKUStoreUnit_8c.model')

    