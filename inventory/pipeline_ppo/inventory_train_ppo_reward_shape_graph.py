import random
import numpy as np
import time
import os

import ray
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
from render.inventory_renderer import AsciiWorldRenderer
from env.inventory_env import InventoryManageEnv
from agents.inventory import FacilityCell, SKUStoreUnit, SKUWarehouseUnit
from env.inventory_utils import Utils, InventoryEnvironmentConfig
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy
# from scheduler.inventory_random_policy import ConsumerBaselinePolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
# from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
# from scheduler.inventory_tf_model import FacilityNet
from scheduler.inventory_torch_model import SKUStoreBatchNormModel as SKUStoreDNN
from scheduler.inventory_torch_model import SKUWarehouseBatchNormModel as SKUWarehouseDNN
from scheduler.inventory_torch_model import ConsumerRewardShapeModel
from config.inventory_config import env_config
from explorer.stochastic_sampling import StochasticSampling

import torch
SEED=7
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

# Configuration ===============================================================================


def filter_keys(d, keys):
    return {k:v for k,v in d.items() if k in keys}

# Training Routines ===============================================================================

def print_training_results(result):
    keys = ['date', 'episode_len_mean', 'episodes_total', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        print(f"- {k}: {result[k]}")


def echelon_policy_map_fn(env, echelon, agent_id):
    facility_id = Utils.agentid_to_fid(agent_id)
    if Utils.is_producer_agent(agent_id):
        return 'baseline_producer'
    if isinstance(env.world.facilities[facility_id], FacilityCell):
        return 'baseline_consumer'
    else:
        agent_echelon = env.world.agent_echelon[facility_id]
        if  agent_echelon == 0: # supplier
            return 'baseline_consumer'
        elif agent_echelon == env.world.total_echelon - 1: # retailer
            # return 'baseline_consumer'
            return 'ppo_store_consumer'
        elif agent_echelon >= echelon: # warehouse and current layer is trainning or has been trained.
            return 'ppo_warehouse_consumer'
        else: # warehouse on layers that haven't been trained yet
            return 'baseline_consumer'


def create_ppo_trainer(env, config, echelon, args):
    policy_map_fn = (lambda x: echelon_policy_map_fn(env, echelon, x))
    for agent_id in env.agent_ids():
        print(agent_id, policy_map_fn(agent_id))
    policies_to_train = (['ppo_store_consumer'] if echelon == env.world.total_echelon -1 else ['ppo_warehouse_consumer'])
    ext_conf = ppo.DEFAULT_CONFIG.copy()
    ext_conf.update({
            "env": InventoryManageEnv,
            "framework": "torch",
            "num_workers": 0,
            "vf_share_layers": True,
            "vf_loss_coeff": 1.00,   
            # estimated max value of vf, used to normalization   
            "vf_clip_param": 100.0,
            "clip_param": 0.2, 
            "use_critic": True,
            "use_gae": True,
            "lambda": 1.0,
            "gamma": 0.99,
            'env_config': config,
            # Number of steps after which the episode is forced to terminate. Defaults
            # to `env.spec.max_episode_steps` (if present) for Gym envs.
            "horizon": config['episod_duration'],
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
                "random_timesteps": args.rollout_fragment_length*args.batch_size*10,
            },
            "multiagent": {
                "policies": filter_keys(policies, ['baseline_producer', 'baseline_consumer', 'ppo_store_consumer', 'ppo_warehouse_consumer']),
                "policy_mapping_fn": policy_map_fn,
                "policies_to_train": policies_to_train
            }
        })

    print(f"Environment: action space producer {env.action_space_producer}, action space consumer {env.action_space_consumer}, observation space {env.observation_space}")
    ppo_trainer = ppo.PPOTrainer(
        env = InventoryManageEnv,
        config = ext_conf)
    return ppo_trainer

def train_ppo(env, n_iterations, step_iterations, ppo_trainer):
    max_mean_reward = -np.inf
    max_mean_reward_checkpoint = None
    policy_map_fn = ppo_trainer.config['multiagent']['policy_mapping_fn']
    policies_to_train = ppo_trainer.config['multiagent']['policies_to_train']
    for i in range(n_iterations):
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_iteration(i, n_iterations)))
        
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_discount_training_mode(False)))
        for agent_id in env.agent_ids():
            policy = ppo_trainer.get_policy(policy_map_fn(agent_id))
            if policy_map_fn(agent_id) in policies_to_train:
                policy.model.discount_training = False

        for j in range(step_iterations):
            print("== Iteration", i, "==", " === Consumer Training step ", j)
            print_training_results(ppo_trainer.train())
        
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_discount_training_mode(True)))
        
        for agent_id in env.agent_ids():
            policy = ppo_trainer.get_policy(policy_map_fn(agent_id))
            if policy_map_fn(agent_id) in policies_to_train:
                policy.model.discount_training = True

        for j in range(step_iterations):
            print("== Iteration", i, "==", " === Discount Training step ", j)
            train_result = ppo_trainer.train()
            policy_reward_mean = np.sum([train_result['policy_reward_mean'].get(key, 0) 
                                            for key in policies_to_train])
            print(max_mean_reward, policy_reward_mean)
            print_training_results(train_result)
            better_result = False
            if policy_reward_mean > max_mean_reward:
                better_result = True
                max_mean_reward = policy_reward_mean
                checkpoint = ppo_trainer.save()
                max_mean_reward_checkpoint = checkpoint
                print("checkpoint saved at", checkpoint)
            if better_result or i % 10 == 0:
                render(env.env_config, ppo_trainer, args)
    # output results of the best policies
    ppo_trainer.restore(max_mean_reward_checkpoint)
    render(env.env_config, ppo_trainer, args)
    return ppo_trainer

def load_policy(trainer, agent_id):
    policy_map_fn = trainer.config['multiagent']['policy_mapping_fn']
    return trainer.get_policy(policy_map_fn(agent_id))

def render(config, trainer, args):
    # Create the environment
    evaluation_epoch_len = 60
    _env = InventoryManageEnv(config)
    _env.set_iteration(1, 1)
    _env.discount_reward_training = False
    _env.env_config.update({'episod_duration': evaluation_epoch_len})
    
    policies = {}
    for agent_id in env.agent_ids():
        policy = load_policy(trainer, agent_id)
        if hasattr(policy, "model"):
            policy.model.discount_training = False
        policies[agent_id] = policy

    # Simulation loop
    tracker = SimulationTracker(evaluation_epoch_len, 1, _env, policies)
    
    if args.pt:
        loc_path = f"{os.environ['PT_OUTPUT_DIR']}/{policy_mode}/"
    else:
        loc_path = 'output/%s/' % policy_mode
    tracker.run_and_render(loc_path, facility_types=[SKUStoreUnit, SKUWarehouseUnit])

    tracker.render('%s/plot_balance_store.png' % loc_path, tracker.step_balances, [SKUStoreUnit])
    tracker.render('%s/plot_reward_store.png' % loc_path, tracker.step_rewards, [SKUStoreUnit])


import ray
from ray import tune
from tqdm import tqdm as tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--episod", type=int, default=100)
parser.add_argument("--rollout-fragment-length", type=int, default=14)
parser.add_argument("--batch-size", type=int, default=2560)
parser.add_argument("--min-batch-size", type=int, default=128)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--stop-iters", type=str)
parser.add_argument("--step-iters", type=int, default=10)
parser.add_argument("--stop-reward", type=float, default=1000.0)
parser.add_argument("--echelon-to-train", type=int, default=2)
parser.add_argument("--pt", type=int, default=0)
parser.add_argument("--config", type=str, default='vanilla')

if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    supply_network_config = InventoryEnvironmentConfig()
    supply_network_config.load_config(args.config)
    env_config = supply_network_config.env_config
    env_config_for_rendering = env_config.copy()
    env_config_for_rendering['supply_network_config'] = supply_network_config
    env = InventoryManageEnv(env_config_for_rendering)

    ppo_policy_config_producer = {
        "model": {
            "fcnet_hiddens": [128, 128],
            "custom_model": "facility_net"
        }
    }

    ppo_policy_config_store_consumer = {
        "model": {
            "fcnet_hiddens": [16, 16],
            "custom_model": "sku_store_net",
            # == LSTM ==
            "use_lstm": False,
            "max_seq_len": 14,
            "lstm_cell_size": 128, 
            "lstm_use_prev_action_reward": False
        }
    }

    ppo_policy_config_warehouse_consumer = {
        "model": {
            "fcnet_hiddens": [16, 16],
            "custom_model": "sku_warehouse_net",
            # == LSTM ==
            "use_lstm": False,
            "max_seq_len": 14,
            "lstm_cell_size": 128, 
            "lstm_use_prev_action_reward": False
        }
    }

    # Model Configuration ===============================================================================
    # models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreDNN)
    # models.ModelCatalog.register_custom_model("sku_warehouse_net", SKUWarehouseDNN)
    models.ModelCatalog.register_custom_model("sku_store_net", ConsumerRewardShapeModel)
    models.ModelCatalog.register_custom_model("sku_warehouse_net", ConsumerRewardShapeModel)

    MyTorchPolicy = PPOTorchPolicy

    policies = {
            'baseline_producer': (ProducerBaselinePolicy, env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
            'baseline_consumer': (ConsumerBaselinePolicy, env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
            'ppo_producer': (MyTorchPolicy, env.observation_space, env.action_space_producer, ppo_policy_config_producer),
            'ppo_store_consumer': (MyTorchPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_store_consumer),
            'ppo_warehouse_consumer': (MyTorchPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_warehouse_consumer)
        }
    
    policy_mode = 'wo_constraint_vanilla'
    total_echelon = env.world.total_echelon
    # number of echelon that to be trained, count from retailer
    # for instance, if echelon_to_be_trained == 2, it means that policies for products in stores and
    # warehouses closest to stores will be trained. 
    echelon_to_train = args.echelon_to_train
    
    stop_iters = args.stop_iters.split(',')

    retailer_ppo_trainer = create_ppo_trainer(env, env_config_for_rendering, total_echelon-1, args)
    retailer_ppo_trainer = train_ppo(env, int(stop_iters[0]), args.step_iters, retailer_ppo_trainer)
    retailer_policy_weight = retailer_ppo_trainer.get_policy('ppo_store_consumer').get_weights()
    pre_warehouse_trainer_weight = None
    for i, echelon in enumerate(range(total_echelon-2, total_echelon-echelon_to_train-1, -1)):
        env.reset()
        warehouse_ppo_trainer = create_ppo_trainer(env, env_config_for_rendering, echelon, args)
        warehouse_ppo_trainer.get_policy('ppo_store_consumer').set_weights(retailer_policy_weight)
        if pre_warehouse_trainer_weight is not None:
            warehouse_ppo_trainer.get_policy('ppo_warehouse_consumer').set_weights(pre_warehouse_trainer_weight)
        warehouse_ppo_trainer = train_ppo(env, int(stop_iters[i+1]), args.step_iters, warehouse_ppo_trainer)
        pre_warehouse_trainer_weight = warehouse_ppo_trainer.get_policy('ppo_warehouse_consumer').get_weights()