import sys
sys.path.append('.')

import random
import numpy as np
import time
import os
from datetime import datetime
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
from utility.tools import SimulationTracker, list_to_figure
# from scheduler.inventory_tf_model import FacilityNet
from scheduler.inventory_torch_model import SKUStoreBatchNormModel as SKUStoreDNN
from scheduler.inventory_torch_model import SKUWarehouseBatchNormModel as SKUWarehouseDNN
from scheduler.inventory_torch_model import ConsumerRewardShapeModel, SKUStoreBatchNormModel
from explorer.stochastic_sampling import StochasticSampling
from env.inventory_callbacks import InventoryMetricCallbacks
from utility.tensorboard import TensorBoard


import torch
# SEED=7
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark = False

# Configuration ===============================================================================


def filter_keys(d, keys):
    return {k:v for k,v in d.items() if k in keys}

# Training Routines ===============================================================================

def print_training_results(result):
    keys = ['date', 'episode_len_mean', 'episodes_total', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        print(f"- {k}: {result[k]}")


def echelon_policy_map_fn(env, echelon, agent_id, incremental_level):
    facility_id = Utils.agentid_to_fid(agent_id)
    facility = env.world.facilities[facility_id]
    if Utils.is_producer_agent(agent_id):
        return 'baseline_producer'
    if isinstance(facility, FacilityCell):
        return 'baseline_consumer'
    else:
        agent_echelon = env.world.agent_echelon[facility_id]
        if  agent_echelon == 0: # supplier
            return 'baseline_consumer'
        elif agent_echelon == env.world.total_echelon - 1: # retailer
            # return 'baseline_consumer'
            return ('ppo_store_consumer' if ((not env.constraint_imposed) or (facility.constraint_automaton is None) or (incremental_level != 1)) else 'ppo_store_consumer_constrain')
        elif agent_echelon >= echelon: # warehouse and current layer is trainning or has been trained.
            return 'ppo_warehouse_consumer'
        else: # warehouse on layers that haven't been trained yet
            return 'baseline_consumer'


def create_ppo_trainer(env, config, echelon, n_iterations, args, incremental_level):
    policy_map_fn = (lambda x: echelon_policy_map_fn(env, echelon, x, incremental_level))
    for agent_id in env.agent_ids():
        print(agent_id, policy_map_fn(agent_id))
    
    if incremental_level == 0:
        ppo_store_consumer_name = ['ppo_store_consumer']
    elif incremental_level == 1:
        ppo_store_consumer_name = ['ppo_store_consumer_constrain']
    else:
        ppo_store_consumer_name = ['ppo_store_consumer']
    policies_to_train = (ppo_store_consumer_name if echelon == env.world.total_echelon -1 else ['ppo_warehouse_consumer'])
    ext_conf = ppo.DEFAULT_CONFIG.copy()
    ext_conf.update({
            "env": InventoryManageEnv,
            "framework": "torch",
            "num_workers": 0,
            "num_envs_per_worker": 2,
            "clip_rewards": True,
            "vf_share_layers": True,
            "vf_loss_coeff": 2.00,   
            "entropy_coeff": 0.00,
            # estimated max value of vf, used to normalization   
            "vf_clip_param": 10.0,
            "clip_param": 0.5, 
            "use_critic": True,
            "use_gae": True,
            "lambda": 1.0,
            "gamma": 0.95,
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
            "num_sgd_iter": 30,
            "shuffle_sequences": True,
            "lr": 1e-4,
            "_fake_gpus": False,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "callbacks": InventoryMetricCallbacks,
            "explore": True,
            "exploration_config": {
                "type": StochasticSampling,
                "random_timesteps": args.rollout_fragment_length*args.batch_size*20,
            },
            "multiagent": {
                "policies": filter_keys(policies, ['baseline_producer', 'baseline_consumer', 'ppo_store_consumer', 'ppo_store_consumer_constrain', 'ppo_warehouse_consumer']),
                "policy_mapping_fn": policy_map_fn,
                "policies_to_train": policies_to_train
            }
        })

    print(f"Environment: action space producer {env.action_space_producer}, action space consumer {env.action_space_consumer}, observation space {env.observation_space}")
    ppo_trainer = ppo.PPOTrainer(
        env = InventoryManageEnv,
        config = ext_conf)
    return ppo_trainer

def record_eval(stage, eval_mean_reward, eval_mean_reward_list, i):
    if stage == 0:
        writer.add_scalar('eval/eval_balance_stat/without_constraint', eval_mean_reward, i)
        for r in eval_mean_reward_list:
            writer.add_scalar('eval/eval_balance_hist/without_constraint', r, i)
    elif stage == 1:
        writer.add_scalar('eval/eval_balance_stat/with_constraint_incremental', eval_mean_reward, i)
        for r in eval_mean_reward_list:
            writer.add_scalar('train/eval_balance_hist/without_constraint', r, i)
    else:
        writer.add_scalar('eval/eval_balance_stat/with_constraint_restart', eval_mean_reward, i)
        for r in eval_mean_reward_list:
            writer.add_scalar('eval/eval_balance_hist/without_constraint', r, i)

def train_ppo(_env, n_iterations, step_iterations, ppo_trainer, workdir, is_constraint_imposed, stage, reward_shape):
    mean_reward_hist = []
    mean_reward_hist_wo_discount = []
    max_mean_reward = -np.inf
    max_mean_reward_wo_discount = -np.inf
    max_mean_reward_checkpoint = None
    policy_map_fn = ppo_trainer.config['multiagent']['policy_mapping_fn']
    policies_to_train = ppo_trainer.config['multiagent']['policies_to_train']

    policies_balance_list = []
    ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_constraint_imposed(is_constraint_imposed)))
    
    policies = {}
    for agent_id in _env.agent_ids():
        policy = ppo_trainer.get_policy(policy_map_fn(agent_id))
        policies[agent_id] = policy

    ppo_trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_policies(policies)))

    for i in range(n_iterations):
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_iteration(i, n_iterations)))
        
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_training_mode(True)))

        if reward_shape:
            ppo_trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.set_discount_training_mode(False)))
            for agent_id in _env.agent_ids():
                policy = ppo_trainer.get_policy(policy_map_fn(agent_id))
                if policy_map_fn(agent_id) in policies_to_train:
                    policy.model.discount_training = False

            for j in range(step_iterations):
                print("== Iteration", i, "==", " === Consumer Training step ", j, "=== Exp. Name ", policy_mode)
                train_result = ppo_trainer.train()
                print_training_results(train_result)
                
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_discount_training_mode(True)))
        for agent_id in _env.agent_ids():
            policy = ppo_trainer.get_policy(policy_map_fn(agent_id))
            if policy_map_fn(agent_id) in policies_to_train:
                policy.model.discount_training = True

        for j in range(step_iterations):
            print("== Iteration", i, "==", " === Discount Training step ", j, "=== Exp. Name ", policy_mode)
            train_result = ppo_trainer.train()
            policy_reward_mean, policy_reward_mean_list = render(_env.env_config, ppo_trainer, args, is_constraint_imposed, workdir, is_render=False)
            # mean_reward_hist.append(policy_reward_mean)
            print_training_results(train_result)
            better_result = False
            print('cur_reward: ', policy_reward_mean, ' best_reward: ', max_mean_reward)
            record_eval(stage, policy_reward_mean, policy_reward_mean_list, i)
            if ( train_result['timesteps_total'] > ppo_trainer.config['exploration_config']['random_timesteps']
                 and policy_reward_mean > max_mean_reward ):
                better_result = True
                max_mean_reward = policy_reward_mean
                checkpoint = ppo_trainer.save()
                max_mean_reward_checkpoint = checkpoint
                print("checkpoint saved at", checkpoint)
            mean_reward_hist.append(policy_reward_mean)
            if (better_result or (i % 10 == 0)):
                render(_env.env_config, ppo_trainer, args, is_constraint_imposed, workdir, is_render=True)

    # output results of the best policies
    ppo_trainer.restore(max_mean_reward_checkpoint)
    render(_env.env_config, ppo_trainer, args, is_constraint_imposed, workdir, is_render=True)
    return ppo_trainer, mean_reward_hist, max_mean_reward_checkpoint

def load_policy(trainer, agent_id):
    policy_map_fn = trainer.config['multiagent']['policy_mapping_fn']
    return trainer.get_policy(policy_map_fn(agent_id))

def render(config, trainer, args, is_constraint_imposed, workdir, is_render=False):
    # Create the environment
    evaluation_epoch_len = 60
    _env = InventoryManageEnv(config)
    _env.set_iteration(1, 1)
    _env.env_config.update({'episod_duration': evaluation_epoch_len})
    
    _env.set_constraint_imposed(is_constraint_imposed)
    _env.set_discount_training_mode(True)
    _env.set_training_mode(False)

    policies = {}
    policy_map_fn = trainer.config['multiagent']['policy_mapping_fn']
    for agent_id in env.agent_ids():
        policy = load_policy(trainer, agent_id)
        if policy_map_fn(agent_id) in ['ppo_store_consumer_constrain', 'ppo_store_consumer']:
            policy.model.discount_training = True
        policies[agent_id] = policy
    _env.set_policies(policies)

    trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_training_mode(False)))
    
    trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_discount_training_mode(True)))

    # Simulation loop
    
    if is_render:
        tracker = SimulationTracker(evaluation_epoch_len, 1, _env, policies)
        tracker.run_and_render(workdir, facility_types=[SKUStoreUnit, SKUWarehouseUnit])
        tracker.render('%s/plot_balance_store.png' % workdir, tracker.step_balances, [SKUStoreUnit])
        tracker.render('%s/plot_reward_store.png' % workdir, tracker.step_rewards, [SKUStoreUnit])

    track_metric_list = []
    evaluation_rounds = 10
    for _ in range(evaluation_rounds):
        tracker = SimulationTracker(evaluation_epoch_len, 1, _env, policies)
        tracker_metric, metric_list = tracker.run_wth_render(facility_types=[SKUStoreUnit])
        track_metric_list.append(tracker_metric)
    return np.mean(track_metric_list), track_metric_list

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
parser.add_argument("--rollout-fragment-length", type=int, default=35)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--min-batch-size", type=int, default=16)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--stop-iters", type=str, default="200,200,200")
parser.add_argument("--step-iters", type=int, default=1)
parser.add_argument("--stop-reward", type=float, default=1000.0)
parser.add_argument("--echelon-to-train", type=int, default=1)
parser.add_argument("--pt", type=int, default=0)
parser.add_argument("--early-stop-iters", type=int, default=0)
parser.add_argument("--config", type=str, default='random')
parser.add_argument("--exp-name", type=str, default=None)
parser.add_argument("--reward-shape", type=int, default=1)

from config.random_config import sku_config, supplier_config, warehouse_config, store_config, env_config, demand_sampler

def restart_ray(reward_shape):
    ray.init()
    if reward_shape:
        models.ModelCatalog.register_custom_model("sku_store_net", ConsumerRewardShapeModel)
        models.ModelCatalog.register_custom_model("sku_warehouse_net", ConsumerRewardShapeModel)
    else:
        models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreBatchNormModel)
        models.ModelCatalog.register_custom_model("sku_warehouse_net", SKUStoreBatchNormModel)
        

if __name__ == "__main__":
    args = parser.parse_args()
    supply_network_config = InventoryEnvironmentConfig()
    
    if args.config != 'random':
        supply_network_config.load_config(args.config)
    else:
        supply_network_config.sku_config = sku_config
        supply_network_config.supplier_config = supplier_config
        supply_network_config.warehouse_config = warehouse_config
        supply_network_config.store_config = store_config
        supply_network_config.env_config = env_config
        supply_network_config.demand_sampler = demand_sampler

    env_config_for_rendering = supply_network_config.env_config.copy()
    env_config_for_rendering['supply_network_config'] = supply_network_config
    reward_shape = (args.reward_shape == 1)
    env_config_for_rendering['reward_shape'] = reward_shape
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


    MyTorchPolicy = PPOTorchPolicy

    policies = {
            'baseline_producer': (ProducerBaselinePolicy, env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
            'baseline_consumer': (ConsumerBaselinePolicy, env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
            'ppo_producer': (MyTorchPolicy, env.observation_space, env.action_space_producer, ppo_policy_config_producer),
            'ppo_store_consumer_constrain': (MyTorchPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_store_consumer),
            'ppo_store_consumer': (MyTorchPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_store_consumer),
            'ppo_warehouse_consumer': (MyTorchPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_warehouse_consumer)
        }
    
    policy_mode = 'ppo_%s' % args.config
    if args.exp_name is not None:
        policy_mode += ('_' + args.exp_name)
    else:
        policy_mode += ('_' + datetime.today().strftime('%m_%d_%H_%M_%S'))

    total_echelon = env.world.total_echelon
    # number of echelon that to be trained, count from retailer
    # for instance, if echelon_to_be_trained == 2, it means that policies for products in stores and
    # warehouses closest to stores will be trained. 
    echelon_to_train = args.echelon_to_train
    
    stop_iters = args.stop_iters.split(',')

    train_hist = []
    train_hist_labels = []

    if args.pt:
        workdir = f"{os.environ['PT_OUTPUT_DIR']}/{policy_mode}/"
        checkpoint_path = f"{os.environ['PT_OUTPUT_DIR']}/{policy_mode}/checkpoints/"
    else:
        workdir = f"output/{policy_mode}/"
        checkpoint_path = f"output/{policy_mode}/checkpoints/"
    
    os.makedirs(workdir + '/train_log/', exist_ok=True)
    writer = TensorBoard(f'{workdir}/train_log/{args.exp_name}')

    os.makedirs(checkpoint_path, exist_ok=True)
    supply_network_config.save_config(workdir + '/config.py')

    restart_ray(reward_shape)
    # first train a model without constraints
    stage = 0
    if int(stop_iters[0]) > 0:
        env = InventoryManageEnv(env_config_for_rendering)
        env.constraint_imposed = False
        retailer_ppo_trainer_woc = create_ppo_trainer(env, env_config_for_rendering, total_echelon-1, int(stop_iters[0]), args, stage)
        if True:
            retailer_ppo_trainer_woc, mean_rewards, _ = train_ppo(env, int(stop_iters[0]), args.step_iters, retailer_ppo_trainer_woc, workdir+'/woc/', False, stage, reward_shape)
            retailer_ppo_trainer_woc.save(f"{checkpoint_path}/woc/")
            train_hist.append(mean_rewards)
        else:
            retailer_ppo_trainer_woc.restore('./output/ppo_tree_wth_rs/checkpoints/woc/checkpoint_40/checkpoint-40')
            train_hist.append([0.0])
        train_hist_labels.append('mean_reward_without_constraint')

        # train a model with constraints based on the trained model from the previous step
        if int(stop_iters[1]) > 0:
            stage = 1
            ray.shutdown()
            restart_ray(reward_shape)
            env = InventoryManageEnv(env_config_for_rendering)
            env.constraint_imposed = True
            retailer_ppo_trainer = create_ppo_trainer(env, env_config_for_rendering, total_echelon-1, int(stop_iters[1]), args, stage)
            # for policy_name in retailer_ppo_trainer.config['multiagent']['policies_to_train']:
            retailer_ppo_trainer.get_policy('ppo_store_consumer').set_weights(retailer_ppo_trainer_woc.get_policy('ppo_store_consumer').get_weights())
            retailer_ppo_trainer.get_policy('ppo_store_consumer_constrain').set_weights(retailer_ppo_trainer_woc.get_policy('ppo_store_consumer').get_weights())
            retailer_ppo_trainer, mean_rewards, _ = train_ppo(env, int(stop_iters[1]), args.step_iters, retailer_ppo_trainer, workdir+'/wc/', True, stage, reward_shape)
            retailer_ppo_trainer.save(f"{checkpoint_path}/wc/")
            train_hist.append(mean_rewards)
            train_hist_labels.append('mean_reward_with_constraint_incremental')
    
    ray.shutdown()
    restart_ray(reward_shape)
    # train with constraints from scratch
    if int(stop_iters[2]) > 0:
        stage = 2
        env = InventoryManageEnv(env_config_for_rendering)
        env.constraint_imposed = True
        retailer_ppo_trainer = create_ppo_trainer(env, env_config_for_rendering, total_echelon-1, int(stop_iters[2]), args, stage)
        retailer_ppo_trainer, mean_rewards, _ = train_ppo(env, int(stop_iters[2]), args.step_iters, retailer_ppo_trainer, workdir+'/wc_scratch/', True, stage, reward_shape)
        retailer_ppo_trainer.save(f"{checkpoint_path}/wc_scratch/")
        train_hist.append(mean_rewards)
        train_hist_labels.append('mean_reward_with_constraint_from_scratch')

    if len(train_hist) > 0:
        list_to_figure(train_hist, train_hist_labels, 'mean reward of policies', workdir + 'training_curve.png')