import sys
sys.path.append('.')

import os

import ray
import argparse
from datetime import datetime

from env.inventory_env import InventoryManageEnv
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy

# from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy

from config.inventory_config import env_config
from utility.visualization import visualization
from env.inventory_utils import Utils, InventoryEnvironmentConfig


from trainer.dqn import ConsumerDQNTorchPolicy
from trainer.dqn_reward_shape import ConsumerRewardShapeDQNTorchPolicy


from trainer.dqn_trainer import Trainer
import numpy as np
from utility.tensorboard import TensorBoard
# from scheduler.inventory_representation_learning_dqn import ConsumerRepresentationLearningDQNTorchPolicy
from utility.tools import SimulationTracker, list_to_figure
from agents.inventory import FacilityCell, SKUStoreUnit, SKUWarehouseUnit

import global_config
# Configuration ===============================================================================

dqn_config_default = {
    "env": InventoryManageEnv,
    "gamma": 0.99,
    "min_replay_history": 20000,
    "update_period": 4,
    "target_update_period": 2000,
    "epsilon_train": 0.02,
    # "epsilon_eval": 0.001,
    "lr": 0.001,
    # "num_iterations": 200,
    "training_steps": 25000,
    # "eval_steps": 500, eval one episode
    "max_steps_per_episode": 60,
    "replay_capacity": 1000000,
    "batch_size": 2048,
    "double_q": True,
    "use_unc_part": True,
    "pretrain": False,
    # "nstep": 1,

}

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--num-iterations", type=int, default=1000)
parser.add_argument("--visualization-frequency", type=int, default=100)
parser.add_argument("--echelon-to-train", type=int, default=1)
parser.add_argument("--stop-iters", type=str, default="20,20,20")
parser.add_argument("--exp-name", type=str, default=None)
parser.add_argument("--config", type=str, default='random')
parser.add_argument("--pt", type=int, default=0)
parser.add_argument("--reward-shape", type=int, default=0)

def echelon_policy_map_fn(env, echelon, agent_id, stage):
    facility_id = Utils.agentid_to_fid(agent_id)
    facility = env.world.facilities[facility_id]
    if Utils.is_producer_agent(agent_id):
        return 'baseline_producer', False
    if isinstance(facility, FacilityCell):
        return 'baseline_consumer', False
    else:
        agent_echelon = env.world.agent_echelon[facility_id]
        if  agent_echelon == 0: # supplier
            return 'baseline_consumer', False
        elif agent_echelon == env.world.total_echelon - 1: # retailer
            return (('dqn_store_consumer', (stage != 1))
                     if ((not env.constraint_imposed) or (facility.constraint_automaton is None)) 
                        else ('dqn_store_consumer_constrain', True))
        elif agent_echelon >= echelon: # warehouse and current layer is trainning or has been trained.
            return 'dqn_warehouse_consumer', False
        else: # warehouse on layers that haven't been trained yet
            return 'baseline_consumer', False

def print_result(result, writer, iter, policies_to_train, action_space):
    rewards_all = result['rewards_all']
    episode_reward_all = result['episode_reward_all']
    policies_to_train_loss = result['policies_to_train_loss']
    policies_to_train_qvalue = result['policies_to_train_qvalue']
    action_distribution = result['action_distribution']
    
    all_reward = []
    all_episode_reward = []
    all_loss = []
    all_qvalue = []
    all_action_distribution = {key: 0 for key in action_space}
    for agent_id, rewards in rewards_all.items():
        if agent_id in policies_to_train:
            all_reward.extend(rewards)
            all_loss.extend(policies_to_train_loss[agent_id])
            all_episode_reward.extend(episode_reward_all[agent_id])
            all_qvalue.extend(policies_to_train_qvalue[agent_id])
            for i in action_space:
                all_action_distribution[i] += action_distribution[agent_id][i]
    print(f"all_step: {result['all_step']}, average episode step: {result['episode_step']}")
    print(f"step_reward    max: {np.max(all_reward):13.6f} mean: {np.mean(all_reward):13.6f} min: {np.min(all_reward):13.6f}")
    print(f"episode_reward max: {np.max(all_episode_reward):13.6f} mean: {np.mean(all_episode_reward):13.6f} min: {np.min(all_episode_reward):13.6f}")
    print(f"loss           max: {np.max(all_loss):13.6f} mean: {np.mean(all_loss):13.6f} min: {np.min(all_loss):13.6f}")
    print(f"qvalue         max: {np.max(all_qvalue):13.6f} mean: {np.mean(all_qvalue):13.6f} min: {np.min(all_qvalue):13.6f}")
    print(f"action dist    {all_action_distribution}")
    print(f"epsilon        {result['epsilon']}")

    writer.add_scalar('train/step_reward', np.mean(all_reward), iter)
    writer.add_scalar('train/episode_reward', np.mean(all_episode_reward), iter)
    writer.add_scalar('train/loss', np.mean(all_loss), iter)
    writer.add_scalar('train/qvalue', np.mean(all_qvalue), iter)
    writer.add_scalar('train/epsilon', result['epsilon'], iter)

    sum_action = np.sum([v for v in all_action_distribution.values()])
    for i in action_space:
        writer.add_scalar(f'action/{i}', all_action_distribution[i]/sum_action, iter)
    return np.mean(all_reward)

def print_eval_result(result, writer, iter):
    print("  == evaluation result == ")
    rewards_all = result['rewards_all']
    episode_reward_all = result['episode_reward_all']
    all_reward = []
    all_episode_reward = []
    sum_agent = 0
    for agent_id, rewards in rewards_all.items():
        if agent_id.startswith('SKUStoreUnit') and Utils.is_consumer_agent(agent_id):
            sum_agent += 1
            if all_reward == []:
                all_reward = [x for x in rewards]
                all_episode_reward = [x for x in episode_reward_all[agent_id]]
            else :
                all_reward = [a+b for a, b in zip(all_reward, rewards)]
                all_episode_reward = [a+b for a,b in zip(all_episode_reward, episode_reward_all[agent_id])]
    all_reward = [x/sum_agent for x in all_reward]
    all_episode_reward = [x/sum_agent for x in all_episode_reward]
    print(f"all_step: {result['all_step']}, average episode step: {result['episode_step']}")
    print(f"step_reward    max: {np.max(all_reward):13.6f} mean: {np.mean(all_reward):13.6f} min: {np.min(all_reward):13.6f}")
    print(f"episode_reward max: {np.max(all_episode_reward):13.6f} mean: {np.mean(all_episode_reward):13.6f} min: {np.min(all_episode_reward):13.6f}")
    print(f"epsilon        {result['epsilon']}")

    writer.add_scalar('train/eval_step_reward', np.mean(all_reward), iter)
    writer.add_scalar('train/eval_episode_reward', np.mean(all_episode_reward), iter)
    writer.add_scalar('train/eval_epsilon', result['epsilon'], iter)

    return np.mean(all_episode_reward)

def render(config, trainer, args, workdir, is_render=False):
    # Create the environment
    evaluation_epoch_len = 60
    _env = InventoryManageEnv(config)
    _env.set_iteration(1, 1)
    _env.discount_reward_training = False
    _env.env_config.update({'episod_duration': evaluation_epoch_len})
    _env.set_constraint_imposed(env.constraint_imposed)
    _env.set_discount_training_mode(True)
    _env.set_training_mode(False)
    _env.set_policies(trainer.policies)
    
    # Simulation loop
    
    if is_render:
        tracker = SimulationTracker(evaluation_epoch_len, 1, _env, trainer.policies)
        tracker.run_and_render(workdir, facility_types=[SKUStoreUnit, SKUWarehouseUnit])
        tracker.render('%s/plot_balance_store.png' % workdir, tracker.step_balances, [SKUStoreUnit])
        tracker.render('%s/plot_reward_store.png' % workdir, tracker.step_rewards, [SKUStoreUnit])

    evaluation_rounds = 10
    track_metric_list = []
    for i in range(evaluation_rounds):
        tracker = SimulationTracker(evaluation_epoch_len, 1, _env, trainer.policies)
        tracker_metric, metric_list = tracker.run_wth_render(facility_types=[SKUStoreUnit])
        track_metric_list.append(tracker_metric)
    return np.mean(track_metric_list), track_metric_list


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

def create_dqn_trainer(env, stage):
    dqn_config = dqn_config_default.copy()
    dqn_config.update({
        "batch_size": 1024,
        "min_replay_history": 1024*50,
        "training_steps": 10240,
        "replay_capacity": 1024*100,
        "lr": 0.001,
        "update_period": 256,
        "target_update_period": 2560,
        "epsilon_train": 0.01,
        "use_unc_part": True,
        "use_cnn_state": False,
        "embeddingmerge": 'cat', # 'cat' or 'dot'
        "activation_func": 'sigmoid', # 'relu', 'sigmoid', 'tanh'
        "use_bn": False,
    })

    if dqn_config["use_cnn_state"]:
        global_config.use_cnn_state = True

    if not env.is_reward_shape_on:
        all_policies = {
            'baseline_producer': ProducerBaselinePolicy(env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
            'baseline_consumer': ConsumerBaselinePolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
            'dqn_warehouse_consumer': ConsumerDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config),
            'dqn_store_consumer': ConsumerDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config),
            'dqn_store_consumer_constrain': ConsumerDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config)
        }

    else:
        all_policies = {
            'baseline_producer': ProducerBaselinePolicy(env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
            'baseline_consumer': ConsumerBaselinePolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
            'dqn_warehouse_consumer': ConsumerRewardShapeDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config),
            'dqn_store_consumer': ConsumerRewardShapeDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config),
            'dqn_store_consumer_constrain': ConsumerRewardShapeDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config)
        }

    obss = env.reset()
    agent_ids = obss.keys()
    policies = {}
    policies_to_train = []

    for agent_id in agent_ids:
        policy, if_train = echelon_policy_map_fn(env, env.world.total_echelon - 1, agent_id, stage)
        policies[agent_id] = all_policies[policy]
        if if_train:
            policies_to_train.append(agent_id)

    env.set_policies(policies)
    dqn_trainer = Trainer(env, policies, policies_to_train, dqn_config)
    return dqn_trainer

def set_reward_shape(trainer, mode):
    trainer.env.set_discount_training_mode(mode)
    for agent_id in trainer.policies_to_train:
        trainer.policies[agent_id].discount_training = mode
        trainer.policies[agent_id].eval_net.discount_training = mode
        trainer.policies[agent_id].target_net.discount_training = mode


def train_dqn(env, dqn_trainer, num_iterations, args, workdir, stage):
    max_mean_reward = -np.inf
    debug = True
    policies = env.policies
    global_config.random_noise = False
    mean_reward_hist = []

    if env.is_reward_shape_on:
        action_space = []
        for i in range(env.action_space_consumer.nvec[0]):
            for j in range(env.action_space_consumer.nvec[1]):
                action_space.append((i,j))
    else:
        action_space = [i for i in range(env.action_space_consumer.n)]

    best_iter = -1
    for i in range(num_iterations):
        if env.is_reward_shape_on:
            print(f'====== Reward Discount Training ===== {policy_mode} ====== ')
            set_reward_shape(dqn_trainer, False)        
            result = dqn_trainer.train(i)
            print_result(result, writer, i, dqn_trainer.policies_to_train, action_space)

        print(f'====== Consumer Training ===== {policy_mode} ====== ')
        set_reward_shape(dqn_trainer, True)        
        result = dqn_trainer.train(i)
        print_result(result, writer, i, dqn_trainer.policies_to_train, action_space)

        # eval_result = dqn_trainer.eval(i)
        # eval_mean_reward = print_eval_result(eval_result, writer, i)
        eval_mean_reward, eval_mean_reward_list = render(env.env_config, dqn_trainer, args, workdir, is_render=False)
        # mean_reward_hist.append(eval_mean_reward)
        print('eval: ', eval_mean_reward, ' best: ', max_mean_reward)
        
        record_eval(stage, eval_mean_reward, eval_mean_reward_list, i)
        
        if eval_mean_reward > max_mean_reward or debug:
            if eval_mean_reward > max_mean_reward:
                best_iter = i
                dqn_trainer.save(args.exp_name, i)
                render(env.env_config, dqn_trainer, args, workdir, is_render=True)
            max_mean_reward = max(max_mean_reward, eval_mean_reward)
        mean_reward_hist.append(eval_mean_reward)
    if best_iter >= 0:
        dqn_trainer.restore(args.exp_name, best_iter)
        render(env.env_config, dqn_trainer, args, workdir, is_render=True)
    return dqn_trainer, mean_reward_hist

from config.random_config import sku_config, supplier_config, warehouse_config, store_config, env_config, demand_sampler

def restart_ray():
    ray.init()
    # models.ModelCatalog.register_custom_model("sku_store_net", ConsumerRewardShapeModel)
    # models.ModelCatalog.register_custom_model("sku_warehouse_net", ConsumerRewardShapeModel)

if __name__ == "__main__":
    restart_ray()
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
    env_config_for_rendering['reward_shape'] = (args.reward_shape == 1)
    env_config_for_rendering['evaluation_len'] = 60
    env = InventoryManageEnv(env_config_for_rendering)
    env.set_discount_training_mode(True)

    policy_mode = 'dqn_%s' % args.config
    if args.exp_name is not None:
        policy_mode += ('_' + args.exp_name)
    else:
        policy_mode += ('_' + datetime.today().strftime('%m_%d_%H_%M_%S'))
    args.exp_name = policy_mode

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

    stage = 0
    env.set_constraint_imposed(False)
    env.set_training_mode(True)
    env.set_discount_training_mode(True)
    dqn_trainer = create_dqn_trainer(env, stage)
    retailer_dqn_trainer_woc, mean_rewards = train_dqn(env, dqn_trainer, int(stop_iters[stage]), args, workdir+'/woc/', stage)
    train_hist.append(mean_rewards)
    train_hist_labels.append('mean reward without constraints')

    stage = 1
    env.set_constraint_imposed(True)
    env.set_training_mode(True)
    env.set_discount_training_mode(True)
    retailer_dqn_trainer_wc = create_dqn_trainer(env, stage)
    for agent_id in retailer_dqn_trainer_woc.policies_to_train:
        retailer_dqn_trainer_wc.get_policy(agent_id).set_weights(retailer_dqn_trainer_woc.get_policy(agent_id).get_weights())
        retailer_dqn_trainer_wc.get_policy(agent_id).evaluation = True
    
    retailer_dqn_trainer_wc, mean_rewards = train_dqn(env, retailer_dqn_trainer_wc, int(stop_iters[stage]), args, workdir+'/wc/', stage)
    train_hist.append(mean_rewards)
    train_hist_labels.append('mean reward with constraints incremental training')
    
    stage = 2
    env.set_constraint_imposed(True)
    env.set_training_mode(True)
    env.set_discount_training_mode(True)
    dqn_trainer = create_dqn_trainer(env, stage)
    retailer_dqn_trainer_restart, mean_rewards = train_dqn(env, dqn_trainer, int(stop_iters[stage]), args, workdir+'/wc_scratch/', stage)
    train_hist.append(mean_rewards)
    train_hist_labels.append('mean reward with constraints training from scratch')

    if len(train_hist) > 0:
        list_to_figure(train_hist, train_hist_labels, 'mean reward of policies', workdir + 'training_curve.png')

