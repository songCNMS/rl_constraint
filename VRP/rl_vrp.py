import random
import numpy as np
import time
import os
import networkx as nx

import ray
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_tf, try_import_torch
import ray.rllib.agents.trainer_template as tt
from ray.rllib.models.tf.tf_action_dist import MultiCategorical
from ray.rllib.models import ModelCatalog
from functools import partial

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
import ray.rllib.agents.qmix.qmix as qmix
from ray.rllib.agents.qmix.qmix_policy import QMixTorchPolicy
import ray.rllib.env.multi_agent_env
import ray.rllib.models as models
from ray.rllib.policy.tf_policy_template import build_tf_policy
from gym.spaces import Box, Tuple, MultiDiscrete, Discrete

import ray.rllib.agents.trainer_template as tt
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing

from models.model import VRPModel
from config.config import vrp_config as env_config
from env.cvrp_env import CVRPEnv
from explorer.stochastic_sampling import StochasticSampling

import matplotlib
import matplotlib.pyplot as plt
from utils.utils import list_to_figure


# Model Configuration ===============================================================================

models.ModelCatalog.register_custom_model("vrp_model", VRPModel)


def print_training_results(result):
    keys = ['date', 'episode_len_mean', 'episodes_total', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        print(f"- {k}: {result[k]}")

    
def train_ppo(args, env, vrp_config, workdir, n_iterations):
    ext_conf = ppo.DEFAULT_CONFIG.copy()
    ext_conf.update({
            "num_workers": 2,
            "num_cpus_per_worker": 1,
            "vf_share_layers": True,
            "vf_loss_coeff": 1.0,      
            "vf_clip_param": 50.0,
            "use_critic": True,
            "use_gae": True,
            "framework": "torch",
            "lambda": 1.0,
            "gamma": 1.0,
            'env_config': vrp_config,
            'timesteps_per_iteration': vrp_config['episode_len'],
            'batch_mode': 'complete_episodes',
            # Size of batches collected from each worker
            "rollout_fragment_length": args.rollout,
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            "train_batch_size": args.batch_size*args.rollout,
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            "sgd_minibatch_size": args.min_batch_size*args.rollout,
            # Number of SGD iterations in each outer loop (i.e., number of epochs to
            # execute per train batch).
            "num_sgd_iter": 20,
            "shuffle_sequences": False,
            "lr": 1e-3,
            "_fake_gpus": True,
            "num_gpus": 0,
            "num_gpus_per_worker": 0,
            "model": {"custom_model": "vrp_model"},
            "explore": True,
            # "exploration_config": {
            #     # The Exploration class to use.
            #     "type": "EpsilonGreedy",
            #     # Config for the Exploration class' constructor:
            #     "initial_epsilon": 1.0,
            #     "final_epsilon": 0.02,
            #     "epsilon_timesteps": args.rollout*args.batch_size*args.iters,  # Timesteps over which to anneal epsilon.
            # },
            "exploration_config": {
                "type": StochasticSampling,
                "random_timesteps": args.rollout*args.batch_size*args.iters // 4,
            },
        })
    
    print(f"Environment: action space {env.action_space}, observation space {env.observation_space}")
    ppo_trainer = ppo.PPOTrainer(
        env = CVRPEnv,
        config = ext_conf)
    
    # ppo_trainer.restore('/root/ray_results/PPO_CVRPEnv_2020-12-29_11-50-29uylrljyr/checkpoint_100/checkpoint-100')
    
    mean_cost_list = []
    total_cost_list = []
    for i in range(n_iterations):
        print("== Iteration", i, "==")
        trainer_result = ppo_trainer.train()
        print_training_results(trainer_result)
        # cost = env.total_cost - (trainer_result['episode_reward_mean']*env.total_cost) / trainer_result['episode_len_mean']
        # cost = (1.0 - trainer_result['episode_reward_mean']/trainer_result['episode_len_mean']) * env.max_cost * env.num_nodes
        cost = trainer_result['episode_reward_mean']
        mean_cost_list.append(cost)
        print('cost: ', cost)
        if (i+1) % 5 == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)
            _total_cost = draw_route(args, ppo_trainer, env, mean_cost_list, workdir)
            total_cost_list.append(_total_cost)
    list_to_figure([total_cost_list], ['total_cost'], 'total_cost', f'{workdir}/rl_vrp_total_cost_{args.problem}.png')
    return ppo_trainer, mean_cost_list


import ray
from ray import tune
from tqdm import tqdm as tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--episod", type=int, default=365)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=150.0)
parser.add_argument("--rollout", type=int, default=160)
parser.add_argument("--batch-size", type=int, default=640)
parser.add_argument("--min-batch-size", type=int, default=64)
parser.add_argument("--problem", type=str, default="A-n32-k5")
parser.add_argument("--pt", type=int, default=0)

def rl_solution_to_graph(trainer, env):
    G = nx.DiGraph()
    position = {}
    route_edges = {}
    vehicle_id = -1
    state = env.reset()
    policy = trainer.get_policy()
    vrp_problem = env.vrp_problem
    total_cost = 0
    visited_node = set([env.depot])

    for node in range(env.num_nodes):
        position[node] = (vrp_problem.node_pos_x[node], vrp_problem.node_pos_y[node])
        route_load = env.demand[node]
        G.add_node(node, demand=route_load)

    # for _ in range(2*env.num_nodes):
    #     if vehicle_id == -1 or (env.cur_node == env.depot and len(route_edges[vehicle_id]) > 0):
    #         vehicle_id += 1
    #         route_edges[vehicle_id] = []
    #     retry = 0
    #     node_index = env.cur_node        
    #     route_load = env.demand[node_index]

    #     while (retry <= 10):
    #         retry += 1
    #         neighbors = env.find_next_nodes()
    #         action, _, _ = policy.compute_single_action( state, info={}, explore=True )
    #         next_node = neighbors[action][1]
    #         if env.validity_check(next_node):
    #             break
    #         # print('retry: ', retry, action)
    #         # if action == env.depot:
    #         #     validity = False
    #     if retry > 10:
    #         next_node = env.depot
    #         action = 0
    #     else:
    #         visited_node.add(next_node)
    #     state, _, _, _ = env.step(action)
    #     previous_node_index = node_index
    #     node_index = env.cur_node
    #     if previous_node_index != node_index:
    #         total_cost += env.cost_matrix[previous_node_index][node_index]
    #         route_edges[vehicle_id].append((previous_node_index, node_index))

    for _ in range(env.num_nodes):
        action, _, _ = policy.compute_single_action( state, info={}, explore=False )
        state, _, _, _ = env.step(action)
    
    for i in range(env.num_trucks):
        node_to_vehicle = env.node_to_vehicle[i, :]
        _cost, path = env.path_cost(node_to_vehicle)
        total_cost += _cost
        total_cost += env.cost_matrix[path[-1]][env.depot]
        route_edges[i] = []
        for n in range(len(path)-1):
            route_edges[i].append((path[n], path[n+1]))
        route_edges[i].append((path[-1], env.depot))
    
    print('ortool cost = ', env.get_ortool_value())
    print('total_cost: ', total_cost, ' total_vehicle: ', env.num_trucks)
    for vehicle_id in route_edges.keys():
        total_demand = 0
        route_str = f"{env.depot}--"
        for route in route_edges[vehicle_id]:
            total_demand += env.demand[route[1]]
            route_str += f"({env.demand[route[1]]})-->{route[1]}--"
        print('vehicle ', vehicle_id, ', total_demand: ', total_demand, ', capacity: ', env.vehicle_capacity)
        print(route_str)
    return G, position, route_edges, total_cost

def draw_route(args, trainer, env, mean_cost_list, workdir):
    plt.figure(figsize=(30,30))
    plt.axis("on")
    G, pos, route_edges, total_cost = rl_solution_to_graph(trainer, env)
    labels = {}
    for node in G.nodes():
        labels[node] = node
    nx.draw_networkx_nodes(G, pos, node_size=1000)
    nx.draw_networkx_labels(G, pos, labels, font_size=30, font_color="black")
    cmap = matplotlib.cm.get_cmap('Spectral')
    max_vehicle_id = np.max(list(route_edges.keys())) + 1.0
    for vehicle_id in route_edges.keys():
        if len(route_edges[vehicle_id]) <= 0:
            continue
        nx.draw_networkx_edges(G, pos, width=2, arrows=True, arrowsize=100,
                                edgelist=route_edges[vehicle_id], 
                                edge_color=cmap(vehicle_id/max_vehicle_id))
    
    plt.show()
    plt.savefig(f'{workdir}/rl_vrp_{args.problem}.png')
    list_to_figure([mean_cost_list], ['mean_cost'], 'mean_cost', f'{workdir}/rl_vrp_cost_{args.problem}.png')
    return total_cost


if __name__ == "__main__":
    args = parser.parse_args()
    vrp_config = env_config.copy()
    vrp_config.update({'problem': args.problem})
    env = CVRPEnv(vrp_config)

    if args.pt:
        workdir = f"{os.environ['PT_OUTPUT_DIR']}/"
    else:
        workdir = f"output/"
    os.makedirs(workdir, exist_ok=True)

    # Parameters of the tracing simulation
    # 'baseline' or 'trained'
    # env.reset()
    # for _ in range(100):
    #     a = random.randint(0, env.config['action_space_size']-1)
    #     next_node = env.find_next_nodes()[a][1]
    #     state, reward, _, _ = env.step(a)
    #     print(a, next_node, reward)
    #     for i in range(env.num_nodes):
    #         for j in range(env.num_nodes):
    #             if state[i*env.num_nodes+j] == 1:
    #                 print(i, '-->', j, '-->')
    #     print(state[(env.num_nodes*env.num_nodes):(env.num_nodes+1)*env.num_nodes])
    #     for i in range(env.num_nodes):
    #         if state[(env.num_nodes+1)*env.num_nodes + i] == 1:
    #             print('at: ', i)

    ray.init()
    env.reset()
    trainer, mean_cost_list = train_ppo(args, env, vrp_config, workdir, n_iterations = args.iters)