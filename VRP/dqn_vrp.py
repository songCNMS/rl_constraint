import os
import numpy as np
from utils.tensorboard import TensorBoard
from env.cvrp_env import CVRPSiteEnv as CVRPEnv
from models.dqn import Trainer
from config.config import vrp_dqn_config as env_config
from models.dqn import DQNTorchPolicy
from utils.utils import list_to_figure
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from colorama import Fore, Back, Style 



dqn_config_default = {
    "env": CVRPEnv,
    "gamma": 1.0,
    "min_replay_history": 20000,
    "update_period": 4,
    "target_update_period": 2000,
    "epsilon_train": 0.02,
    "lr": 0.001,
    "training_steps": 25000,
    "max_steps_per_episode": 60,
    "replay_capacity": 1000000,
    "batch_size": 2048,
    "double_q": True
}

def print_result(action_space_size, result, writer, iter):
    all_reward = result['rewards_all']
    all_episode_reward = result['episode_reward_all']
    all_loss = result['train_loss']
    all_qvalue = result['train_qvalue']
    all_action_distribution = result['action_distribution']

    print(f"step_reward    max: {np.max(all_reward):13.6f} mean: {np.mean(all_reward):13.6f} min: {np.min(all_reward):13.6f}")
    print(f"episode_reward max: {np.max(all_episode_reward):13.6f} mean: {np.mean(all_episode_reward):13.6f} min: {np.min(all_episode_reward):13.6f}")
    print(f"loss           max: {np.max(all_loss):13.6f} mean: {np.mean(all_loss):13.6f} min: {np.min(all_loss):13.6f}")
    print(f"qvalue         max: {np.max(all_qvalue):13.6f} mean: {np.mean(all_qvalue):13.6f} min: {np.min(all_qvalue):13.6f}")
    print(f"action dist    {all_action_distribution}")
    print(f"epsilon        {result['epsilon']:13.6f}")

    writer.add_scalar('train/step_reward', np.mean(all_reward), iter)
    writer.add_scalar('train/episode_reward', np.mean(all_episode_reward), iter)
    writer.add_scalar('train/loss', np.mean(all_loss), iter)
    writer.add_scalar('train/qvalue', np.mean(all_qvalue), iter)
    writer.add_scalar('train/epsilon', result['epsilon'], iter)

    sum_action = sum(all_action_distribution)
    for i in range(action_space_size):
        writer.add_scalar(f'action/{i}', all_action_distribution[i]/sum_action, iter)
    return np.mean(all_episode_reward)


def rl_solution_to_graph(trainer, env):
    G = nx.DiGraph()
    position = {}
    route_edges = {}
    vehicle_id = -1
    state, infos = env.reset()
    policy = trainer.get_policy()
    vrp_problem = env.vrp_problem
    total_cost = 0

    for node in range(env.num_nodes):
        position[node] = (vrp_problem.node_pos_x[node], vrp_problem.node_pos_y[node])
        route_load = env.demand[node]
        G.add_node(node, demand=route_load)

    dones = False
    while not dones:
        action, _, _ = policy.compute_single_action( state, info=infos, explore=False )
        state, _, dones, infos = env.step(action)
    
    for i in range(env.num_trucks):
        node_to_vehicle = env.node_to_vehicle[i, :]
        _cost, path = env.path_cost(node_to_vehicle)
        total_cost += _cost
        total_cost += env.cost_matrix[path[-1]][env.depot]
        route_edges[i] = []
        for n in range(len(path)-1):
            route_edges[i].append((path[n], path[n+1]))
        route_edges[i].append((path[-1], env.depot))
    
    valid_route = True
    print('ortool cost = ', env.get_ortool_value())
    print('total_cost: ', total_cost, ' total_vehicle: ', env.num_trucks)
    total_fulfilled_demand = 0
    for vehicle_id in route_edges.keys():
        total_demand = 0
        route_str = f"{env.depot}--"
        for route in route_edges[vehicle_id]:
            total_demand += env.demand[route[1]]
            route_str += f"({env.demand[route[1]]})-->{route[1]}--"
        total_fulfilled_demand += total_demand
        if total_demand <= env.vehicle_capacity:
            print(f"vehicle: {vehicle_id}, total_demand: {total_demand}, capacity: {env.vehicle_capacity}")
        else:
            valid_route = False
            print(Back.GREEN + f"vehicle: {vehicle_id}, total_demand: {total_demand}, capacity: {env.vehicle_capacity}") 
            print(Style.RESET_ALL)
        print(route_str)
    if np.sum(env.demand) != total_fulfilled_demand:
        # valid_route = False
        print(Back.GREEN + f"total_demand_in_network: {np.sum(env.demand)}, total_fulfilled_demand: {total_fulfilled_demand}")
        print(Style.RESET_ALL)
    else:
        print(f"total_demand_in_network: {np.sum(env.demand)}, total_fulfilled_demand: {total_fulfilled_demand}")
    
    return G, position, route_edges, total_cost, valid_route

def draw_route(args, trainer, env, mean_cost_list, workdir, suffix, is_render):
    plt.figure(figsize=(30,30))
    plt.axis("on")
    G, pos, route_edges, total_cost, valid_route = rl_solution_to_graph(trainer, env)
    if is_render:
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
        plt.savefig(f'{workdir}/dqn_vrp_{args.problem}_{suffix}.png')
    plt.close()
    list_to_figure([mean_cost_list], ['mean_cost'], 'mean_cost', f'{workdir}/dqn_cost_{args.problem}_{suffix}.png')
    return total_cost

def create_trainer(env, args, workdir):
    dqn_config = dqn_config_default.copy()
    dqn_config.update({
        "batch_size": args.training_step,
        "min_replay_history": args.training_step*20,
        "training_steps": args.training_step,
        "lr": 0.0005,
        "target_update_period": args.training_step // 2,
        "update_period": args.training_step // 10,
        "replay_capacity": args.training_step*100
    })

    policy = DQNTorchPolicy(env.observation_space, env.action_space, env.config, dqn_config)
    dqn_trainer = Trainer(env, policy, dqn_config)
    return dqn_trainer


def train_dqn(dqn_trainer, env, args, workdir, suffix):
    action_space_size = env.action_space.n
    if not os.path.exists('train_log'):
        os.mkdir('train_log')
    writer = TensorBoard(f'train_log/{args.run_name}')

    max_mean_reward = - 1000
    mean_cost_list = []
    total_cost_list = []
    min_route_cost = 10000000

    for i in range(args.iters):
        print(suffix)
        result = dqn_trainer.train(i)
        now_mean_reward = print_result(action_space_size, result, writer, i)
        if now_mean_reward > max_mean_reward:
            dqn_trainer.policy.save_param(f"{args.problem}_{suffix}_best")
        if (i+1) % 5 == 0 or (now_mean_reward > max_mean_reward):
            _total_cost = draw_route(args, dqn_trainer, env, mean_cost_list, workdir, suffix, is_render=(now_mean_reward > max_mean_reward))
            total_cost_list.append(_total_cost)
        max_mean_reward = max(max_mean_reward, now_mean_reward)
        mean_cost_list.append(max_mean_reward)
    list_to_figure([total_cost_list], ['total_cost'], 'total_cost', f'{workdir}/dqn_total_cost_{args.problem}_{suffix}.png')
    return mean_cost_list


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
parser.add_argument("--problem", type=str, default="p01")
parser.add_argument("--run-name", type=str, default="dqn")
parser.add_argument("--pt", type=int, default=0)
parser.add_argument("--training-step", type=int, default=2048)
parser.add_argument("--constraint-id", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    vrp_config = env_config.copy()
    vrp_config.update({'problem': args.problem, "constraint_id": args.constraint_id})
    env = CVRPEnv(vrp_config)

    if args.pt:
        workdir = f"{os.environ['PT_OUTPUT_DIR']}/{args.problem}_{args.constraint_id}/"
    else:
        workdir = f"output/vrp/{args.problem}_{args.constraint_id}/"
    os.makedirs(workdir, exist_ok=True)
    

    metric_list = []
    metric_labels = []

    env.reset()
    env.is_constraint_imposed = False
    trainer_woc = create_trainer(env, args, workdir)
    total_cost_list = train_dqn(trainer_woc, env, args, workdir, 'woc')
    metric_list.append(total_cost_list)
    metric_labels.append('mean_reward_without_constraint')

    env.reset()
    env.is_constraint_imposed = True
    # trainer_wc = create_trainer(env, args, workdir)
    # trainer_wc.policy.load_param(f"{args.problem}_woc_best")
    trainer_wc = trainer_woc
    total_cost_list = train_dqn(trainer_wc, env, args, workdir, 'wc_incremental')
    metric_list.append(total_cost_list)
    metric_labels.append('mean_reward_with_constraint_incremental')

    env.reset()
    env.is_constraint_imposed = True
    trainer_wc_restart = create_trainer(env, args, workdir)
    total_cost_list = train_dqn(trainer_wc_restart, env, args, workdir, 'wc_restart')
    metric_list.append(total_cost_list)
    metric_labels.append('mean_reward_with_constraint_from_scratch')

    list_to_figure(metric_list, metric_labels, 'mean reward of policies', f'{workdir}/dqn_total_cost_{args.problem}.png', smoothed=False)