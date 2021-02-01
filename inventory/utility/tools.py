from env.inventory_utils import Utils
import numpy as np
import matplotlib.pyplot as plt
import ray
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import seaborn as sns
from tqdm import tqdm as tqdm
from render.inventory_renderer import AsciiWorldRenderer
from agents.inventory import ProductUnit, SKUSupplierUnit, SKUStoreUnit, SKUWarehouseUnit
sns.set_style("darkgrid")


class SimulationTracker:
    def __init__(self, episod_len, n_episods, env, policies):
        self.episod_len = episod_len
        self.global_balances = np.zeros((n_episods, episod_len))
        self.global_rewards = np.zeros((n_episods, episod_len))
        self.env = env
        self.policies = policies
        self.facility_names = list(env.agent_ids())
        self.step_balances = np.zeros((n_episods, self.episod_len, len(self.facility_names)))
        self.step_rewards = np.zeros((n_episods, self.episod_len, len(self.facility_names)))
        self.n_episods = n_episods
        self.sku_to_track = None
        self.stock_status = None
        self.stock_in_transit_status = None
        self.reward_status = None
        self.demand_status = None
        self.reward_discount_status = None
        self.order_to_distribute = None
    
    def add_sample(self, episod, t, global_balance, global_reward, step_balances, step_rewards):
        self.global_balances[episod, t] = global_balance
        self.global_rewards[episod, t] = global_reward
        for i, f in enumerate(self.facility_names):
            self.step_balances[episod, t, i] = step_balances[f]
            self.step_rewards[episod, t, i] = step_rewards[f]

    def add_sku_status(self, episod, t, stock, order_in_transit, demands, rewards, balances, reward_discounts, order_to_distribute ):
        if self.sku_to_track is None:
            self.sku_to_track = set(list(stock.keys()) + list(order_in_transit.keys()) + list(demands.keys()))
            self.stock_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.stock_in_transit_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.demand_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.reward_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.balance_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.reward_discount_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.order_to_distribute = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
        for i, sku_name in enumerate(self.sku_to_track):
            self.stock_status[episod, t, i] = stock[sku_name]
            self.stock_in_transit_status[episod, t, i] = order_in_transit[sku_name]
            self.demand_status[episod, t, i] = demands[sku_name]
            self.reward_status[episod, t, i] = rewards[sku_name]
            self.balance_status[episod, t, i] = balances[sku_name]
            self.reward_discount_status[episod, t, i] = reward_discounts[sku_name]
            self.order_to_distribute[episod, t, i] = order_to_distribute[sku_name]

    def render_sku(self, loc_path):
        for i, sku_name in enumerate(self.sku_to_track):
            fig, ax = plt.subplots(3, 1, figsize=(25, 10))
            x = np.linspace(0, self.episod_len, self.episod_len)
            stock = self.stock_status[0, :, i]
            order_in_transit = self.stock_in_transit_status[0, :, i]
            demand = self.demand_status[0, :, i] 
            reward = self.reward_status[0, :, i]
            balance = self.balance_status[0, :, i]
            reward_discount = self.reward_discount_status[0, :, i]
            order_to_distribute = self.order_to_distribute[0, :, i]
            ax[0].set_title('SKU Stock Status by Episod')
            for y_label, y in [('stock', stock), 
                               ('order_in_transit', order_in_transit), 
                               ('demand', demand),
                               ('order_to_distribute', order_to_distribute)]:
                ax[0].plot(x, y, label=y_label )
            
            ax[1].set_title('SKU Reward / Balance Status by Episod')
            ax[1].plot(x, balance, label='Balance' )
            ax_r = ax[1].twinx()
            ax_r.plot(x, reward, label='Reward', color='r')
            
            ax[2].plot(x, reward_discount, label='Reward Discount')
            fig.legend()
            fig.savefig(f"{loc_path}/{sku_name}.png")
            plt.close(fig=fig)

    def render(self, file_name, metrics, facility_types):
        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(0, self.episod_len, self.episod_len)
        
        _agent_list = []
        _step_idx = []
        for i, f in enumerate(self.facility_names):
            facility = self.env.world.facilities[Utils.agentid_to_fid(f)]
            if type(facility) in facility_types and Utils.is_consumer_agent(f):
                _agent_list.append(f)
                _step_idx.append(i)
        _step_metrics = [metrics[0, :, i] for i in _step_idx]

        # axs[0].set_title('Global balance')
        # axs[0].plot(x, self.global_balances.T)                                        

        axs[0].set_title('Cumulative Sum')
        axs[0].plot(x, np.cumsum(np.sum(_step_metrics, axis = 0)) ) 
        
        
        axs[1].set_title('Breakdown by Agent (One Episod)')
        axs[1].plot(x, np.cumsum(_step_metrics, axis = 1).T)              
        axs[1].legend(_agent_list, loc='upper left')
        
        fig.savefig(file_name)
        plt.close(fig=fig)
        # plt.show()

    def run_wth_render(self, facility_types=[SKUStoreUnit]):
        env = self.env
        policies = self.policies
        obss = env.reset()
        _, infos = env.state_calculator.world_to_state(env.world)
        rnn_states = {}
        rewards = {}
        for agent_id in obss.keys():
            rnn_states[agent_id] = policies[agent_id].get_initial_state()
            rewards[agent_id] = 0
        for epoch in range(self.episod_len):
            action_dict = {}
            for agent_id, obs in obss.items():
                policy = policies[agent_id]
                action, _, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id], explore=False ) 
                if self.env.is_reward_shape_on and hasattr(policy, 'policy_action_space'):
                    action_dict[agent_id] = policy.policy_action_space[action]
                else:
                    action_dict[agent_id] = action
            
            obss, rewards, _, infos = env.step(action_dict, is_downsampling=False)
            step_balances = {}
            step_rewards = {}
            for agent_id in rewards.keys():
                step_balances[agent_id] = env.world.facilities[Utils.agentid_to_fid(agent_id)].economy.step_balance.total()
                step_rewards[agent_id] = env.world.facilities[Utils.agentid_to_fid(agent_id)].step_reward

            self.add_sample(0, epoch, env.world.economy.global_balance().total(), env.world.economy.global_reward(), step_balances, step_rewards)
            # some stats
            stock_status = env.get_stock_status()
            order_in_transit_status = env.get_order_in_transit_status()
            demand_status = env.get_demand_status()
            reward_status = env.get_reward_status()
            balance_status = env.get_balance_status()
            reward_discount_status = env.get_reward_discount_status()
            order_to_distribute_status = env.get_order_to_distribute_status()

            self.add_sku_status(0, epoch, stock_status, 
                                order_in_transit_status, demand_status, 
                                reward_status, balance_status, 
                                reward_discount_status,  order_to_distribute_status)
            
        _step_idx = []
        for i, f in enumerate(self.facility_names):
            facility = self.env.world.facilities[Utils.agentid_to_fid(f)]
            if type(facility) in facility_types and Utils.is_consumer_agent(f):
                _step_idx.append(i)
        _step_metrics = [self.step_rewards[0, :, i] for i in _step_idx]
        _step_metrics_list = np.cumsum(np.sum(_step_metrics, axis = 0))
        return np.sum(_step_metrics), _step_metrics_list

    def run_and_render(self, loc_path, facility_types=[SKUStoreUnit]):
        metric, metric_list = self.run_wth_render(facility_types=facility_types)
        os.makedirs(loc_path, exist_ok=True)
        self.render('%s/plot_balance.png' % loc_path, self.step_balances, facility_types)
        self.render('%s/plot_reward.png' % loc_path, self.step_rewards, facility_types)
        self.render_sku(loc_path)
        return metric, metric_list
        # if steps_to_render is not None:
        #     print('Rendering the animation...')
        #     AsciiWorldRenderer.plot_sequence_images(frame_seq, f"output/{policy_mode}/sim.mp4")

    def run(self):
        env = self.env
        policies = self.policies
        obss = env.reset()
        _, infos = env.state_calculator.world_to_state(env.world)
        rnn_states = {}
        rewards = {}
        for agent_id in obss.keys():
            rnn_states[agent_id] = policies[agent_id].get_initial_state()
            rewards[agent_id] = 0
        # for epoch in tqdm(range(self.episod_len)):
        for epoch in range(self.episod_len):
            # print(f"{epoch}/{self.episod_len}")
            action_dict = {}
            for agent_id, obs in obss.items():
                policy = policies[agent_id]
                action, _, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id], explore=False ) 
                action_dict[agent_id] = action
            obss, rewards, _, infos = env.step(action_dict, is_downsampling=False)
            step_balances = {}
            step_rewards = {}
            for agent_id in rewards.keys():
                step_balances[agent_id] = env.world.facilities[Utils.agentid_to_fid(agent_id)].economy.step_balance.total()
                step_rewards[agent_id] = env.world.facilities[Utils.agentid_to_fid(agent_id)].step_reward
        

def print_hardware_status():
    
    import multiprocessing as mp
    print('Number of CPU cores:', mp.cpu_count())
    stream = os.popen('cat /proc/meminfo | grep Mem')
    print(f"Memory: {stream.read()}")
    
    stream = os.popen('lspci | grep -i nvidia ')
    print(f"GPU status: {stream.read()}")

    print(device_lib.list_local_devices())

    ray.shutdown()
    ray.init(num_gpus=1)
    print(f"ray.get_gpu_ids(): {ray.get_gpu_ids()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")


def list_to_figure(results, labels, caption, loc_path):
    num_result = len(results)
    if num_result <= 0:
        return
    num_points = np.max([len(results[i]) for i in range(num_result)])
    for i in range(num_result):
        if len(results[i]) < num_points:
            results[i].extend([results[i][-1]]*(num_points-len(results[i])))
    plt.figure(figsize=(10,7))
    x = np.linspace(0, num_points, num_points)                                         
    plt.title(caption)
    for y_label, y in zip(labels, results):
        plt.plot(x, y, label=y_label )
    plt.legend(loc='best')
    plt.savefig(loc_path)
    plt.close()