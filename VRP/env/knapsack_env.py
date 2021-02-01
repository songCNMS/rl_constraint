import gym
from gym.spaces import Box
import numpy as np
import random
from utils.utils import KnapsackProblem


class KnapsackEnv(gym.Env):

    def __init__(self, config):
        self.knapsack_problem = KnapsackProblem(config['problem'])
        self.config = config
        self.values = self.knapsack_problem.values
        self.weights = self.knapsack_problem.weights[0]
        self.capacity = self.knapsack_problem.capacity[0]
        self.action_space = gym.spaces.Discrete(2)
        self.num_items = len(self.values)
        self.max_value = np.max(self.values)
        self.min_value = np.min(self.values)

        self.remaining_capacity = np.array([1.0])
        self.items_to_place = np.ones(self.num_items)
        self.cur_item_encoded = np.ones(self.num_items)

        self.state_dim = self.num_items + 1
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.state_dim, ), dtype=np.float64)
        self.state = np.zeros(self.state_dim)
        self.cur_item = 0
        self.cur_item_index = 0
        self.timestep = 0
        self.episode_len = self.num_items
        value_per_weight = [(i, v/w) for i, (w,v) in enumerate(zip(self.weights, self.values))]
        value_per_weight = sorted(value_per_weight, key=(lambda x: x[1]), reverse=True)
        self.items_in_sequence = [x[0] for x in value_per_weight]
    
    def get_ortool_value(self):
        return self.knapsack_problem.get_opt_value()

    def state_calculator(self):
        return np.concatenate((self.items_to_place, self.remaining_capacity))

    def reset(self):
        # random.shuffle(self.items_in_sequence)
        self.timestep = 0
        self.cur_item_index = 0
        self.remaining_capacity = np.array([1.0])
        self.items_to_place = np.ones(self.num_items)
        self.state = self.state_calculator()
        return self.state

    def step(self, action):
        self.timestep += 1
        cur_item = self.items_in_sequence[self.cur_item_index]
        self.items_to_place[cur_item] = 0.0
        if action == 0:
            self.step_reward = 0.0
        else:
            if self.remaining_capacity[0] + 1.0 / self.capacity > self.weights[cur_item] / self.capacity:
                self.remaining_capacity[0] = max(0.0, self.remaining_capacity[0] - self.weights[cur_item] / self.capacity)
                self.step_reward = self.values[cur_item] / self.max_value
            else:
                self.remaining_capacity[0] = 0.0
                self.step_reward = -1.0
        self.cur_item_index += 1
        dones = ((self.cur_item_index >= self.num_items))
        self.state = self.state_calculator()
        return self.state, self.step_reward, dones, {}