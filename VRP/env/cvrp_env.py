import gym
from gym.spaces import Box
import numpy as np
from utils.utils import VRProblem
import random
import math 
from constraints.constraint_atoms import CVPRConstraints, cvrp_constraints



def compute_angle(depot, p):
    v1 = [depot[0],depot[1],depot[0]+1,depot[1]]
    v2 = [depot[0],depot[1],p[0],p[1]]
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
    return included_angle

class CVRPEnv(gym.Env):
    
    def __init__(self, config):
        self.vrp_problem = VRProblem(config['problem'])
        self.config = config
        self.num_nodes = self.vrp_problem.num_nodes
        self.num_trucks = self.vrp_problem.num_trucks
        self.pox_x = self.vrp_problem.node_pos_x
        self.pox_y = self.vrp_problem.node_pos_y
        self.vehicle_capacity = self.vrp_problem.vehicle_capacity
        self.action_space = gym.spaces.Discrete(self.num_trucks)
        self.cost_matrix = self.vrp_problem.get_nodes_coordinates()
        self.demand = [d for d in self.vrp_problem.node_demand]
        self.max_demand = np.max(self.demand)
        self.min_demand = np.min(self.demand)
        self.max_cost_per_demand = self.min_demand / np.max(self.cost_matrix)
        self.total_cost = np.sum(self.cost_matrix)
        self.max_cost = np.max(self.cost_matrix)

        self.is_constraint_imposed = False
        self.constraint_automaton = CVPRConstraints(cvrp_constraints[0])

        self.vehicle_remaining_capacity = np.ones(self.num_trucks)
        self.node_to_vehicle = np.zeros((self.num_trucks, self.num_nodes))
        self.cur_node_encoded = np.zeros(self.num_nodes)
        self.node_remaining_demand = np.zeros(self.num_nodes)
        self.vehicle_position = np.zeros((self.num_trucks, self.num_nodes))
        self.constraint_status = np.zeros(self.constraint_automaton.num_constraint_states)
        
        self.depot = self.vrp_problem.depots[0] - 1
        self.cur_node = self.depot
        self.cur_node_index = 0
        self.step_reward = 0
        self.timestep = 0
        self.episode_len = self.num_nodes * 100
        self.cur_vehicle_id = 0
        self.node_in_sequence = []
        
        angle_list = []
        depot_pos = (self.pox_x[self.depot], self.pox_y[self.depot])
        for i in range(self.num_nodes):
            if i == self.depot:
                continue
            p = (self.pox_x[i], self.pox_y[i])
            angle = compute_angle(depot_pos, p)
            angle_list.append((angle, i))
        angle_list = sorted(angle_list, key=lambda x: x[0])
        self.node_in_sequence = [a[1] for a in angle_list]
        
        self.state = None
        self.reset()
        self.state_dim = self.state.shape[0]
        self.observation_space = Box(low=0.0, high=self.num_trucks, shape=(self.state_dim, ), dtype=np.float64)

    def get_ortool_value(self):
        return self.vrp_problem.get_ortool_opt_val()

    def reset(self):
        self.cur_node_index = 0
        # random.shuffle(self.node_in_sequence)
        self.demand = [d for d in self.vrp_problem.node_demand]
        self.cur_node = self.node_in_sequence[self.cur_node_index]
        self.vehicle_remaining_capacity = np.ones(self.num_trucks)
        self.node_to_vehicle = np.zeros((self.num_trucks, self.num_nodes))
        self.cur_node_encoded = np.zeros(self.num_nodes)

        # self.node_to_vehicle[:, self.cur_node] = 1.0
        self.cur_node_encoded[self.cur_node] = 1.0
        # self.node_remaining_demand = np.array([d/self.vehicle_capacity for d in self.demand])
        self.node_remaining_demand = np.ones(self.num_nodes)
        self.node_remaining_demand[self.depot] = 0.0
        self.vehicle_position_encoded = np.zeros(self.num_nodes)
        self.vehicle_position_encoded[self.depot] = self.num_trucks
        self.vehicle_position = np.zeros(self.num_trucks)
        self.vehicle_position[:] = self.depot

        self.state = self.state_calculator()
        
        self.step_reward = 0
        self.timestep = 0

        if self.is_constraint_imposed:
            self.constraint_automaton.reset()
            self.constraint_status[self.constraint_automaton.automata_state] = 1
        return np.array(self.state)

    def path_cost(self, node_to_vehicle):
        node_on_path = []
        for i in range(self.num_nodes):
            if node_to_vehicle[i] == 1.0 and i != self.depot:
                node_on_path.append(i)
        visited_node = [self.depot]
        cur_node = self.depot
        total_cost = 0.0
        for _ in range(len(node_on_path)):
            cost = []
            for node in node_on_path:
                if node not in visited_node:
                    cost.append((node, self.cost_matrix[cur_node][node]))
            cost = sorted(cost, key=(lambda x: x[1]))
            cur_node = cost[0][0]
            total_cost += cost[0][1]
            visited_node.append(cur_node)
        return total_cost, visited_node

    def validity_check(self):
        for i in range(self.num_trucks):
            if np.dot(self.node_to_vehicle[i, :], self.demand) > self.vehicle_capacity:
                return False
        return True

    def state_calculator(self):
        return np.concatenate((self.vehicle_remaining_capacity,
                               self.vehicle_position_encoded,
                               self.node_remaining_demand,
                               self.constraint_status))
        # return np.concatenate((self.vehicle_remaining_capacity, 
        #                        self.node_to_vehicle.flatten(),
        #                        self.vehicle_position.flatten(),
        #                        self.cur_node_encoded, 
        #                        self.node_remaining_demand))


    def step(self, action):
        self.step_reward = 0.0
        dones = False
        self.node_to_vehicle[action, self.cur_node] = 1.0
        cur_vehicle_pos = int(self.vehicle_position[action])
        
        if self.vehicle_remaining_capacity[action] + 1.0/self.vehicle_capacity > self.demand[self.cur_node] / self.vehicle_capacity:
            self.vehicle_remaining_capacity[action] = max(0.0, self.vehicle_remaining_capacity[action]-self.demand[self.cur_node] / self.vehicle_capacity)
            self.step_reward = - self.cost_matrix[cur_vehicle_pos][self.cur_node] / self.max_cost
        else:
            self.vehicle_remaining_capacity[action] = 0.0
            self.step_reward = -1.0 - self.cost_matrix[cur_vehicle_pos][self.cur_node] / self.max_cost
        
        self.vehicle_position_encoded[self.cur_node] -= 1.0
        self.node_remaining_demand[self.cur_node] = 0.0
        self.cur_node_encoded[self.cur_node] = 0.0
        self.cur_node_index += 1
        dones = (dones | (self.cur_node_index >= len(self.node_in_sequence)))
        if not dones:
            self.cur_node = self.node_in_sequence[self.cur_node_index]
        else:
            self.cur_node = self.depot

        self.vehicle_position_encoded[self.cur_node] += 1.0
        self.cur_node_encoded[self.cur_node] = 1.0
        self.vehicle_position[action] = self.cur_node

        if self.is_constraint_imposed:
            self.constraint_automaton.step(self.node_to_vehicle)
            if not self.constraint_automaton.is_accepted:
                self.step_reward -= 1.0
            self.constraint_status[:] = 0
            self.constraint_status[self.constraint_automaton.automata_state] = 1        

        self.state = self.state_calculator()
        return np.array(self.state), self.step_reward, dones, {}


class CVRPSiteEnv(gym.Env):
    
    def __init__(self, config):
        self.vrp_problem = VRProblem(config['problem'])
        self.config = config
        self.num_nodes = self.vrp_problem.num_nodes
        self.num_trucks = self.vrp_problem.num_trucks
        self.pox_x = self.vrp_problem.node_pos_x
        self.pox_y = self.vrp_problem.node_pos_y
        self.vehicle_capacity = self.vrp_problem.vehicle_capacity
        self.action_space = gym.spaces.Discrete(self.num_nodes)
        self.cost_matrix = self.vrp_problem.get_nodes_coordinates()
        self.demand = [d for d in self.vrp_problem.node_demand]
        self.max_demand = np.max(self.demand)
        self.min_demand = np.min(self.demand)
        self.max_cost_per_demand = self.min_demand / np.max(self.cost_matrix)
        self.total_cost = np.sum(self.cost_matrix)
        self.max_cost = np.max(self.cost_matrix)
        self.min_cost = np.min(self.cost_matrix)

        self.is_constraint_imposed = False
        self.constraint_automaton = CVPRConstraints(cvrp_constraints[config['constraint_id']])
        
        self.depot = self.vrp_problem.depots[0] - 1
        self.cur_node = self.depot
        self.cur_node_index = 0
        self.step_reward = 0
        self.timestep = 0
        self.episode_len = self.num_nodes * 20
        self.cur_vehicle_id = 1
        self.node_in_sequence = []

        self.vehicle_total_remaining_capacity = np.array([self.num_trucks-1.0])
        self.cur_vehicle_remaining_capacity = np.array([1.0])
        self.node_to_vehicle = np.zeros((self.num_trucks, self.num_nodes))
        self.cur_node_encoded = np.zeros(self.num_nodes)
        self.node_remaining_demand = np.zeros(self.num_nodes)
        self.cur_vehicle_position = np.zeros(self.num_nodes)
        self.cur_vehicle_position[self.depot] = 1.0
        self.constraint_status = np.zeros(self.constraint_automaton.num_constraint_states)

        angle_list = []
        depot_pos = (self.pox_x[self.depot], self.pox_y[self.depot])
        for i in range(self.num_nodes):
            if i == self.depot:
                continue
            p = (self.pox_x[i], self.pox_y[i])
            angle = compute_angle(depot_pos, p)
            angle_list.append((angle, i))
        angle_list = sorted(angle_list, key=lambda x: x[0])
        self.node_in_sequence = [self.depot] + [a[1] for a in angle_list]
        # self.node_in_sequence = [i for i in range(self.num_nodes)]
        
        self.state = None
        self.action_mask = None
        self.reset()
        self.state_dim = self.state.shape[0]
        self.observation_space = Box(low=0.0, high=self.num_trucks+1, shape=(self.state_dim, ), dtype=np.float64)

    
    def state_calculator(self):
        self.state = np.concatenate((self.cur_vehicle_remaining_capacity,
                                     self.cur_vehicle_position,
                                     self.constraint_status,
                                     self.node_remaining_demand))
        return self.state

    def action_mask_calculator(self):
        action_mask = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            next_node = self.node_in_sequence[i]
            if next_node == self.cur_node:
                continue
            elif self.cur_vehicle_remaining_capacity[0] + 0.01/self.vehicle_capacity < self.demand[next_node] / self.vehicle_capacity:
                continue
            elif (next_node != self.depot) & (self.node_remaining_demand[next_node] == 0):
                continue
            else:
                action_mask[i] = 1
        self.action_mask = action_mask
        return self.action_mask

        
    def reset(self):
        self.cur_vehicle_id = 1
        self.demand = [d for d in self.vrp_problem.node_demand]
        self.cur_node = self.node_in_sequence[self.depot]
        self.node_to_vehicle = np.zeros((self.num_trucks, self.num_nodes))
        self.vehicle_total_remaining_capacity = np.array([self.num_trucks-1.0])
        self.cur_vehicle_remaining_capacity = np.array([1.0])
        self.node_to_vehicle = np.zeros((self.num_trucks, self.num_nodes))
        self.node_remaining_demand = np.ones(self.num_nodes)
        self.cur_vehicle_position = np.zeros(self.num_nodes)
        self.cur_vehicle_position[self.depot] = 1.0

        self.state = self.state_calculator()
        self.action_mask = self.action_mask_calculator()
        
        self.step_reward = 0
        self.timestep = 0

        if self.is_constraint_imposed:
            self.constraint_automaton.reset()
            self.constraint_status[:] = 0
            self.constraint_status[self.constraint_automaton.automata_state] = 1
        return self.state, {'action_mask': self.action_mask}

    def step(self, action):
        self.step_reward = 0.0
        dones = False
        next_node = self.node_in_sequence[action]
        cur_remaining_capacity = self.cur_vehicle_remaining_capacity[0]
        self.timestep += 1

        self.cur_vehicle_remaining_capacity[0] = max(0.0, cur_remaining_capacity-self.demand[next_node] / self.vehicle_capacity)
        self.step_reward += (1.0 - self.cost_matrix[self.cur_node][next_node] / self.max_cost)
        
        self.node_to_vehicle[:, next_node] = 0.0
        visit_order = np.sum([1 if x > 0 else 0 for x in self.node_to_vehicle[self.cur_vehicle_id-1, :]])
        self.node_to_vehicle[self.cur_vehicle_id-1, next_node] = visit_order + 1.0
        
        self.node_remaining_demand[next_node] = 0.0

        if next_node == self.depot:
            if self.vehicle_total_remaining_capacity[0] > 0.0: 
                self.cur_vehicle_remaining_capacity[0] = 1.0
                self.vehicle_total_remaining_capacity[0] -= 1.0
            else:
                self.cur_vehicle_remaining_capacity[0] = 0.0
            self.cur_vehicle_id += 1
        
        
        dones = ((self.cur_vehicle_id > self.num_trucks) | (self.timestep >= self.episode_len))

        if self.is_constraint_imposed and self.constraint_automaton.is_accepted:
            self.constraint_automaton.step({'state': self.state,
                                            'demand': self.node_remaining_demand,
                                            'max_cost': self.max_cost,
                                            'cost': self.cost_matrix[self.cur_node][next_node]})
            if not self.constraint_automaton.is_accepted:
                self.step_reward = -(self.max_cost-self.min_cost) * 100 / self.max_cost
                dones = True
            self.constraint_status[:] = 0
            self.constraint_status[self.constraint_automaton.automata_state] = 1        
        
        self.cur_node = next_node
        self.cur_vehicle_position[:] = 0.0
        self.cur_vehicle_position[self.cur_node] = 1.0

        self.state = self.state_calculator()
        self.action_mask = self.action_mask_calculator()
        return self.state, self.step_reward, dones, {'action_mask': self.action_mask}

    def get_ortool_value(self):
        return self.vrp_problem.get_ortool_opt_val()

    def path_cost(self, node_to_vehicle):
        node_on_path = []
        for i in range(self.num_nodes):
            if node_to_vehicle[i] > 0.0 and i != self.depot:
                node_on_path.append((node_to_vehicle[i], i))
        node_on_path = sorted(node_on_path, key=lambda x: x[0])
        node_on_path = [x[1] for x in node_on_path]
        visited_node = [self.depot] + node_on_path
        pre_node = self.depot
        total_cost = 0.0
        for cur_node in node_on_path:
            total_cost += self.cost_matrix[pre_node][cur_node]
            pre_node = cur_node
        return total_cost, visited_node

    def validity_check(self):
        for i in range(self.num_trucks):
            if np.dot(self.node_to_vehicle[i, :], self.demand) > self.vehicle_capacity:
                return False
        return True
