from abc import ABC
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
import pickle

from numpy.lib.arraysetops import isin
import networkx as nx
from enum import Enum, auto
from agents.base import Cell, Agent, BalanceSheet
from agents.inventory import *
from agents.inventory_order import *
from env.inventory_utils import Utils
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box, Tuple, MultiDiscrete, Discrete
from env.inventory_action_calculator import ActionCalculator
from env.inventory_reward_calculator import RewardCalculator
from env.inventory_state_calculator import StateCalculator
from agents.inventory import World
from env.gamma_sale_retailer_world import load_sale_sampler as gamma_sale_sampler
# from env.gamma_sale_retailer_world import load_noisy_sale_sampler as gamma_sale_sampler
from env.online_retailer_world import load_sale_sampler as online_sale_sampler


initial_balance = 1000000

class WorldBuilder():
    
    @staticmethod
    def create(supply_network_config, x = 80, y = 32):
        world = World(x, y, supply_network_config)
        world.grid = [[TerrainCell(xi, yi) for yi in range(y)] for xi in range(x)]

        def default_economy_config(order_cost=0, initial_balance = initial_balance):
            return ProductUnit.EconomyConfig(order_cost, initial_balance)
        
        # facility placement
        map_margin = 4
        size_y_margins = world.size_y - 2*map_margin

        supplier_x = 10
        retailer_x = 70
        
        n_supplies = supply_network_config.get_supplier_num()
        suppliers = []
        supplier_skus = []
        supplier_sources = dict()
        for i in range(n_supplies):
            supplier_config = SupplierCell.Config(max_storage_capacity=supply_network_config.get_supplier_capacity(i),
                                                  unit_storage_cost=supply_network_config.get_supplier_unit_storage_cost(i),
                                                  fleet_size=supply_network_config.get_supplier_fleet_size(i),
                                                  unit_transport_cost=supply_network_config.get_supplier_unit_transport_cost(i))
            if n_supplies > 1:
                supplier_y = int(size_y_margins/(n_supplies - 1)*i + map_margin)
            else:
                supplier_y = int(size_y_margins/2 + map_margin)
            f = SupplierCell(supplier_x, supplier_y, 
                             world, supplier_config, 
                             default_economy_config() )
            f.idx_in_config = i
            f.facility_info = supply_network_config.get_supplier_info(i)
            f.facility_short_name = supply_network_config.get_supplier_short_name()
            world.agent_echelon[f.id] = 0
            world.place_cell(f) 
            suppliers.append(f)
            sku_info_list = supply_network_config.get_sku_of_supplier(i)
            for _, sku_info in enumerate(sku_info_list):
                bom = BillOfMaterials({}, sku_info['sku_name'])
                supplier_sku_config = ProductUnit.Config(sources=None, 
                                                         unit_manufacturing_cost=sku_info['cost'], 
                                                         sale_gamma=sku_info.get('sale_gamma', 10), 
                                                         bill_of_materials=bom)
                sku = SKUSupplierUnit(f, supplier_sku_config, 
                                      default_economy_config(order_cost=f.facility_info['order_cost']) )
                sku.idx_in_config = sku_info['sku_name']
                f.sku_in_stock.append(sku)
                sku.distribution = f.distribution
                sku.storage = f.storage
                sku.sku_info = sku_info
                f.storage.try_add_units({sku_info['sku_name']: sku_info['init_stock']})
                supplier_skus.append(sku)
                if sku_info['sku_name'] not in supplier_sources:
                    supplier_sources[sku_info['sku_name']] = []
                supplier_sources[sku_info['sku_name']].append(sku)
                world.agent_echelon[sku.id] = 0

        # distribution  
        n_echelon = supply_network_config.get_num_warehouse_echelon()
        
        pre_warehouses = suppliers
        all_warehouses = []
        warehouse_skus = []
        pre_warehouse_sources = supplier_sources
        for echelon in range(n_echelon):
            echelon_gap = (retailer_x-supplier_x)/(n_echelon+1)
            echelon_x = int(supplier_x+(echelon+1)*echelon_gap)
            n_warehouses = supply_network_config.get_warehouse_num(echelon)
            warehouses = []
            warehouse_sources = dict()
            for i in range(n_warehouses):
                warehouse_config = WarehouseCell.Config(max_storage_capacity=supply_network_config.get_warehouse_capacity(echelon, i), 
                                                        unit_storage_cost=supply_network_config.get_warehouse_unit_storage_cost(echelon, i),
                                                        fleet_size=supply_network_config.get_warehouse_fleet_size(echelon, i),
                                                        unit_transport_cost=supply_network_config.get_warehouse_unit_transport_cost(echelon, i))
                if n_warehouses > 1:
                    warehouse_y = int(size_y_margins/(n_warehouses - 1)*i + map_margin)
                else:
                    warehouse_y = int(size_y_margins/2 + map_margin)
                w =  WarehouseCell(echelon_x, warehouse_y, 
                                world, warehouse_config, 
                                default_economy_config() )
                w.idx_in_config = i
                w.echelon_level = echelon
                w.facility_info = supply_network_config.get_warehouse_info(echelon, i)
                w.facility_short_name = supply_network_config.get_warehouse_short_name(echelon)
                world.agent_echelon[w.id] = 1+echelon
                world.place_cell(w) 
                warehouses.append(w)
                WorldBuilder.connect_cells(world, w, *pre_warehouses)
                sku_info_list = supply_network_config.get_sku_of_warehouse(echelon, i)
                for _, sku_info in enumerate(sku_info_list):
                    candidate_upstream_suppliers = pre_warehouse_sources[sku_info['sku_name']]
                    upstream_suppliers = []
                    for s in candidate_upstream_suppliers:
                        if i in s.facility.facility_info['downstream_facilities']:
                            upstream_suppliers.append(s)
                    bom = BillOfMaterials({sku_info['sku_name']: 1}, sku_info['sku_name'])
                    warehouse_sku_config = ProductUnit.Config(sources=upstream_suppliers, 
                                                              unit_manufacturing_cost=sku_info.get('cost', 10), 
                                                              sale_gamma=sku_info.get('sale_gamma', 10), 
                                                              bill_of_materials=bom)
                    sku = SKUWarehouseUnit(w, warehouse_sku_config,
                                           default_economy_config(order_cost= w.facility_info['order_cost']) )
                    sku.idx_in_config = sku_info['sku_name']
                    w.sku_in_stock.append(sku)
                    sku.distribution = w.distribution
                    sku.storage = w.storage
                    sku.sku_info = sku_info
                    warehouse_skus.append(sku)
                    w.storage.try_add_units({sku_info['sku_name']: sku_info.get('init_stock', 0)})
                    if sku_info['sku_name'] not in warehouse_sources:
                        warehouse_sources[sku_info['sku_name']] = []
                    warehouse_sources[sku_info['sku_name']].append(sku)
                    world.agent_echelon[sku.id] = 1+echelon
                    # update downstreaming sku list in supplier_list
                    for s_sku in upstream_suppliers:
                        s_sku.downstream_skus.append(sku)

            all_warehouses.extend(warehouses)
            pre_warehouse_sources = warehouse_sources
            pre_warehouses = warehouses

        # final consumers
        n_stores = supply_network_config.get_store_num()
        stores = []
        store_skus = []
        for i in range(n_stores):
            store_config = RetailerCell.Config(max_storage_capacity=supply_network_config.get_store_capacity(i), 
                                               unit_storage_cost=supply_network_config.get_store_unit_storage_cost(i),
                                               fleet_size=1000,
                                               unit_transport_cost=10)
            if n_stores > 1:
                retailer_y = int(size_y_margins/(n_stores - 1)*i + map_margin)
            else:
                retailer_y = int(size_y_margins/2 + map_margin)
            r = RetailerCell(retailer_x, retailer_y, 
                             world, store_config, 
                             default_economy_config() )
            r.idx_in_config = i
            r.facility_info = supply_network_config.get_store_info(i)
            r.facility_short_name = supply_network_config.get_store_short_name()
            world.agent_echelon[r.id] = 1+n_echelon
            world.place_cell(r)
            stores.append(r)
            WorldBuilder.connect_cells(world, r, *pre_warehouses)
            sku_info_list = supply_network_config.get_sku_of_store(i)
            for _, sku_info in enumerate(sku_info_list):
                candidate_upstream_warehouses = pre_warehouse_sources[sku_info['sku_name']]
                upstream_warehouses = []
                for s in candidate_upstream_warehouses:
                    if i in s.facility.facility_info['downstream_facilities']:
                        upstream_warehouses.append(s)
                bom = BillOfMaterials({sku_info['sku_name']: 1}, sku_info['sku_name'])
                retail_sku_config = ProductUnit.Config(sources=upstream_warehouses, 
                                                       unit_manufacturing_cost=sku_info.get('cost', 10), 
                                                       sale_gamma=sku_info.get('sale_gamma', 10), 
                                                       bill_of_materials=bom)
                                
                if supply_network_config.get_demand_sampler() == "DYNAMIC_GAMMA":
                    sku = SKUStoreUnit(r, retail_sku_config, default_economy_config(order_cost=r.facility_info['order_cost']) )
                elif supply_network_config.get_demand_sampler() == "GAMMA":
                    sale_sampler = gamma_sale_sampler(i, supply_network_config)
                    sku = OuterSKUStoreUnit(r, retail_sku_config, default_economy_config(order_cost=r.facility_info['order_cost']), sale_sampler )
                else:
                    sale_sampler = online_sale_sampler(f"data/OnlineRetail/store{i+1}.csv")
                    sku = OuterSKUStoreUnit(r, retail_sku_config, default_economy_config(order_cost=r.facility_info['order_cost']), sale_sampler )
                sku.idx_in_config = sku_info['sku_name']
                r.sku_in_stock.append(sku)
                sku.storage = r.storage
                sku.sku_info = sku_info
                r.storage.try_add_units({sku_info['sku_name']:  sku_info.get('init_stock', 0)})
                store_skus.append(sku)
                world.agent_echelon[sku.id] = 1+n_echelon

                # update downstreaming sku list in warehouse_list
                for w_sku in upstream_warehouses:
                    w_sku.downstream_skus.append(sku)
    
        for facility in suppliers + all_warehouses + stores:
            world.facilities[facility.id] = facility
        for sku in supplier_skus + warehouse_skus + store_skus:
            sku.construct_constraint_automaton()
            world.facilities[sku.id] = sku
            if sku.sku_info.get('price', 0) > world.max_price:
                world.max_price = sku.sku_info.get('price', 0)
        world.total_echelon = supply_network_config.get_total_echelon()
        return world
        
    @staticmethod
    def connect_cells(world, source, *destinations):
        for dest_cell in destinations:
            WorldBuilder.build_railroad(world, source.x, source.y, dest_cell.x, dest_cell.y)

    @staticmethod    
    def build_railroad(world, x1, y1, x2, y2):
        step_x = np.sign(x2 - x1)
        step_y = np.sign(y2 - y1)

        # make several attempts to find a route non-adjacent to existing roads  
        for i in range(5):
            xi = min(x1, x2) + int(abs(x2 - x1) * rnd.uniform(0.15, 0.85))
            if not (world.is_railroad(xi-1, y1+step_y) or world.is_railroad(xi+1, y1+step_y)):
                break

        for x in range(x1 + step_x, xi, step_x):
            world.create_cell(x, y1, RailroadCell) 
        if step_y != 0:
            for y in range(y1, y2, step_y):
                world.create_cell(xi, y, RailroadCell) 
        for x in range(xi, x2, step_x):
            world.create_cell(x, y2, RailroadCell) 
    
    
class InventoryManageEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env_config = env_config
        self.is_reward_shape_on = env_config['reward_shape']
        self.supply_network_config = env_config['supply_network_config']
        self.world = WorldBuilder.create(self.supply_network_config, 80, 16)

        # is constraint imposed
        self.constraint_imposed = False
        self.training = True
        self.policies = None

        self.current_iteration = 0
        self.n_iterations = 0
        self.pickled_world = None
        self.product_ids = self._product_ids()
        # maximal number of sources of an agent
        self.max_sources_per_facility = 0
        # maximal number of fleets of a facility
        self.max_fleet_size = 0
        self.facility_types = {}
        facility_class_id = 0
        # indicate training mode
        self.discount_reward_training = True
        for f in self.world.facilities.values():
            if isinstance(f, FacilityCell):
                sources_num = 0
                for sku in f.sku_in_stock:
                    if sku.consumer is not None and sku.consumer.sources is not None:
                        sources_num = len(sku.consumer.sources)
                        if sources_num > self.max_sources_per_facility:
                            self.max_sources_per_facility = sources_num
                    
                if f.distribution is not None:      
                    if len(f.distribution.fleet) > self.max_fleet_size:
                        self.max_fleet_size = len(f.distribution.fleet)
                    
            facility_class = f.__class__.__name__
            if facility_class not in self.facility_types:
                self.facility_types[facility_class] = facility_class_id
                facility_class_id += 1
                
        self.state_calculator = StateCalculator(self)
        self.reward_calculator = RewardCalculator(env_config)
        self.action_calculator = ActionCalculator(self)
                         
        self.action_space_producer = MultiDiscrete([ 
            1,                             # unit price
            1,                             # production rate level
        ])
        
        
        # self.action_space_consumer = MultiDiscrete([ 
        #     self.max_sources_per_facility,               # consumer source id
        #     len(supply_network_config.get_consumer_action_space())         # consumer_quantity
        # ])
        if self.is_reward_shape_on:
            self.action_space_consumer = MultiDiscrete([
                len(self.supply_network_config.get_reward_discount_action_space()),
                len(self.supply_network_config.get_consumer_action_space())])
        else:
             self.action_space_consumer = Discrete(len(self.supply_network_config.get_consumer_action_space()))
                    
        example_state, _ = self.state_calculator.world_to_state(self.world)
        state_dim = len(list(example_state.values())[0])
        
        # 计算状态空间的大小，每个facility对应一个完整的状态
        self.observation_space = Box(low=0.00, high=100.00, shape=(state_dim, ), dtype=np.float64)

    def reset(self):
        self.world = WorldBuilder.create(self.supply_network_config, 80, 16)
        # self.world.reset()
        # randomly reset env to a state stored before
        # if np.random.random() <= 0.2:
            # self.pop()
        self.timestep = 0
        state, infos = self.state_calculator.world_to_state(self.world)
        return state
        # return self.head_running(state, infos)

    def stash(self):
        if self.pickled_world is None:
            self.pickled_world = pickle.dumps(self.world)
        # print('world saved')


    def pop(self):
        if self.pickled_world is not None:
            self.world = pickle.loads(self.pickled_world)
            self.pickled_world = None
            # print('world restored')

    def head_running(self, obss, infos):
        policies = self.policies
        rnn_states = {}
        for agent_id in obss.keys():
            rnn_states[agent_id] = policies[agent_id].get_initial_state()
        for _ in range(self.env_config['heading_timesteps']):
            action_dict = {}
            for agent_id, obs in obss.items():
                policy = policies[agent_id]
                action, _, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id], explore=True ) 
                if self.is_reward_shape_on and hasattr(policy, 'policy_action_space'):
                    action_dict[agent_id] = policy.policy_action_space[action]
                else:
                    action_dict[agent_id] = action
            obss, _, _, infos = self.step(action_dict, is_downsampling=False)
        return obss


    def tail_running(self, obss, infos):
        rewards = {agent_id: 0 for agent_id in obss.keys()}
        policies = self.policies
        gamma = self.env_config['gamma']
        rnn_states = {}
        for agent_id in obss.keys():
            rnn_states[agent_id] = policies[agent_id].get_initial_state()
        discount = 1.
        for _ in range(self.env_config['tail_timesteps']):
            action_dict = {}
            for agent_id, obs in obss.items():
                policy = policies[agent_id]
                action, _, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id],
                                                                    explore=True)
                if self.is_reward_shape_on and hasattr(policy, 'policy_action_space'):
                    action_dict[agent_id] = policy.policy_action_space[action]
                else:
                    action_dict[agent_id] = action
            control = self.action_calculator.action_dictionary_to_control(action_dict, self.world)
            _, rewards_outcome = self.world.act(control)

            cur_rewards = self.reward_calculator.calculate_reward(self, rewards_outcome)
            obss, infos = self.state_calculator.world_to_state(self.world)
            discount *= gamma
            for agent_id in obss.keys():
                rewards[agent_id] += discount*cur_rewards[agent_id]
        return rewards


    def step(self, action_dict, is_downsampling=False):
        control = self.action_calculator.action_dictionary_to_control(action_dict, self.world)
        balances_outcome, rewards_outcome  = self.world.act(control)
        self.timestep += 1
        
        for agent_id in balances_outcome.facility_step_balance_sheets.keys():
            balances_outcome.facility_step_balance_sheets[agent_id] = balances_outcome.facility_step_balance_sheets[agent_id].total()

        seralized_states, info_states = self.state_calculator.world_to_state(self.world)
        
        # update rewards and done status according to constraint automaton
        dones = {}
        timeout = (self.world.time_step >= self.env_config['episod_duration'])
        
        for agent_id in rewards_outcome.facility_step_balance_sheets.keys(): 
            facility = self.world.facilities[agent_id]
            if isinstance(facility, ProductUnit):
                info_state = info_states[Utils.agentid_consumer(agent_id)]
                facility.update_atom_status(info_state)
                if self.constraint_imposed:
                    facility.constraint_automaton_step(info_state)
                    if not facility.is_accepted:
                        # facility.step_reward += self.env_config['constraint_violate_reward']
                        facility.step_reward = -0.3
                        rewards_outcome.facility_step_balance_sheets[agent_id] = facility.step_reward
                        # balance_sheet_penalty = BalanceSheet(0, self.env_config['constraint_violate_reward'])
                        # facility.economy.deposit([balance_sheet_penalty])
        
        # update states
        seralized_states, info_states = self.state_calculator.world_to_state(self.world)
        rewards = self.reward_calculator.calculate_reward(self, rewards_outcome)
        if timeout and self.training:
            # timeout_rewards = self.tail_running(seralized_states, info_states)
            for agent_id in rewards.keys():
                facility = self.world.facilities[Utils.agentid_to_fid(agent_id)]
                if Utils.is_consumer_agent(agent_id) and isinstance(facility, SKUStoreUnit):
                    rewards[agent_id] += facility.get_ending_reward()
                    # rewards[agent_id] += timeout_rewards[agent_id]  
        dones = {agent_id: timeout for agent_id in rewards.keys()}
        dones['__all__'] = np.all(list(dones.values()))
        return seralized_states, rewards, dones, info_states
    
    def agent_ids(self):
        agents = []
        for f_id in self.world.facilities.keys():
            agents.append(Utils.agentid_producer(f_id))
        for f_id in self.world.facilities.keys():
            agents.append(Utils.agentid_consumer(f_id))
        return agents
    
    def set_discount_training_mode(self, mode):
        self.discount_reward_training = mode
    
    def set_training_mode(self, mode):
        self.training = mode

    def set_policies(self, policies):
        self.policies = policies

    def set_iteration(self, iteration, n_iterations):
        self.current_iteration = iteration
        self.n_iterations = n_iterations
    
    def set_constraint_imposed(self, mode):
        self.constraint_imposed = mode
    
    def n_products(self):
        return len(self._product_ids())

    def get_stock_status(self):
        stock_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                stock_info[facility_key] = facility.storage.stock_levels.get(sku_name, 0)
            elif isinstance(facility, FacilityCell):
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                stock_info[facility_key] = np.sum(list(facility.storage.stock_levels.values()))
        return stock_info

    def get_demand_status(self):
        demand_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                demand_info[facility_key] = facility.get_latest_sale()
            else:
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                demand_info[facility_key] = 0
        return demand_info

    def get_balance_status(self):
        balance_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                balance_info[facility_key] = facility.economy.step_balance.total()
            else:
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                balance_info[facility_key] = 0
        return balance_info
    
    def get_reward_status(self):
        reward_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                reward_info[facility_key] = facility.step_reward
            else:
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                reward_info[facility_key] = facility.step_reward
        return reward_info

    def get_reward_discount_status(self):
        reward_discount_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                if facility.consumer is not None:
                    reward_discount_info[facility_key] = facility.consumer.reward_discount
                else:
                    reward_discount_info[facility_key] = 0.0
            else:
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                reward_discount_info[facility_key] = 0.0
        return reward_discount_info
        
    def get_order_in_transit_status(self):
        order_in_transit_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                order_in_transit_info[facility_key] = 0
                if facility.consumer is not None and facility.consumer.sources is not None:
                    for source in facility.consumer.sources:
                        order_in_transit_info[facility_key] += facility.consumer.open_orders.get(source.id, {}).get(sku_name, 0)
            elif isinstance(facility, FacilityCell):
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                order_in_transit_info[facility_key] = 0
                for sku in facility.sku_in_stock:
                    if sku.consumer is not None and sku.consumer.sources is not None:
                        for source in sku.consumer.sources:
                            order_in_transit_info[facility_key] += sku.consumer.open_orders.get(source.id, {}).get(sku.sku_info['sku_name'], 0)
        return order_in_transit_info

    def get_order_to_distribute_status(self):
        order_to_distribute_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                order_to_distribute_info[facility_key] = 0
                if facility.distribution is not None:
                    order_to_distribute_info[facility_key] += facility.distribution.get_pending_order()[sku_name]
            elif isinstance(facility, FacilityCell):
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                order_to_distribute_info[facility_key] = 0
                for sku in facility.sku_in_stock:
                    if sku.distribution is not None:
                        order_to_distribute_info[facility_key] += sku.distribution.get_pending_order()[sku.sku_info['sku_name']]
        return order_to_distribute_info

    # 获取所有商品ID
    def _product_ids(self):
        return self.supply_network_config.get_all_skus()
