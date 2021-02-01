import env
import numpy as np
import random as rnd
from enum import Enum, auto
import math
import json


class InventoryEnvironmentConfig:

    def __init__(self):
        self.sku_config = None
        self.supplier_config = None
        self.warehouse_config = None
        self.store_config = None
        self.env_config = None
        self.demand_sampler = None
    
    def save_config(self, file_name):
        dump_str = ""
        for name, config in [('sku_config', self.sku_config), 
                       ('store_config', self.store_config),
                       ('warehouse_config', self.warehouse_config), 
                       ('supplier_config', self.supplier_config),
                       ('env_config', self.env_config), 
                       ('demand_sampler', self.demand_sampler)]:
            dump_str += f"{name} = {json.dumps(config)}\n"
        with open(file_name, 'w') as f:
            f.write(dump_str)
        return


    def load_config(self, config_name):
        if config_name == 'graph':
            from config.inventory_config_graph import sku_config, supplier_config, warehouse_config, store_config, env_config, demand_sampler
        elif config_name == 'tree':
            from config.inventory_config_tree import sku_config, supplier_config, warehouse_config, store_config, env_config, demand_sampler
        else:
            from config.inventory_config_vanilla import sku_config, supplier_config, warehouse_config, store_config, env_config, demand_sampler
        self.sku_config = sku_config
        self.store_config = store_config
        self.warehouse_config = warehouse_config
        self.supplier_config = supplier_config
        self.env_config = env_config
        self.demand_sampler = demand_sampler

    def get_env_config(self):
        return self.env_config

 
    def get_demand_sampler(self):
        return self.demand_sampler

    def get_reward_discount(self, inventory_level, is_retailer, echelon):
        """
        Given facility_type and the current inventory_level, return a reward discount. 
        The higher of the current inventory_level, the smaller of the reward discount.
        In this way, we discourage further replenishment if the current inventory is enough
        Args:
            inventory_level: expected days to sold out
            is_retailer: is the current facility a retailer
            echelon_level: the echelone level of the facility
        Return: a reward discount
        """
        inventory_level_bounds = [12, 7, 0]
        reward_discounts = [0.4, 0.9, 1.0]
        total_echelon = self.get_num_warehouse_echelon()
        if not is_retailer:
            inventory_level_bounds = [b+5*(total_echelon-echelon) for b in inventory_level_bounds]
        reward_discount = 1.05
        for i, b in enumerate(inventory_level_bounds):
            if inventory_level >= b:
                reward_discount = reward_discounts[i]
                break
        return math.pow(reward_discount, abs(inventory_level))

    def get_reward_discount_action_space(self):
        # discount_factors = [x/10.0 for x in range(-4, 10, 2)]
        discount_factors = [0.0, 0.5]
        # discount_factors = [0.0]
        return discount_factors

    def get_consumer_action_space(self):
        # procurement space
        consumer_quantity_space = [0,1,2,3,4,5,6,7]
        return consumer_quantity_space

    def get_consumer_quantity_action(self, consumer_quantity):
        consumer_action_space = self.get_consumer_action_space()
        for i, q in enumerate(consumer_action_space):
            if q > consumer_quantity:
                return i
        return len(consumer_action_space) - 1

    def get_sku_num(self):
        return self.sku_config['sku_num']

    def get_sku_name(self, sku_idx):
        return self.sku_config['sku_names'][sku_idx]

    def get_all_skus(self):
        return self.sku_config['sku_names']

    # get supplier config
    def get_supplier_num(self):
        return self.supplier_config['supplier_num']
    
    def get_supplier_name(self):
        return self.supplier_config['name']

    def get_supplier_short_name(self):
        return self.supplier_config['short_name']

    def get_supplier_info(self, supplier_idx):
        assert supplier_idx < self.get_supplier_num(), "supplier_idx must be less than total supplier number"
        keys = self.supplier_config.keys()
        supplier = {}
        for key in keys:
            if isinstance(self.supplier_config[key], list):
                supplier[key] = self.supplier_config[key][supplier_idx]
        return supplier

    def get_supplier_capacity(self, supplier_idx):
        return self.supplier_config['storage_capacity'][supplier_idx]

    def get_supplier_fleet_size(self, supplier_idx):
        assert supplier_idx < self.get_supplier_num(), "supplier_idx must be less than total supplier number"
        return self.supplier_config['fleet_size'][supplier_idx]

    def get_supplier_unit_storage_cost(self, supplier_idx):
        assert supplier_idx < self.get_supplier_num(), "supplier_idx must be less than total supplier number"
        return self.supplier_config['unit_storage_cost'][supplier_idx]

    def get_supplier_unit_transport_cost(self, supplier_idx):
        assert supplier_idx < self.get_supplier_num(), "supplier_idx must be less than total supplier number"
        return self.supplier_config['unit_transport_cost'][supplier_idx]

    def get_supplier_of_sku(self, sku_name):
        assert (sku_name in self.get_all_skus()), f"sku must be in {self.get_all_skus()}"
        supplier_list = []
        for supplier_idx in range(self.get_supplier_num()):
            sku_relation = self.supplier_config['sku_relation'][supplier_idx]
            for sku in sku_relation:
                if sku['sku_name'] == sku_name:
                    supplier_list.append(supplier_idx)
                    continue
        return supplier_list
    
    def get_sku_of_supplier(self, supplier_idx):
        assert supplier_idx < self.get_supplier_num(), "supplier_idx must be less than total supplier number"
        sku_relation = self.supplier_config['sku_relation'][supplier_idx]
        return sku_relation

    # get warehouse config
    def get_num_warehouse_echelon(self):
        return len(self.warehouse_config)

    def get_total_echelon(self):
        return len(self.warehouse_config) + 2

    def get_warehouse_num(self, i):
        return self.warehouse_config[i]['warehouse_num']

    def get_warehouse_name(self, i):
        return self.warehouse_config[i]['name']

    def get_warehouse_short_name(self, i):
        return self.warehouse_config[i]['short_name']

    def get_warehouse_info(self, i, warehouse_idx):
        assert warehouse_idx < self.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        keys = self.warehouse_config[i].keys()
        warehouse = {}
        for key in keys:
            if isinstance(self.warehouse_config[i][key], list):
                warehouse[key] = self.warehouse_config[i][key][warehouse_idx]
        return warehouse

    def get_warehouse_capacity(self, i, warehouse_idx):
        assert warehouse_idx < self.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return self.warehouse_config[i]['storage_capacity'][warehouse_idx]

    def get_warehouse_fleet_size(self, i, warehouse_idx):
        assert warehouse_idx < self.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return self.warehouse_config[i]['fleet_size'][warehouse_idx]

    def get_warehouse_unit_storage_cost(self, i, warehouse_idx):
        assert warehouse_idx < self.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return self.warehouse_config[i]['unit_storage_cost'][warehouse_idx]


    def get_warehouse_unit_transport_cost(self, i, warehouse_idx):
        assert warehouse_idx < self.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return self.warehouse_config[i]['unit_transport_cost'][warehouse_idx]


    def get_warehouse_of_sku(self, i, sku_name):
        assert (sku_name in self.get_all_skus()), f"sku must be in {self.get_all_skus()}"
        warehouse_list = []
        for warehouse_idx in range(self.get_warehouse_num(i)):
            sku_relation = self.warehouse_config[i]['sku_relation'][warehouse_idx]
            for sku in sku_relation:
                if sku['sku_name'] == sku_name:
                    warehouse_list.append(warehouse_idx)
                    continue
        return warehouse_list
    

    def get_sku_of_warehouse(self, i, warehouse_idx):
        assert warehouse_idx < self.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        sku_relation = self.warehouse_config[i]['sku_relation'][warehouse_idx]
        return sku_relation
    
    # get store config
    def get_store_num(self):
        return self.store_config['store_num']


    def get_store_name(self):
        return self.store_config['name']

    def get_store_short_name(self):
        return self.store_config['short_name']

    def get_store_info(self, store_idx):
        assert store_idx < self.get_store_num(), "store_idx must be less than total store number"
        keys = self.store_config.keys()
        store = {}
        for key in keys:
            if isinstance(self.store_config[key], list):
                store[key] = self.store_config[key][store_idx]
        return store

    def get_store_capacity(self, store_idx):
        assert store_idx < self.get_store_num(), "store_idx must be less than total store number"
        return self.store_config['storage_capacity'][store_idx]

    def get_store_unit_storage_cost(self, store_idx):
        assert store_idx < self.get_store_num(), "store_idx must be less than total store number"
        return self.store_config['unit_storage_cost'][store_idx]

    def get_store_unit_transport_cost(self, store_idx):
        assert store_idx < self.get_store_num(), "store_idx must be less than total store number"
        return self.store_config['unit_transport_cost'][store_idx]

    def get_store_of_sku(self, sku_name):
        assert (sku_name in self.get_all_skus()), f"sku must be in {self.get_all_skus()}"
        store_list = []
        for store_idx in range(self.get_store_num()):
            sku_relation = self.store_config['sku_relation'][store_idx]
            for sku in sku_relation:
                if sku['sku_name'] == sku_name:
                    store_list.append(store_idx)
                    continue
        return store_list
    
    def get_sku_of_store(self, store_idx):
        assert store_idx < self.get_store_num(), "store_idx must be less than total store number"
        sku_relation = self.store_config['sku_relation'][store_idx]
        return sku_relation


class Utils:

    @staticmethod
    def agentid_producer(facility_id):
        return facility_id + 'p'
    
    @staticmethod
    def agentid_consumer(facility_id):
        return facility_id + 'c'
    
    @staticmethod
    def is_producer_agent(agent_id):
        return agent_id[-1] == 'p'
    
    @staticmethod
    def is_consumer_agent(agent_id):
        return agent_id[-1] == 'c'
    
    @staticmethod
    def agentid_to_fid(agent_id):
        return agent_id[:-1]


