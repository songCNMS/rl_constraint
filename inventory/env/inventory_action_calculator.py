import copy 

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
import yaml
import time

import ray.rllib.agents.ppo.ppo

from gym.spaces import Box, Tuple, MultiDiscrete, Discrete
import numpy as np
from pprint import pprint
from collections import OrderedDict 
from dataclasses import dataclass
import random as rnd
import statistics
from itertools import chain
from collections import defaultdict
from env.inventory_utils import Utils
from agents.inventory import *
from agents.inventory_order import OuterSKUStoreUnit


class ActionCalculator:
    
    def __init__(self, env):
        self.env = env
    
    def action_dictionary_to_control(self, action_dict, world):
        actions_by_facility = defaultdict(list)
        for agent_id, action in action_dict.items():
            f_id = Utils.agentid_to_fid(agent_id)
            actions_by_facility[f_id].append((agent_id, action)) 
        
        controls = {}
        for f_id, actions in actions_by_facility.items():
            controls[f_id] = self._actions_to_control( world.facilities[ f_id ], actions )
        return World.Control(facility_controls = controls)
    
    # 生成Action -- 即Control
    def _actions_to_control(self, facility, actions):       
        control = FacilityCell.Control(
            unit_price = 0,
            production_rate = 0,
            consumer_product_id = 0,
            consumer_source_id = 0,
            consumer_quantity = 0,
            consumer_vlt = 0,
            reward_discount = 0
        ) 

        consumer_action_list = self.env.supply_network_config.get_consumer_action_space()
        reward_discount_list = self.env.supply_network_config.get_reward_discount_action_space()
        if isinstance(facility, FacilityCell):
            return control

        for agent_id, action in actions:
            action = np.array(action).flatten()
            if Utils.is_producer_agent(agent_id):
                if isinstance(facility, SKUSupplierUnit):
                    control.production_rate = facility.sku_info['production_rate']
            if Utils.is_consumer_agent(agent_id):
                product_id = facility.bom.output_product_id 
                control.consumer_product_id = product_id
                if facility.consumer.sources is not None:
                    source = facility.consumer.sources[0]
                    control.consumer_vlt = source.sku_info['vlt']
                    control.consumer_source_id = 0
                if self.env.is_reward_shape_on:
                    control.consumer_quantity = consumer_action_list[action[1]] *  facility.get_sale_mean()
                    if not self.env.discount_reward_training:
                        control.reward_discount = reward_discount_list[action[0]]
                    else:
                        control.reward_discount = 0
                else:
                    control.consumer_quantity = consumer_action_list[action[0]] *  facility.get_sale_mean()
                    control.reward_discount = 0
        return control