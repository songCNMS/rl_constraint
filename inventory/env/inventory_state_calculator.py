from abc import ABC
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
import networkx as nx
import scipy.stats as st
import statistics
from enum import Enum, auto
from agents.base import Cell, Agent, BalanceSheet
from agents.inventory import *
from env.inventory_utils import Utils
from constraints.constraint_atoms import *


class StateCalculator:
    
    def __init__(self, env):
        self.env = env

    # 生成每个falicity的状态，附加全局信息
    def world_to_state(self, world):
        state = {}
        for facility_id, facility in world.facilities.items():
            f_state = self._state(facility)  
            self._add_global_features(f_state, world)
            state[Utils.agentid_producer(facility_id)] = f_state
            state[Utils.agentid_consumer(facility_id)] = f_state
            
        return self._serialize_state(state), state  
    
    # 
    def _state(self, facility):
        state = {}

        # add features to state
        # features adding order is important
        self._add_facility_features(state, facility)
        self._add_storage_features(state, facility)        
        self._add_bom_features(state, facility)
        self._add_distributor_features(state, facility)
        self._add_sale_features(state, facility)   
        self._add_vlt_features(state, facility)  
        self._add_consumer_features(state, facility)
        self._add_price_features(state, facility)       
                    
        return state

    def _add_global_features(self, state, world):
        state['global_time'] = world.time_step # / self.env.env_config['episod_duration']
        # state['global_storage_utilization'] = [ f.storage.used_capacity() / f.storage.max_capacity for f in world.facilities.values() ]
        #state['balances'] = [ self._balance_norm(f.economy.total_balance.total()) for f in world.facilities.values() ]
        
    def _balance_norm(self, v):
        return v/1000

    
    def _add_facility_features(self, state, _facility):
        # 对facility类型进行one-hot encoding
        facility_type = [0] * len(self.env.facility_types)
        facility_type[self.env.facility_types[_facility.__class__.__name__]] = 1
        state['facility_type'] = facility_type
        state['facility'] = _facility
        state['is_accepted'] = [0] * self.env.env_config['constraint_state_hist_len']
        state['constraint_idx'] = _facility.constraint_idx
        for atom_name in atoms.keys():
            state[atom_name] = list(np.ones(self.env.env_config['constraint_state_hist_len']))
        # one-hot encoding of facility id w.r.t. facilities of the same type
        sku_list = sorted(self.env.supply_network_config.get_all_skus())
        facility_id_one_hot = [0] * len(sku_list)
        state['facility_id'] = facility_id_one_hot
        
        if isinstance(_facility, FacilityCell):
            state['facility_info'] = _facility.facility_info
            state['sku_info'] = {}
            state['is_positive_balance'] = 1 if _facility.economy.total_balance.total() > 0 else 0
        else:
            state['facility_info'] = _facility.facility.facility_info
            state['sku_info'] = _facility.sku_info
            state['is_positive_balance'] = 1 if _facility.facility.economy.total_balance.total() > 0 else 0
            if _facility.constraint_automaton_state is not None:
                # state['is_accepted'] = list(_facility.constraint_state_hist)
                state['is_accepted'][_facility.constraint_automaton_state] = 1
            sku_name = _facility.sku_info['sku_name']
            state['facility_id'][sku_list.index(sku_name)] = 1

            for atom_name in atoms.keys():
                state[atom_name] = list(_facility.atom_valid_hist[atom_name])

        # add echelon feature
        facility = _facility
        if isinstance(facility, ProductUnit):
            facility = _facility.facility
        if isinstance(facility, WarehouseCell):
            # reserve 0 for supplier 
            state['echelon_level'] = facility.echelon_level + 1
        elif isinstance(facility, SupplierCell):
            state['echelon_level'] = 0
        else:
            state['echelon_level'] = self.env.world.supply_network_config.env_config['total_echelons']

    # storage related features
    def _add_storage_features(self, state, _facility):
        state['storage_levels'] = [0] * self.env.n_products()
        if isinstance(_facility, FacilityCell):
            facility = _facility
        else:
            facility = _facility.facility
        state['storage_capacity'] = facility.storage.max_capacity
        for i, prod_id in enumerate(self.env.product_ids):
            if prod_id in facility.storage.stock_levels.keys():
                state['storage_levels'][i] = facility.storage.stock_levels[prod_id]
        state['storage_utilization'] = facility.storage.total_stock

    # BOM相关特征
    def _add_bom_features(self, state, _facility):
        state['bom_inputs'] = [0] * self.env.n_products()  
        state['bom_outputs'] = [0] * self.env.n_products()
        if isinstance(_facility, FacilityCell):
            return
        prod_id = _facility.sku_info['sku_name']
        state['bom_inputs'][self.env.product_ids.index(prod_id)] = 1
        state['bom_outputs'][self.env.product_ids.index(prod_id)] = 1
        # for i, prod_id in enumerate(self.env.product_ids):
            # for sku in sku_in_stock:
            #     if prod_id in sku.bom.inputs.keys():
            #         state['bom_inputs'][i] = sku.bom.inputs[prod_id]
            #     if prod_id == sku.bom.output_product_id:
            #         state['bom_outputs'][i] = sku.bom.output_lot_size

    # VLT 信息
    def _add_vlt_features(self, state, _facility):
        state['vlt'] = [0] * (self.env.max_sources_per_facility * self.env.n_products())
        state['max_vlt'] = 0
        if not isinstance(_facility, ProductUnit):
            return
        if _facility.consumer is None or _facility.consumer.sources is None:
            return

        state['max_vlt'] = _facility.get_max_vlt()
        sku = _facility
        for i, source in enumerate(sku.consumer.sources):  
            for j, product_id in enumerate(self.env.product_ids):
                if source.bom.output_product_id == product_id:
                    state['vlt'][i*self.env.n_products() + j] = source.sku_info.get('vlt', 2)

    # Sale 信息
    def _add_sale_features(self, state, _facility):
        state['sale_mean'] = 1.0
        state['sale_std'] = 1.0
        state['sale_gamma'] = 1.0
        state['service_level'] = 0.95
        state['total_backlog_demand'] = 0
        hist_len = self.env.env_config['sale_hist_len']
        state['sale_hist'] = [0] * hist_len
        state['backlog_demand_hist'] = [0] * hist_len
        state['consumption_hist'] = [0] * self.env.env_config['consumption_hist_len']
        state['pending_order'] = [0] * self.env.env_config['pending_order_len']
        if not isinstance(_facility, ProductUnit):
             return
        state['service_level'] = _facility.sku_info['service_level']
        state['sale_mean'] = _facility.get_sale_mean()
        state['sale_gamma'] = state['sale_mean']
        state['sale_std'] = _facility.get_sale_std()
        if _facility.consumer is not None:
            state['consumption_hist'] = list(_facility.consumer.latest_consumptions)
            state['pending_order'] = list(_facility.consumer.pending_order_daily)
        if _facility.seller is not None:
            state['total_backlog_demand']  = _facility.seller.total_backlog_demand
            state['sale_hist'] = _facility.seller.sale_hist
            state['backlog_demand_hist'] = _facility.seller.backlog_demand_hist
            state['sale_gamma'] = _facility.sku_info['sale_gamma']

    # 计算在途订单的数量和商品数量
    def _add_distributor_features(self, state, _facility):
        state['distributor_in_transit_orders'] = 0
        state['distributor_in_transit_orders_qty'] = 0
        if not isinstance(_facility, FacilityCell):
            facility = _facility.facility
        else:
            facility = _facility
        if facility.distribution is not None:
            q = facility.distribution.order_queue
            ordered_quantity = sum([ order.quantity for order in q ])
            state['distributor_in_transit_orders'] = len(q)
            state['distributor_in_transit_orders_qty'] = ordered_quantity

    # 构建商品配送网络特征        
    def _add_consumer_features(self, state, _facility):
        # which source exports which product
        # state['consumer_source_export_mask'] = [0] * (self.env.max_sources_per_facility * self.env.n_products())
        # provide the agent with the statuses of tier-one suppliers' inventory and in-transit orders
        state['consumer_source_inventory'] = [0] * self.env.n_products()
        state['consumer_in_transit_orders'] = [0] * self.env.n_products()

        # local information, only for a particular SKU
        state['inventory_in_stock'] = 0
        state['inventory_in_transit'] = 0
        state['inventory_in_distribution'] = 0
        state['inventory_estimated'] = 0
        state['inventory_rop'] = 0
        state['is_over_stock'] = 0
        state['is_out_of_stock'] = 0
        state['is_below_rop'] = 0

        if not isinstance(_facility, ProductUnit):
            return
        if _facility.consumer is None or _facility.consumer.sources is None:
            return

        sku = _facility
        
        # global information, which contains information of all other SKUs
        # for i, source in enumerate(sku.consumer.sources):  
        #     for j, product_id in enumerate(self.env.product_ids):
        #         if source.bom.output_product_id == product_id:
        #             state['consumer_source_export_mask'][i*self.env.n_products() + j] = 1
        
        
        in_transit_orders = sku.facility.get_in_transit_order()
        for i, product_id in enumerate(self.env.product_ids):
            state['consumer_in_transit_orders'][i] += in_transit_orders[product_id]

        
        # in stock
        state['inventory_in_stock'] = sku.facility.storage.stock_levels.get(sku.sku_info['sku_name'], 0)
        # in transit
        state['inventory_in_transit'] = state['consumer_in_transit_orders'][self.env.product_ids.index(sku.sku_info['sku_name'])]
        # to be delivered to downstreaming facilities
        if sku.facility.distribution is not None:
            state['inventory_in_distribution'] = sku.facility.distribution.get_pending_order()[sku.sku_info['sku_name']]

        state['inventory_estimated'] = (state['inventory_in_stock'] 
                                        + state['inventory_in_transit']
                                        - state['inventory_in_distribution'])
        if (state['inventory_estimated'] >= 0.5*state['storage_capacity']):
            state['is_over_stock'] = 1
        
        if (state['inventory_estimated'] <= 0):
            state['is_out_of_stock'] = 1
        
        state['inventory_rop'] = (state['max_vlt']*state['sale_mean'] 
                                  + np.sqrt(state['max_vlt'])*state['sale_std']*st.norm.ppf(state['service_level']))
        
        if state['inventory_estimated'] < state['inventory_rop']:
            state['is_below_rop'] = 1
        
    def _add_price_features(self, state, _facility):
        state['max_price'] = self.env.world.max_price
        state['sku_price'] = 0
        state['sku_cost'] = 0
        if isinstance(_facility, ProductUnit):
            state['sku_price'] = _facility.sku_info.get('price', 0)
            state['sku_cost'] = _facility.sku_info.get('cost', 0)

    def _safe_div(x, y):
        if y != 0:
            return x
        return 0
    
    # Convert state to vector and normalized
    # Currently, normalization is done by storage capacity of the facility each product belongs to
    def _serialize_state(self, state):
        result = {}
        # not all information in state will be used for RL algorithm
        # define information used in RL as well as their normalization here
        # 'echelon_level', 'is_accepted', 'facility_id',  + list(atoms.keys())
        keys_in_state = [(None, ['is_over_stock', 'is_out_of_stock', 'is_below_rop', 
                                 'constraint_idx', 'is_accepted', 'consumption_hist']),
                         ('storage_capacity', ['storage_utilization']),
                         ('sale_gamma', ['sale_std',
                                         'sale_hist',
                                         'pending_order',
                                         'inventory_in_stock', 
                                         'inventory_in_transit',
                                         'inventory_estimated',
                                         'inventory_rop']),
                         ('max_price', ['sku_price', 'sku_cost'])]
        
        for k, v in state.items():
            result[k] = []
            for norm, feilds in keys_in_state:
                for feild in feilds:
                    vals = v[feild]
                    if not isinstance(vals, list):
                        vals = [vals]
                    if norm is not None:
                        vals = [max(0.0, min(100.0, x/(v[norm]+0.01))) for x in vals]
                    result[k].extend(vals)    
            result[k] = np.array(result[k])
        # print(keys_in_state, result)
        return result
