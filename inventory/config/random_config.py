import random
import numpy as np
from flloat.parser.ltlf import LTLfParser

demand_sampler = 'DYNAMIC_GAMMA'
# demand_sampler = 'GAMMA'

constraints = ['G(stock_constraint)',
               'G(is_replenish_constraint -> ((X!is_replenish_constraint)&(XX!is_replenish_constraint)))',
               'G(low_profit -> low_stock_constraint)']

# constraints = ['G(is_replenish_constraint -> ((X!is_replenish_constraint)&(XX!is_replenish_constraint)))']

def construct_formula(constraint):
    parser = LTLfParser()
    formula = parser(constraint)
    return formula


constraint_formulas = {constraint: construct_formula(constraint) for constraint in constraints}
constraint_automata = {constraint: constraint_formulas[constraint].to_automaton().determinize() for constraint in constraints}

max_constraint_states = int(np.max([len(a.states) for a in constraint_automata.values()])) + 1

env_config = {
    'global_reward_weight_producer': 0.50,
    'global_reward_weight_consumer': 0.50,
    'downsampling_rate': 1,
    'episod_duration': 21,
    "initial_balance": 100000,
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "constraint_state_hist_len": max_constraint_states, 
    "total_echelons": 3,
    "replenishment_discount": 0.9, 
    "reward_normalization": 1e7,
    "constraint_violate_reward": -1e6,
    "gamma": 0.99,
    "tail_timesteps": 7,
    "heading_timesteps": 7
}

sku_num = random.randint(6, 8)

# sku_num = 2

sku_config = {
    "sku_num": sku_num,
    "sku_names": ['SKU%i' % i for i in range(sku_num)]
}


sku_cost = {'SKU%i' % i: random.randint(10, 500) for i in range(sku_num)}
sku_product_cost = {'SKU%i' % i:  int(sku_cost['SKU%i' % i] * 0.9) for i in range(sku_num)}
sku_price = {'SKU%i' % i: int(sku_cost['SKU%i' % i] * (1 + random.randint(10, 100) / 100)) for i in range(sku_num)}
sku_gamma = {'SKU%i' % i: random.randint(5, 100) for i in range(sku_num)}
total_gamma = sum(list(sku_gamma.values()))

sku_vlt = {'SKU%i' % i: random.randint(1, 3) for i in range(sku_num)}

supplier_config = {
    "name": 'SUPPLIER',
    "short_name": "M",
    "supplier_num": 1,
    "fleet_size": [10*sku_num],
    "unit_storage_cost": [1],
    "unit_transport_cost": [1],
    "storage_capacity": [total_gamma*100],
    "order_cost": [200],
    "delay_order_penalty": [1000],
    "downstream_facilities": [[0]], 
    "sku_relation": [
        [{"sku_name": "SKU%i" % i, 
          "price": sku_cost["SKU%i" % i], 
          "cost": sku_product_cost["SKU%i" % i], 
          "service_level": .95, 
          "vlt": 3, 
          "init_stock": int(sku_gamma["SKU%i" % i]*50), 
          "production_rate": int(sku_gamma["SKU%i" % i]*50)} 
          for i in range(sku_num)]
    ]
}

warehouse_config = [
    {
        "name": "WAREHOUSE",
        "short_name": "R",
        "warehouse_num": 1,
        "fleet_size": [10*sku_num],
        "unit_storage_cost": [1],
        "unit_transport_cost": [1],
        "storage_capacity": [total_gamma*100],
        "order_cost": [500],
        "delay_order_penalty": [1000], 
        "downstream_facilities": [[0]],
        "sku_relation": [
            [{"sku_name": "SKU%i" % i, "service_level": .96, 
              "vlt": sku_vlt["SKU%i" % i], 
              "price": sku_cost["SKU%i" % i], 
              "cost": sku_cost["SKU%i" % i], 
              'init_stock': int(sku_gamma["SKU%i" % i]*20)}
              for i in range(sku_num)]
        ]
    }
]

import time
sku_constraints = {}
# sku_constraints["SKU0"] = constraints[1]
# sku_constraints = {}

constraint_num = 0
for i in range(sku_num):
    time.sleep(0.5)
    if random.random() <= 0.3:
        constraint_num += 1
        sku_constraints["SKU%i" % i] = constraints[random.randint(0, len(constraints)-1)]

if constraint_num == 0:
    sku_constraints["SKU0"] = constraints[1]


store_config = {
    "name": "STORE",
    "short_name": "S",
    "store_num": 1,
    "storage_capacity": [total_gamma*2000],
    "unit_storage_cost": [1],
    "order_cost": [500],
    "sku_relation": [
        [{"sku_name": "SKU%i" % i, 
          "price": sku_price["SKU%i" % i], "service_level": 0.95, 
          "cost": sku_cost["SKU%i" % i], 
          "sale_gamma": sku_gamma["SKU%i" % i], 
          'init_stock': sku_gamma["SKU%i" % i] * (sku_vlt["SKU%i" % i] + random.randint(1, 3)), 
          'max_stock': 1000,
          "constraint": sku_constraints.get("SKU%i" % i, None)}
         for i in range(sku_num)]
    ]
}


