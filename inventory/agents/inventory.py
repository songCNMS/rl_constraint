from abc import ABC
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
import math
from scipy.ndimage.interpolation import shift

from numpy.lib.arraysetops import isin
import networkx as nx
from enum import Enum, auto
from ray.rllib.utils.annotations import override
from agents.base import Cell, Agent, BalanceSheet
from flloat.parser.ltlf import LTLfParser
from constraints.constraint_atoms import *
from config.random_config import constraint_automata, constraint_formulas, constraints
    

class TerrainCell(Cell):
    def __init__(self, x, y):
        super(TerrainCell, self).__init__(x, y)

class RailroadCell(Cell):
    def __init__(self, x, y):
        super(RailroadCell, self).__init__(x, y)

        
# ======= Transportation
class Transport(Agent):
    
    @dataclass 
    class Economy:
        unit_transport_cost: int   # cost per unit per movement
        def step_balance_sheet(self, transport):
            return BalanceSheet(0, -transport.payload * self.unit_transport_cost)
        def step_reward(self, transport): # reward/cost for RL training
            return -transport.payload * self.unit_transport_cost
    
    @dataclass
    class Control:
        pass
    
    def __init__(self, source, economy):
        self.source = source
        self.destination = None
        self.path = None
        self.location_pointer = 0
        self.step = 0
        self.payload = 0 # units
        self.economy = economy
        self.patient = 100
        self.cur_time = 0
        self.product_id = None

    def schedule(self, world, destination, product_id, quantity, vlt):
        self.destination = destination
        self.product_id = product_id
        self.requested_quantity = quantity
        self.path = world.find_path(self.source.x, self.source.y, self.destination.facility.x, self.destination.facility.y)
        if self.path == None:
            raise Exception(f"Destination {destination} is unreachable")

        # positive: to destination, negative: back to source, otherwise: stay in source
        # compute step based on path len and vlt
        # print(self.destination.id, self.product_id, self.requested_quantity, vlt)
        self.step = max(1, self.path_len() // vlt)
        self.sku = None
        for _sku in self.source.sku_in_stock:
            if _sku.bom.output_product_id == self.product_id:
                self.sku = _sku
                break
        self.location_pointer = 0

    def path_len(self):
        if self.path is None:
            return 0
        else:
            return len(self.path)
    
    def is_enroute(self):
        return self.destination is not None

    def current_location(self):
        if self.path is None:
            return (self.source.x, self.source.y)
        else:
            return self.path[self.location_pointer]
        
    def try_loading(self, quantity):
        # available_stock = self.source.storage.take_available(self.product_id, quantity)
        if self.source.storage.try_take_units({self.product_id: quantity}):
            self.payload = self.requested_quantity
        else:
            self.payload += self.source.storage.take_available(self.product_id, self.requested_quantity - self.payload)
            self.cur_time += 1
        
    def try_unloading(self):        
        unloaded = self.destination.storage.try_add_units({ self.product_id: self.payload }, all_or_nothing = False)
        if len(unloaded) > 0:
            unloaded_units = sum(unloaded.values())
            self.destination.consumer.on_order_reception(self.sku.id, self.product_id, unloaded_units, self.payload)
            self.payload = 0    # all units that were not sucessfully unloaded will be lost

    def reset(self):
        self.__init__(self.source, self.economy)

    def act(self, control):
        if self.step > 0: 
            if self.location_pointer == 0 and self.payload < self.requested_quantity:
                self.try_loading(self.requested_quantity-self.payload)
            
            if self.cur_time > self.patient or self.payload == self.requested_quantity:     # will stay at the source until loaded or timeout
                self.cur_time = 0 # reset cur_time
                if self.location_pointer < len(self.path) - 1:
                    self.location_pointer = min(len(self.path)-1, self.location_pointer+self.step)
                else:
                    self.step = -self.step   # arrived to the destination

        if self.step < 0: 
            if self.location_pointer == len(self.path) - 1 and self.payload > 0:
                self.try_unloading()
                
            if self.payload == 0:    # will stay at the destination until unloaded
                if self.location_pointer > 0: 
                    self.location_pointer = max(0, self.location_pointer+self.step)
                else:
                    self.step = 0    # arrived back to the source
                    self.destination = None
        
        return self.economy.step_balance_sheet(self), self.economy.step_reward(self)


# ======= Basic facility components (units)

@dataclass    
class BillOfMaterials:
    # One manufacturing cycle consumes inputs 
    # and produces output_lot_size units of output_product_id
    inputs: Counter  # (product_id -> quantity per lot)
    output_product_id: str
    output_lot_size: int = 1
        
    def input_units_per_lot(self):
        return sum(self.inputs.values())
        

class StorageUnit(Agent):
    
    @dataclass 
    class Economy:
        unit_storage_cost: int    # cost per unit per time step

        def step_balance_sheet(self, storage):
            return BalanceSheet(0, -storage.used_capacity() * self.unit_storage_cost)

        def step_reward(self, storage):
            return -storage.used_capacity() * self.unit_storage_cost
        
    @dataclass
    class Config:
        max_storage_capacity: int
        unit_storage_cost: int
    
    def __init__(self, max_capacity, economy):
        self.max_capacity = max_capacity
        self.stock_levels = Counter()
        self.economy = economy
        self.total_stock = 0
    
    def used_capacity(self):
        return self.total_stock
    
    def available_capacity(self):
        return self.max_capacity - self.total_stock
    
    def try_add_units(self, product_quantities, all_or_nothing = True) -> dict:
        
        # validation
        if all_or_nothing and self.available_capacity() < sum(product_quantities.values()):
            return {}
        
        # depositing
        unloaded_quantities = {}
        for p_id, q in product_quantities.items(): 
            unloading_qty = min(self.available_capacity(), q)
            self.stock_levels[p_id] += unloading_qty
            self.total_stock += unloading_qty
            unloaded_quantities[p_id] = unloading_qty
            
        return unloaded_quantities
    
    def try_take_units(self, product_quantities):
        # validation
        for p_id, q in product_quantities.items():
            if self.stock_levels[p_id] < q:
                return False
        # withdrawal
        for p_id, q in product_quantities.items():
            self.stock_levels[p_id] -= q  
            self.total_stock -= q
        return True
    
    def take_available(self, product_id, quantity):
        available = self.stock_levels[product_id]
        actual = min(available, quantity)
        self.stock_levels[product_id] -= actual
        self.total_stock -= actual
        return actual

    def reset(self):
        self.__init__(self.max_capacity, self.economy)
    
    def act(self, control = None):
        # Balance and reward of storage are calulated per product
        # return BalanceSheet(0, 0), 0
        return self.economy.step_balance_sheet(self), 0
        

class DistributionUnit(Agent):
    @dataclass 
    class Economy:
        unit_price: int = 0
        order_checkin: int = 0  # balance for the current time step
            
        def profit(self, units_sold):
            return self.unit_price * units_sold
        
    @dataclass
    class Config:  
        fleet_size: int
        unit_transport_cost: int
    
    @dataclass
    class Control:
        unit_price: int                            
    
    @dataclass
    class Order:
        destination: Cell
        product_id: str
        quantity: int
        vlt: int
    
    def __init__(self, facility, fleet_size, distribution_economy, transport_economy):
        self.facility = facility
        self.fleet = [ Transport(facility, transport_economy) for i in range(fleet_size) ]
        self.order_queue = deque()
        self.economy = distribution_economy
        self.checkin_order = Counter()
        self.transportation_cost = Counter()
        self.delay_order_penalty = Counter()
        self.is_reset = True
        self.pending_orders = Counter()
    

    # get number of sku that needs to be delivered to downstream facilities
    def get_pending_order(self):
        if self.is_reset:
            self.pending_orders = Counter()
            for order in self.order_queue:
                self.pending_orders[order.product_id] += order.quantity
            for vechicle in self.fleet:
                if vechicle.is_enroute():
                    self.pending_orders[vechicle.product_id] += (vechicle.requested_quantity - vechicle.payload)
            self.is_reset = False
        return self.pending_orders
        
    def place_order(self, order):
        if order.quantity > 0:
            for sku in self.facility.sku_in_stock:
                if sku.bom.output_product_id == order.product_id:
                    self.order_queue.append(order)   # add order to the queue
                    order_total = sku.sku_info['price'] * order.quantity
                    self.checkin_order[order.product_id] += order.quantity
                    return order_total
        return 0

    def reset(self):
        self.is_reset = True
        self.pending_orders = Counter()
        for f in self.fleet:
            f.reset()
        self.order_queue = deque()
        self.economy.order_checkin = 0
        self.checkin_order = Counter()
        self.transportation_cost = Counter()
        self.delay_order_penalty = Counter()

    def act(self, control):
        # if control is not None:
            # self.economy.unit_price = control.unit_price    # update unit price
        self.is_reset = True
        step_balance = BalanceSheet()
        delay_order_penalty = self.facility.facility_info.get('delay_order_penalty', 0)
        
        for vechicle in self.fleet:
            if vechicle.is_enroute() and vechicle.payload < vechicle.requested_quantity:
                self.delay_order_penalty[vechicle.product_id] += delay_order_penalty
            if len(self.order_queue) > 0 and not vechicle.is_enroute():
                order = self.order_queue.popleft()
                vechicle.schedule( self.facility.world, order.destination, order.product_id, order.quantity, order.vlt )
                transportation_balance, reward = vechicle.act(None)
                self.transportation_cost[order.product_id] += abs(reward)
            else:
                transportation_balance, reward = vechicle.act(None)
                self.transportation_cost[vechicle.product_id] += abs(reward)
            step_balance += transportation_balance
        for order in self.order_queue:
            self.delay_order_penalty[order.product_id] += delay_order_penalty
        # Distribution balance and reward 
        return step_balance, 0

    
class ManufacturingUnit(Agent):
    @dataclass 
    class Economy:
        unit_cost: int                   # production cost per unit 
            
        def cost(self, units_produced):
            return -self.unit_cost * units_produced
        
        def step_balance_sheet(self, units_produced):
            return BalanceSheet(0, self.cost(units_produced))
        
    @dataclass
    class Config:
        unit_manufacturing_cost: int
    
    @dataclass
    class Control:
        production_rate: int                  # lots per time step
    
    def __init__(self, facility, economy):
        self.facility = facility
        self.economy = economy
        self.manufacturing_cost = 0

    def reset(self):
        self.manufacturing_cost = 0
    
    def act(self, control):
        units_produced = 0
        sku_num = len(self.facility.facility.sku_in_stock)
        unit_num_upper_bound = self.facility.storage.max_capacity // sku_num
        if control is not None:
            units_produced = max(0, min(unit_num_upper_bound-self.facility.storage.stock_levels[self.facility.bom.output_product_id], control.production_rate))
            self.facility.storage.stock_levels[self.facility.bom.output_product_id] += units_produced
            # for _ in range(control.production_rate):
            #     # check we have enough storage space for the output lot
            #     if ( self.facility.storage.available_capacity() >= self.facility.bom.output_lot_size - self.facility.bom.input_units_per_lot() 
            #             and self.facility.storage.stock_levels[self.facility.bom.output_product_id] < unit_num_upper_bound ): 
            #         # check we have enough input materials 
            #         if self.facility.storage.try_take_units(self.facility.bom.inputs):                
            #             self.facility.storage.stock_levels[self.facility.bom.output_product_id] += self.facility.bom.output_lot_size
            #             units_produced += self.facility.bom.output_lot_size
        self.manufacturing_cost = self.facility.sku_info['cost'] * units_produced
        return BalanceSheet(0, -self.manufacturing_cost), -self.manufacturing_cost  

    
class ConsumerUnit(Agent):
    @dataclass
    class Economy:
        order_cost: int
        total_units_purchased: int = 0
        total_units_received: int = 0
    
    @dataclass
    class Control:
        consumer_product_id: int  # what to purchase
        consumer_source_id: int   # where to purchase  
        consumer_quantity: int    # how many to purchase
        consumer_vlt: int         # how many days it will take for this order being fulfilled by default
        reward_discount: float
            
    @dataclass
    class Config:
        sources: list
    
    def __init__(self, facility, sources, economy):
        self.facility = facility
        self.sources = sources
        self.open_orders = {}
        self.economy = economy
        self.products_received = 0
        self.lost_product_value = 0
        self.latest_consumptions = [0] * self.facility.world.supply_network_config.env_config['consumption_hist_len']
        self.pending_order_daily = [0] * self.facility.world.supply_network_config.env_config['pending_order_len']
        self.reward_discount = 0.0
    
    def on_order_reception(self, source_id, product_id, quantity, original_quantity):
        self.economy.total_units_received += quantity
        self.lost_product_value += (original_quantity - quantity)
        self.products_received += original_quantity
        # if quantity is less than original_quantity
        # no enough space for extra quantities which will be lost
        self._update_open_orders(source_id, product_id, -original_quantity)

    def get_in_transit_quantity(self):
        quantity = 0
        product_id = self.facility.sku_info['sku_name']
        for source_id in self.open_orders:
            quantity += self.open_orders[source_id].get(product_id, 0)
        return quantity

    def _update_pending_order(self):
        self.pending_order_daily = shift(self.pending_order_daily, -1, cval=0)

    def _update_latest_consumptions(self):
        self.latest_consumptions = shift(self.latest_consumptions, -1, cval=0)
    
    def _update_open_orders(self, source_id, product_id, qty_delta):
        if qty_delta > 0:
            if source_id not in self.open_orders:
                self.open_orders[source_id] = Counter()
            self.open_orders[source_id][product_id] += qty_delta
        else:
            self.open_orders[source_id][product_id] += qty_delta
            self.open_orders[source_id] += Counter() # remove zeros
            if len(self.open_orders[source_id]) == 0:
                del self.open_orders[source_id]
    
    def reset(self):
        self.open_orders = {}
        self.economy.total_units_received = 0
        self.economy.total_units_purchased = 0
        self.lost_product_value = 0
        self.products_received = 0
        self.latest_consumptions = [0] * self.facility.world.supply_network_config.env_config['consumption_hist_len']
        self.pending_order_daily = [0] * self.facility.world.supply_network_config.env_config['pending_order_len']
        

    def act(self, control):
        self._update_pending_order()
        self._update_latest_consumptions()
        if control is None or control.consumer_product_id is None or control.consumer_quantity <= 0 or self.sources is None:
            return BalanceSheet(), 0
        
        source_obj = self.sources[control.consumer_source_id]
        # sku_cost = source_obj.sku_info.get('cost', 0)
        # order_cost =  - self.economy.order_cost - sku_cost * control.consumer_quantity
        # TODO: no enough money, only for RetailerCell now
        # if isinstance(self.facility.facility, RetailerCell):
        #     # print(self.facility.id, self.facility.facility.economy.total_balance.total(), order_cost)
        #     if self.facility.facility.economy.total_balance.total() < abs(order_cost):
        #         return BalanceSheet()
        
        # source_service_level = source_obj.sku_info.get('service_level', 1.0)
        # if rnd.random() >= source_service_level:
            # return BalanceSheet(source_obj.sku_info.get('penalty', 0), 0)

        self._update_open_orders(source_obj.id, control.consumer_product_id, control.consumer_quantity)
        sku_price = self.facility.get_selling_price()
        # simple demand forecast algorithm, AVG(latest sales in 14 days)
        # TODO: More advanced forecast algorithm
        sale_mean = max(1.0, self.facility.get_sale_mean())
        order = DistributionUnit.Order(self.facility, control.consumer_product_id, control.consumer_quantity, control.consumer_vlt)
        # Get inventory in stock and transit
        in_transit_orders = sum(self.open_orders.values(), Counter())
        sku_in_stock_and_transit = (self.facility.storage.stock_levels[control.consumer_product_id]
                                     + in_transit_orders[control.consumer_product_id])
        if self.facility.distribution is not None:
            inventory_to_be_distributed = self.facility.distribution.get_pending_order()[control.consumer_product_id]
            sku_in_stock_and_transit -= inventory_to_be_distributed
        
        # holding_cost = self.facility.storage.economy.unit_storage_cost
        # expect_holding_cost = (sku_in_stock_and_transit + control.consumer_quantity) * holding_cost / 2
        # place the order
        order_product_cost = source_obj.distribution.place_order( order )
        
        self.latest_consumptions[-1] = 1.0 # order.quantity
        if order.vlt < len(self.pending_order_daily):
            self.pending_order_daily[order.vlt - 1] += order.quantity

        self.economy.total_units_purchased += control.consumer_quantity
        order_cost = self.facility.facility.facility_info['order_cost']
        
        # for SKU in warehouse, take 10% of its cost as profit
        # is_retailer = isinstance(self.facility, SKUStoreUnit)
        order_profit = sku_price*order.quantity
        balance =  BalanceSheet(0, -order_cost-order_product_cost)
        # replenishment_discount = supply_network_config.env_config['replenishment_discount']
        # base_stock = self.facility.base_stock
        # if base_stock is None:
        #     base_stock = control.consumer_vlt * sale_mean
        # Expect days to clear existing stock
        # expect_sale_days = (sku_in_stock_and_transit + control.consumer_quantity) / sale_mean
        # base_sale_days = base_stock / sale_mean
        # let sale_day_diff be bounded to avoid overflow
        # if expect_sale_days <= base_sale_days:
        #     sale_day_diff = 0
        # else:
        #     sale_day_diff = min(15, int(expect_sale_days - base_sale_days))
        # reward_discount = math.pow(replenishment_discount, sale_day_diff)
        self.reward_discount = control.reward_discount
        reward = - order_cost - order_product_cost + self.reward_discount * order_profit
        return balance, reward

    
class SellerUnit(Agent):
    @dataclass
    class Economy:
        unit_price: int = 0
        total_units_sold: int = 0
        latest_sale: int = 0
            
        def market_demand(self, sale_gamma):
            return int(np.random.gamma(sale_gamma))
        
        def profit(self, units_sold, unit_price):
            return units_sold * unit_price
        
        def step_balance_sheet(self, units_sold, unit_price, out_stock_demand, backlog_ratio):
            # （销售利润，缺货损失）
            balance = BalanceSheet(self.profit(units_sold, unit_price), -self.profit(out_stock_demand, unit_price)*backlog_ratio)
            # balance = BalanceSheet(self.profit(units_sold, unit_price), 0)
            return balance
        
    @dataclass
    class Config:
        sale_gamma: int
    
    @dataclass
    class Control:
        unit_price: int
            
    def __init__(self, facility, config, economy):
        
        self.facility = facility
        self.economy = economy
        self.config = config
        hist_len = self.facility.world.supply_network_config.env_config['sale_hist_len']
        self.backlog_demand_hist = [0] * hist_len
        self.sale_hist = [self.config.sale_gamma] * hist_len
        self.total_backlog_demand = 0
        self.step_reward = 0
            
    def _update_sale_hist(self, sale):
        self.sale_hist.append(sale)
        self.sale_hist = self.sale_hist[1:]
    
    def _update_backlog_demand_hist(self, backlog_demand):
        self.backlog_demand_hist.append(backlog_demand)
        self.backlog_demand_hist = self.backlog_demand_hist[1:]
    
    def sale_mean(self):
        return np.mean(self.sale_hist)

    def sale_std(self):
        return np.std(self.sale_hist)
    
    def reset(self):
        self.economy.total_units_sold = 0
        self.total_backlog_demand = 0
        self.step_reward = 0

    def act(self, control):
        # update the current unit price
        sku_price =  self.facility.get_selling_price()
        # sku_cost =  self.facility.sku_info['cost']
        self.economy.unit_price = sku_price
        product_id = self.facility.bom.output_product_id
        demand = self.economy.market_demand(self.facility.sku_info['sale_gamma'])
        self.economy.latest_sale = demand
        self._update_sale_hist(demand)
        sold_qty = self.facility.storage.take_available(product_id, demand)
        self.economy.total_units_sold += sold_qty
        out_stock_demand = max(0, demand - sold_qty)
        self.total_backlog_demand += out_stock_demand
        self._update_backlog_demand_hist(out_stock_demand)
        backlog_ratio = self.facility.sku_info.get('backlog_ratio', 0.3)
        balance = self.economy.step_balance_sheet( sold_qty, self.economy.unit_price, out_stock_demand, backlog_ratio )
        self.step_reward = balance.total()
        # reward = balance.loss
        return balance, self.step_reward

# ======= Base facility class (collection of units)

class ProductUnit(Agent):
    @dataclass
    class Config(ConsumerUnit.Config, 
                 ManufacturingUnit.Config, 
                 SellerUnit.Config):
        bill_of_materials: BillOfMaterials

    @dataclass
    class EconomyConfig:
        order_cost: int
        initial_balance: int
    
    @dataclass 
    class Economy:
        total_balance: BalanceSheet
        step_balance: BalanceSheet
        def deposit(self, balance_sheets):
            self.step_balance = sum(balance_sheets)
            self.total_balance += self.step_balance
            return self.step_balance
    
    @dataclass
    class Control(ConsumerUnit.Control,
                  ManufacturingUnit.Control, 
                  SellerUnit.Control):
        pass

    def __init__(self, facility, config, economy_config):
        self.initial_balance = economy_config.initial_balance  
        self.economy = ProductUnit.Economy(BalanceSheet(self.initial_balance, 0), BalanceSheet(self.initial_balance, 0))
        self.facility = facility
        self.world = facility.world
        self.id_num = facility.world.generate_id()
        self.id = f"{self.__class__.__name__}_{self.id_num}"
        self.idx_in_config = None
        self.distribution = None
        self.storage = None
        self.consumer = None
        self.sku_info = None
        self.manufacturing = None
        self.seller = None
        self.bom = config.bill_of_materials
        self.downstream_skus = []
        self.unit_price = 0
        self.step_reward = 0
        self.base_stock = None
        self.constraint_automaton = None
        self.constraint_automaton_state = None
        self.constraint_atoms = None
        self.constraint_idx = [0] * len(constraints)
        self.is_accepted = True
        self.is_terminal = False
        self.constraint_state_hist = np.zeros(self.facility.world.supply_network_config.env_config['constraint_state_hist_len'])
        self.atom_valid_hist = {}
        for atom_name in atoms.keys():
            self.atom_valid_hist[atom_name] = np.zeros(self.facility.world.supply_network_config.env_config['constraint_state_hist_len'])

    def reset_constraint_automaton(self, f_state):
        def _get_random_precessor(dfa, state):
            return dfa.initial_state
            # return np.random.choice(list(dfa.states))
            # precessors = []
            # for t in dfa.get_transitions():
            #     if t[-1] == state:
            #         precessors.append(t[0])
            # return np.random.choice(precessors)
        self.is_accepted = True
        self.is_terminal = False
        for atom_name in atoms.keys():
            self.atom_valid_hist[atom_name] = np.zeros(self.facility.world.supply_network_config.env_config['constraint_state_hist_len'])

        if self.constraint_automaton is not None:
            self.constraint_automaton_state = _get_random_precessor(self.constraint_automaton, self.constraint_automaton_state)
            shift(self.constraint_state_hist, -1, cval=0)
            self.constraint_state_hist[-1] = self.constraint_automaton_state
            # self.constraint_automaton_state = self.constraint_automaton.initial_state

    # construct dfa for a given constraint
    def construct_constraint_automaton(self):
        constraint = self.sku_info.get('constraint', None)
        if constraint is not None:
            # parser = LTLfParser()
            # formula = parser(constraint)
            # self.constraint_automaton = formula.to_automaton().determinize()
            self.constraint_automaton = constraint_automata[constraint]
            self.constraint_automaton_state = self.constraint_automaton.initial_state
            self.constraint_atoms = constraint_formulas[constraint].find_labels()
            self.constraint_idx[constraints.index(constraint)] = 1

    def update_atom_status(self, f_state):
        for atom_name in atoms.keys():
            if (self.constraint_automaton is not None) and (atom_name in self.constraint_atoms):
                status_hist = self.atom_valid_hist[atom_name]
                shift(status_hist, -1, cval=0)
                status_hist[-1] = (1 if atoms[atom_name](f_state) else -1)
                self.atom_valid_hist[atom_name] = status_hist

    # step forward constraint automaton using the lastest state info
    # return True if constraints are not voliated; False otherwise
    def constraint_automaton_step(self, f_state):
        def _is_automaton_state_rejected(dfa, state):
            if dfa.is_accepting(state):
                return False
            transitions = dfa.get_transitions_from(state)
            for t in transitions:
                if t[-1] != state:
                    return False
            return True
        def _is_automaton_state_terminal(dfa, state):
            if not dfa.is_accepting(state):
                return False
            transitions = dfa.get_transitions_from(state)
            for t in transitions:
                if t[-1] != state:
                    return False
            return True
        if self.constraint_automaton is not None:
            if not self.is_accepted:
                self.reset_constraint_automaton(f_state)
            else:
                if not self.is_terminal:
                    automaton_label = dict()
                    for atom in self.constraint_atoms:
                        automaton_label[atom] = atoms[atom](f_state)
                    self.constraint_automaton_state = self.constraint_automaton.get_successor(self.constraint_automaton_state, automaton_label)
                    if _is_automaton_state_rejected(self.constraint_automaton, self.constraint_automaton_state):
                        self.is_accepted = False
                    if _is_automaton_state_terminal(self.constraint_automaton, self.constraint_automaton_state):
                        self.is_terminal = True
        shift(self.constraint_state_hist, -1, cval=0)
        self.constraint_state_hist[-1] = self.constraint_automaton_state
        return self.is_accepted

    # get base stock, used for guided reward calculation in ConsumerUnit
    def get_base_stock(self):
        from scheduler.base_stock_policy_utils import BaseStockPolityUtility
        hist_len = self.world.supply_network_config.env_config['sale_hist_len']
        base_stock_list = BaseStockPolityUtility.get_base_stock([self], hist_len)
        return base_stock_list[self.id][hist_len // 2]

    def get_reward_normalizer(self):
        if self.seller is not None:
            return self.sku_info['price'] * self.sku_info['sale_gamma'] * 4
        else:
            return self.sku_info['price'] * 100
    
    def update_base_stock(self):
        # update base stock randomly with probability 0.1
        if np.random.random() <= 0.1:
            self.base_stock = self.get_base_stock()

    def get_latest_sale(self):
        sale = 0
        for d_sku in self.downstream_skus:
            sale += d_sku.get_latest_sale()
        return sale

    def get_sale_mean(self):
        sale_mean = 0
        for d_sku in self.downstream_skus:
            sale_mean += d_sku.get_sale_mean()
        return sale_mean

    def get_sale_std(self):
        sale_std = 0
        for d_sku in self.downstream_skus:
            sale_std += d_sku.get_sale_std()
        return sale_std / np.sqrt(min(1, len(self.downstream_skus)))

    def get_max_vlt(self):
        vlt = 1
        if self.consumer is not None:
            for source in self.consumer.sources:
                if source.sku_info.get('vlt', 1) > vlt:
                    vlt = source.sku_info.get('vlt', 1)
        return vlt
    
    # current inventory quantity in stock
    def get_inventory_in_stock(self):
        product_id = self.sku_info['sku_name']
        return self.storage.stock_levels[product_id]

    # get (max)-selling price in retailers
    def get_selling_price(self):
        price = 0.0
        for sku in self.downstream_skus:
            price = max(price, sku.get_selling_price())
        return price

    def get_seller_step_reward(self):
        reward = 0.0
        for sku in self.downstream_skus:
            reward += sku.get_seller_step_reward()
        return reward
    
    def reset(self):
        self.construct_constraint_automaton()
        self.is_accepted = True
        units = filter(None, [self.consumer, self.manufacturing, self.seller])
        for unit in units:
            unit.reset()
        self.economy = ProductUnit.Economy(BalanceSheet(self.initial_balance, 0), BalanceSheet(self.initial_balance, 0))

    def act(self, control): 
        # self.update_base_stock()
        units = filter(None, [self.consumer, self.manufacturing, self.seller])
        balance_rewards = [ u.act(control) for u in  units ]
        balance_sheets = [b[0] for b in balance_rewards]
        rewards = [r[1] for r in balance_rewards]
        product_id = self.sku_info['sku_name']
        if self.storage is not None:
            storage_reward = -self.storage.stock_levels[product_id]*self.storage.economy.unit_storage_cost
        else:
            storage_reward = 0
        storage_balance = BalanceSheet(0, storage_reward)
        if self.distribution is not None:
            checkin_order = self.distribution.checkin_order[product_id]
            transportation_cost = self.distribution.transportation_cost[product_id]
            delay_order_penalty = self.distribution.delay_order_penalty[product_id]
            distribution_cost = -(transportation_cost+delay_order_penalty)
            self.distribution.checkin_order[product_id] = 0
            self.distribution.transportation_cost[product_id] = 0
            self.distribution.delay_order_penalty[product_id] = 0
            distribution_balance = BalanceSheet(checkin_order*self.sku_info['price'], distribution_cost)
            # using selling price to calculate reward of a distributor
            distribution_reward = distribution_balance.total()
        else:
            distribution_reward = 0
            distribution_balance = BalanceSheet()
        # pop up rewards of downstreaming SKUs
        for sku in self.downstream_skus:
            balance_sheets.append(sku.economy.step_balance)
            rewards.append(sku.step_reward)
        
        self.economy.deposit(balance_sheets + [storage_balance, distribution_balance])
        self.step_reward = sum(rewards + [storage_reward, distribution_reward]) / self.get_reward_normalizer()
        # reward can be different from balance
        return self.economy.step_balance, self.step_reward


class FacilityCell(Cell, Agent):
    @dataclass
    class Config(StorageUnit.Config, 
                 DistributionUnit.Config):
        pass

    @dataclass 
    class Economy:
        step_balance: BalanceSheet
        total_balance: BalanceSheet
        def deposit(self, balance_sheets):
            self.step_balance = sum(balance_sheets)
            self.total_balance += self.step_balance
            return self.step_balance  
    
    @dataclass
    class Control(ConsumerUnit.Control,
                  DistributionUnit.Control, 
                  ManufacturingUnit.Control, 
                  SellerUnit.Control):
        pass
    
    def __init__(self, x, y, world, config, economy_config):  
        super(FacilityCell, self).__init__(x, y)
        self.id_num = world.generate_id()
        self.idx_in_config = None
        self.id = f"{self.__class__.__name__}_{self.id_num}"
        self.world = world
        self.initial_balance = economy_config.initial_balance
        self.economy = FacilityCell.Economy(BalanceSheet(self.initial_balance, 0), BalanceSheet(self.initial_balance, 0))
        self.storage = None
        self.distribution = None
        self.consumer = None
        self.seller = None
        self.sku_in_stock = []
        self.facility_short_name = None # for rendering
        self.facility_info = None
        self.step_reward = 0
        self.is_reset = True
        self.storage_levels = Counter()
        self.in_transit_orders = Counter()
        self.constraint_idx = [0] * len(constraints)
    
    def get_in_transit_order(self):
        if self.is_reset:
            for sku in self.sku_in_stock:
                self.in_transit_orders[sku.sku_info['sku_name']] = sku.consumer.get_in_transit_quantity()
            self.is_reset = False
        return self.in_transit_orders

    def reset(self):
        self.is_reset = True
        units = filter(None, [self.storage, self.distribution])
        for unit in units:
            unit.reset()
        self.economy = FacilityCell.Economy(BalanceSheet(self.initial_balance, 0), BalanceSheet(self.initial_balance, 0))
        # sample inventory in stock randomly
        for sku in self.sku_in_stock:
            sku.reset()
            init_stock = np.random.randint(sku.sku_info.get('init_stock', 100))
            self.storage.try_add_units({sku.sku_info['sku_name']: init_stock})

    def act(self, control): 
        self.is_reset = True
        units = filter(None, [self.storage, self.distribution])
        _ = [ u.act(control) for u in  units ]
        balance_rewards =  [(s.economy.step_balance, s.step_reward) for s in self.sku_in_stock]
        balance_sheets = [b[0] for b in balance_rewards]
        rewards = [r[1] for r in balance_rewards]
        self.economy.deposit(balance_sheets)
        self.step_reward = sum(rewards)
        return self.economy.step_balance, self.step_reward


# ======= Concrete facility classes
def create_distribution_unit(facility, config):
    return DistributionUnit(facility, 
                            config.fleet_size, 
                            DistributionUnit.Economy(), 
                            Transport.Economy(config.unit_transport_cost))
                    

class SKUSupplierUnit(ProductUnit):
    def __init__(self, facility, config, economy_config):
        super(SKUSupplierUnit, self).__init__(facility, config, economy_config)
        self.manufacturing = ManufacturingUnit(self, ManufacturingUnit.Economy(config.unit_manufacturing_cost))
        self.consumer = ConsumerUnit(self, config.sources, ConsumerUnit.Economy(economy_config.order_cost))

class SKUWarehouseUnit(ProductUnit):
    def __init__(self, facility, config, economy_config):
        super(SKUWarehouseUnit, self).__init__(facility, config, economy_config)
        self.consumer = ConsumerUnit(self, config.sources, ConsumerUnit.Economy(economy_config.order_cost))

class SKUStoreUnit(ProductUnit):
    def __init__(self, facility, config, economy_config):
        super(SKUStoreUnit, self).__init__(facility, config, economy_config)
        self.consumer = ConsumerUnit(self, config.sources, ConsumerUnit.Economy(economy_config.order_cost))
        self.seller = SellerUnit(self, SellerUnit.Config(sale_gamma=config.sale_gamma), SellerUnit.Economy())

    @override(ProductUnit)
    def get_latest_sale(self):
        if len(self.seller.sale_hist) > 0:
            return self.seller.sale_hist[-1]
        else:
            return 0

    @override(ProductUnit)
    def get_sale_mean(self):
        # return self.sku_info['sale_gamma']
        return self.seller.sale_mean()
    
    @override(ProductUnit)
    def get_sale_std(self):
        return self.seller.sale_std()

    @override(ProductUnit)
    def get_selling_price(self):
        return self.sku_info['price']
    
    @override(ProductUnit)
    def get_seller_step_reward(self):
        return self.seller.step_reward

    def get_eoq_quantity(self):
        holding_cost = self.storage.economy.unit_storage_cost
        order_cost = self.consumer.economy.order_cost
        return math.sqrt(2*self.get_sale_mean()*order_cost/holding_cost)

    def get_ending_reward(self):
        stock = self.get_inventory_in_stock()
        demand = self.get_eoq_quantity()
        sold_qty = min(demand, stock)
        out_stock_demand = max(0.0, demand-sold_qty)
        backlog_ratio = self.sku_info.get('backlog_ratio', 0.3)
        balance = self.seller.economy.step_balance_sheet( sold_qty, self.seller.economy.unit_price, out_stock_demand, backlog_ratio )
        return balance.total() / self.get_reward_normalizer()

class SupplierCell(FacilityCell):
    def __init__(self, x, y, world, config, economy_config):
        super(SupplierCell, self).__init__(x, y, world, config, economy_config)
        self.storage = StorageUnit(config.max_storage_capacity, StorageUnit.Economy(config.unit_storage_cost))
        self.distribution = create_distribution_unit(self, config)


class WarehouseCell(FacilityCell):
    def __init__(self, x, y, world, config, economy_config):
        super(WarehouseCell, self).__init__(x, y, world, config, economy_config) 
        self.storage = StorageUnit(config.max_storage_capacity, StorageUnit.Economy(config.unit_storage_cost))
        self.distribution = create_distribution_unit(self, config)
        # indicate echelon level of warehouse
        # the greater of the number, the closer the warehouse to the retailers
        self.echelon_level = 0
        
class RetailerCell(FacilityCell):
    def __init__(self, x, y, world, config, economy_config):
        super(RetailerCell, self).__init__(x, y, world, config, economy_config) 
        self.storage = StorageUnit(config.max_storage_capacity, StorageUnit.Economy(config.unit_storage_cost))


# ======= The world

class World:
    @dataclass
    class Economy:
        def __init__(self, world):
            self.world = world
            
        def global_balance(self) -> BalanceSheet: 
            total_balance = BalanceSheet()
            for facility in self.world.facilities:
                if isinstance(facility, FacilityCell):
                    total_balance += facility.economy.total_balance
            return total_balance

        def global_reward(self):
            total_reward = 0
            for facility in self.world.facilities:
                if isinstance(facility, FacilityCell):
                    total_reward += facility.step_reward
            return total_reward

    
    @dataclass
    class Control:
        facility_controls: dict
    
    @dataclass
    class StepOutcome:
        facility_step_balance_sheets: dict
    
    def __init__(self, x, y, config):
        self.size_x = x
        self.size_y = y
        self.grid = None
        self.economy = World.Economy(self)
        self.facilities = dict()
        self.id_counter = 0
        self.time_step = 0

        # store max price/cost appearing in the world, mainly for normalization
        self.max_price = 0

        # totlal echelons of the whole supply chain
        # including suppliers and stores
        self.total_echelon = 0
        # map each facility/product id to echelon level
        self.agent_echelon = dict()
        self.supply_network_config = config
        
    def reset(self):
        self.time_step = 0
        for facility in self.facilities.values():
            if isinstance(facility, FacilityCell):
                facility.reset()

    def generate_id(self):
        self.id_counter += 1
        return self.id_counter
        
    def act(self, control):
        rewards = dict()
        balances = dict()
        facilities = list(self.facilities.values())
        # facitilty acting according to order: for retailers up to suppliers, echelon by echelon
        for f_type in [ProductUnit, FacilityCell]:
            for echelon in range(self.total_echelon-1, -1, -1):
                for facility in facilities:
                    if isinstance(facility, f_type) and self.agent_echelon[facility.id] == echelon:
                         balances[facility.id], rewards[facility.id] = facility.act( control.facility_controls.get(facility.id) )
        self.time_step += 1
        return World.StepOutcome(balances), World.StepOutcome(rewards)
    
    def create_cell(self, x, y, clazz):
        self.grid[x][y] = clazz(x, y)

    def place_cell(self, *cells):
        for c in cells:
            self.grid[c.x][c.y] = c
    
    def is_railroad(self, x, y):
        return isinstance(self.grid[x][y], RailroadCell)
    
    def is_traversable(self, x, y):
        return not isinstance(self.grid[x][y], TerrainCell)
    
    def c_tostring(x,y):
        return np.array([x,y]).tostring()
                
    def map_to_graph(self):
        g = nx.Graph()
        for x in range(1, self.size_x-1):
            for y in range(1, self.size_y-1):
                for c in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                    if self.is_traversable(x, y) and self.is_traversable(c[0], c[1]):
                        g.add_edge(World.c_tostring(x, y), World.c_tostring(c[0], c[1]))
        return g
    
    @lru_cache(maxsize = 32)  # speedup the simulation
    def find_path(self, x1, y1, x2, y2):
        g = self.map_to_graph()
        path = nx.astar_path(g, source=World.c_tostring(x1, y1), target=World.c_tostring(x2, y2))
        path_np = [np.fromstring(p, dtype=int) for p in path]
        return [(p[0], p[1]) for p in path_np]
    
    def get_facilities(self, clazz):
        return filter(lambda f: isinstance(f, clazz), self.facilities.values())
