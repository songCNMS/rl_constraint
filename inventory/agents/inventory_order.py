from dataclasses import dataclass
from agents.inventory import SellerUnit, BalanceSheet, SKUStoreUnit, ConsumerUnit
        
    
class OuterSellerUnit(SellerUnit):
    @dataclass
    class Economy:
        unit_price: int = 0
        total_units_sold: int = 0
        latest_sale: int = 0
            
        def market_demand(self, sale_sampler, product_id, step):
            return sale_sampler.sample_sale_and_price(product_id, step)
        
        def profit(self, units_sold, unit_price):
            return units_sold * unit_price
        
        def step_balance_sheet(self, units_sold, unit_price, out_stock_demand, backlog_ratio):
            # （销售利润，缺货损失）
            balance = BalanceSheet(self.profit(units_sold, unit_price), -self.profit(out_stock_demand, unit_price)*backlog_ratio)
            return balance

    def __init__(self, facility, config, economy, sale_sampler):
        super(OuterSellerUnit, self).__init__(facility, config, economy)
        self.sale_sampler = sale_sampler
        self.step = 0
    
    def act(self, control):
        # update the current unit price
        sku_price =  self.facility.sku_info['price']
        sku_cost =  self.facility.sku_info['cost']
        self.economy.unit_price = sku_price
        self.step = (self.step + 1) % self.sale_sampler.total_span
        product_id = self.facility.bom.output_product_id
        demand, _ = self.economy.market_demand(self.sale_sampler, product_id, self.step)
        self.economy.latest_sale = demand
        self._update_sale_hist(demand)
        sold_qty = self.facility.storage.take_available(product_id, demand)
        self.economy.total_units_sold += sold_qty
        out_stock_demand = max(0, demand - sold_qty)
        self.total_backlog_demand += out_stock_demand
        self._update_backlog_demand_hist(out_stock_demand)
        backlog_ratio = self.facility.sku_info.get('backlog_ratio', 0.1)
        balance = self.economy.step_balance_sheet( sold_qty, self.economy.unit_price, 0, backlog_ratio )
        reward = sold_qty*(sku_price-sku_cost) - (sku_price-sku_cost)*out_stock_demand
        return balance, reward


class OuterSKUStoreUnit(SKUStoreUnit):
    def __init__(self, facility, config, economy_config, sale_sampler):
        super(OuterSKUStoreUnit, self).__init__(facility, config, economy_config)
        self.consumer = ConsumerUnit(self, config.sources, ConsumerUnit.Economy(economy_config.order_cost))
        self.seller = OuterSellerUnit(self, OuterSellerUnit.Config(sale_gamma=config.sale_gamma), OuterSellerUnit.Economy(), sale_sampler)