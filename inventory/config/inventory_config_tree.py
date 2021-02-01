sku_config = {"sku_num": 2, "sku_names": ["SKU0", "SKU1"]}
store_config = {"name": "STORE", "short_name": "S", "store_num": 1, "storage_capacity": [110000], "unit_storage_cost": [1], "order_cost": [500], "sku_relation": [[{"sku_name": "SKU0", "price": 142, "service_level": 0.95, "cost": 77, "sale_gamma": 46, "init_stock": 0, "max_stock": 1000, "constraint": "G(is_replenish_constraint -> ((X!is_replenish_constraint)&(XX!is_replenish_constraint)))"}, {"sku_name": "SKU1", "price": 494, "service_level": 0.95, "cost": 256, "sale_gamma": 9, "init_stock": 0, "max_stock": 1000}]]}
warehouse_config = [{"name": "WAREHOUSE", "short_name": "R", "warehouse_num": 1, "fleet_size": [20], "unit_storage_cost": [1], "unit_transport_cost": [1], "storage_capacity": [5500], "order_cost": [500], "delay_order_penalty": [1000], "downstream_facilities": [[0]], "sku_relation": [[{"sku_name": "SKU0", "service_level": 0.96, "vlt": 3, "price": 77, "cost": 77, "init_stock": 920}, {"sku_name": "SKU1", "service_level": 0.96, "vlt": 3, "price": 256, "cost": 256, "init_stock": 180}]]}]
supplier_config = {"name": "SUPPLIER", "short_name": "M", "supplier_num": 1, "fleet_size": [20], "unit_storage_cost": [1], "unit_transport_cost": [1], "storage_capacity": [5500], "order_cost": [200], "delay_order_penalty": [1000], "downstream_facilities": [[0]], "sku_relation": [[{"sku_name": "SKU0", "price": 77, "cost": 69, "service_level": 0.95, "vlt": 3, "init_stock": 2300, "production_rate": 2300}, {"sku_name": "SKU1", "price": 256, "cost": 230, "service_level": 0.95, "vlt": 3, "init_stock": 450, "production_rate": 450}]]}
env_config = {"global_reward_weight_producer": 0.5, "global_reward_weight_consumer": 0.5, "downsampling_rate": 1, "episod_duration": 28, "initial_balance": 100000, "consumption_hist_len": 8, "sale_hist_len": 8, "pending_order_len": 8, "constraint_state_hist_len": 5, "total_echelons": 3, "replenishment_discount": 0.9, "reward_normalization": 10000000.0, "constraint_violate_reward": -1000000.0, "gamma": 0.99, "tail_timesteps": 7, "heading_timesteps": 7}
demand_sampler = "DYNAMIC_GAMMA"
