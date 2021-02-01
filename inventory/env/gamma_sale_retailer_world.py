from datetime import datetime, timedelta
from data_adapter.gamma_sale_adapter import GammaSaleAdapter, NoisyGammaSaleAdapter


def load_sale_sampler(store_idx, supply_network_config):
    config = dict()
    config['dt_col'] = 'DT'
    config['dt_format'] = '%Y-%m-%d'
    config['sale_col'] = 'Sales'
    config['id_col'] = 'SKU'
    config['total_span'] = 365*2
    config['store_idx'] = store_idx
    config['sale_price_col'] = 'Price'
    config['start_dt'] = datetime.today().date()
    config['supply_network_config'] = supply_network_config
    sale_sampler = GammaSaleAdapter(config)
    return sale_sampler


def load_noisy_sale_sampler(store_idx, supply_network_config):
    config = dict()
    config['dt_col'] = 'DT'
    config['dt_format'] = '%Y-%m-%d'
    config['sale_col'] = 'Sales'
    config['id_col'] = 'SKU'
    config['total_span'] = 365*2
    config['store_idx'] = store_idx
    config['sale_price_col'] = 'Price'
    config['start_dt'] = datetime.today().date()
    config['supply_network_config'] = supply_network_config
    sale_sampler = NoisyGammaSaleAdapter(config)
    return sale_sampler