import pandas as pd
from datetime import datetime, timedelta


class BaseSaleAdapter(object):

    def __init__(self, config):
        self.config = config
        self.dt_col = self.config['dt_col']
        self.dt_format = self.config['dt_format']
        self.sale_col = self.config['sale_col']
        self.id_col = self.config['id_col']
        self.sale_price_col = self.config['sale_price_col']
        self.file_format = self.config.get('file_format', 'CSV')
        self.encoding = self.config.get('encoding', 'utf-8')
        self.file_loc = self.config['file_loc']
        self.sale_ts_cache = dict()
        self.sale_price_ts_cache = dict()
        self.total_span = 0
        self.cache_data()

    def _transfer_to_daily_sale(self):
        pass

    def _transfer_to_original_sale(self):
        pass

    # forecast results based on historical data
    def get_forecast(self, id_val):
        pass

    def sample_sale_and_price(self, id_val, gap):
        return (self.sale_ts_cache[id_val][gap], self.sale_price_ts_cache[id_val][gap])

    def cache_data(self):
        self.df = self._read_df()
        self._transfer_to_daily_sale()
        id_list = self.df[self.id_col].unique().tolist()
        dt_min, dt_max = self.df[self.dt_col].min(), self.df[self.dt_col].max()
        self.total_span = (dt_max - dt_min).days + 1
        for id_val in id_list:
            df_tmp = self.df[self.df[self.id_col] == id_val]
            df_tmp[f"{self.dt_col}_str"] = df_tmp[self.dt_col].map(lambda x: x.strftime(self.dt_format))
            sale_cache_tmp = df_tmp.set_index(f"{self.dt_col}_str").to_dict('dict')[self.sale_col]
            sale_price_cache_tmp = df_tmp.set_index(f"{self.dt_col}_str").to_dict('dict')[self.sale_price_col]
            dt_tmp = dt_min
            self.sale_ts_cache[id_val] = []
            self.sale_price_ts_cache[id_val] = []
            sale_price_mean = df_tmp[self.sale_price_col].mean()
            while dt_tmp <= dt_max:
                dt_tmp_str = datetime.strftime(dt_tmp, self.dt_format)
                self.sale_ts_cache[id_val].append(sale_cache_tmp.get(dt_tmp_str, 0))
                self.sale_price_ts_cache[id_val].append(sale_price_cache_tmp.get(dt_tmp_str, sale_price_mean))
                dt_tmp = dt_tmp + timedelta(days=1)

    def _read_df(self):
        if self.file_format == 'CSV':
            self.df = pd.read_csv(self.file_loc, encoding=self.encoding, parse_dates=[self.dt_col])
        elif self.file_format == 'EXCEL':
            self.df = pd.read_excel(self.file_loc, encoding=self.encoding, parse_dates=[self.dt_col])
        else:
            raise BaseException('Not Implemented')
        return self.df