import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from random import randint
import matplotlib.pyplot as plt
import logging, re, json, copy, string
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import tqdm


class MetaData(type):
    def __new__(cls, name, bases, namespace, **kwds):
        namespace['DAYS_IN_YEAR'] = 365
        namespace['HOURS_IN_DAY'] = 24
        namespace['MINUTES_IN_HOUR'] = 60

        logger_name = f'{name}_logger'
        logger = logging.getLogger(logger_name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            '%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        namespace['logger'] = logger

        new_cls = super().__new__(cls, name, bases, namespace)

        original_init = new_cls.__init__

        def new_init(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            original_init(self, *args, **kwargs)

        new_cls.__init__ = new_init
        return new_cls


class SpotMarketData(metaclass=MetaData):
    logger: logging.Logger
    start_month: str = '2022-10'
    end_month: str = '2023-12'
    max_concur: int = 8
    data_root: Path = Path('../spot/monthly/klines')
    universe: list = ['BTCUSDT', 'ETHUSDT']
    time_to_use: str = 'open_time'
    time_unit: str = 'ms'
    headers: list = [
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_volume",
                        "n_trades",
                        "taker_buy_base_volume",
                        "taker_buy_quote_volume",
                        "ignore",
                    ]

    def __repr__(self) -> str:
        universe = ', '.join(self.universe)
        return f'From: {self.start_month} to: {self.end_month}  time_to_use: {self.time_to_use}   {universe}'

    def load_one_csv(self, filename: Path, headers: list, time_to_use: str, time_unit: int) -> pd.DataFrame:
        df = pd.read_csv(filename, header=None, names=headers)
        df.sort_values(time_to_use, inplace=True)
        df.drop_duplicates(subset=[time_to_use], keep="first", inplace=True)
        df["datetime"] = pd.to_datetime(df[time_to_use], unit=time_unit).dt.tz_localize(None)
        df.drop(labels=["open_time", "close_time", "ignore"], axis=1, inplace=True)
        df.set_index("datetime", inplace=True)
        return df.sort_index()


    def load_csv(self):
        with ThreadPoolExecutor(max_workers=self.max_concur) as executor:
            to_do_map = defaultdict(tuple)
            for inst in self.universe:
                inst_dir = self.data_root / inst / '1m' 
                for f in inst_dir.iterdir():
                    pattern = f'{inst}' + '-1m-' + '(\d{4}-\d{2})' + '.csv'
                    m = re.search(pattern, f.name)
                    if not m:
                        self.logger.critical(f'File {f} does not exist!')

                    year_mon = m.group(1)
                    if self.start_month <= year_mon <= self.end_month:
                        future = executor.submit(self.load_one_csv, 
                                                 f,
                                                 self.headers,
                                                 self.time_to_use,
                                                 self.time_unit)
                        to_do_map[future] = (inst, year_mon)
            done_iter = tqdm.tqdm(as_completed(to_do_map), total=len(to_do_map))
            for future in done_iter:
                try:
                    result = future.result()
                except FileNotFoundError:
                    self.logger.error(f'{f} not found.')
                except pd.errors.EmptyDataError:
                    self.logger.error(f'{f} has no data.')
                except pd.errors.ParserError:
                    self.logger.error(f'{f} has an error format')
                except Exception as e:
                    self.logger.error(f'{f} unexpected error.')
                else:
                    inst, year_mon = to_do_map[future]
                    self.raw_data[inst][year_mon] = result

        self.combine_data()

    def combine_data(self):
        for inst in self.raw_data:
            if len(self.raw_data[inst]) > 0:
                data = [x.reset_index() for x in self.raw_data[inst].values()]
                df = pd.concat(data, axis=0, ignore_index=True)
                df.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
                df.set_index('datetime', inplace=True)
                self.combined_data[inst] = df.sort_index()

    def check_raw(self):
        inst_list = ', '.join(list(self.raw_data.keys()))
        self.logger.info(f'Universe: {inst_list}')
        inst_list, start_mon, end_mon, n_months = [], [], [], []
        start_time, end_time = [], []
        for inst in sorted(self.raw_data):
            inst_list += inst,
            months = sorted(self.raw_data[inst])
            start_mon += months[0],
            end_mon += months[-1],
            n_months += len(months),
            start_time += self.combined_data[inst].index[0],
            end_time += self.combined_data[inst].index[-1],

        self.df_raw_data_stat = pd.DataFrame({
            'Inst': inst_list,
            'Start': start_mon,
            'End': end_mon,
            'Length': n_months,
            'sTime': start_time,
            'eTime': end_time,
        })
        self.df_raw_data_stat.set_index('Inst', inplace=True)
        print(self.df_raw_data_stat)

    def transform_data(self):
        self.logger.info(f"Transforming data frames ...")
        symbols = sorted(self.combined_data)
        columns = set(self.combined_data[symbols[0]].columns.to_list())
        for sym in symbols[1:]:
            columns &= set(self.combined_data[sym].columns.to_list())
        columns = sorted(columns)
        level_1 = []
        for col in columns:
            tmp_res = {}
            for sym in symbols:
                tmp_res[sym] = self.combined_data[sym][col]
            level_1.append(pd.DataFrame(tmp_res))
        return pd.concat(level_1, axis=1, keys=columns)

    def resample_all(self,
                     freq: str = '10T',
                     prices: list = [
                        'open',
                        'high',
                        'low',
                        'close'],
                     volumes: list = [
                        'volume',
                        'quote_volume',
                        'taker_buy_base_volume',
                        'taker_buy_quote_volume'],
                     trades: list = [
                         'n_trades']):
        self.logger.info(f'Resampling raw data ...')
        self.resampled_data = defaultdict(pd.DataFrame)
        with ThreadPoolExecutor(max_workers=self.max_concur) as executor:
            to_do_map = defaultdict(str)
            for inst in self.combined_data:
                future = executor.submit(self.resample_to,
                                            inst,
                                            freq,
                                            prices,
                                            volumes,
                                            trades)
                to_do_map[future] = inst

            done_iter = tqdm.tqdm(as_completed(to_do_map), total=len(to_do_map))
            for future in done_iter:
                try:
                    result = future.result()
                except Exception as e:
                    self.logger.error(f'Resamping unexpected error.')
                else:
                    inst = to_do_map[future]
                    self.resampled_data[inst] = result
                    self.logger.debug(f'Done resampling calculation for {inst}.')


    def resample_to(self, instrument: str, freq: str, prices: list, volumes: list, trades: list):
        df = self.combined_data[instrument]
        df['tr'] = pd.DataFrame({
                                 'cl-lo': df['close'].shift() - df['low'],
                                 'hi-lo': df['high'] - df['close'].shift(),
                                 'hi-lo': df['high'] - df['low']
                                }).bfill().max(axis=1)
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        fields = defaultdict(pd.Series)
        fields['open']   = df['open'].resample(freq, closed='left', label='left').first()
        fields['close']  = df['close'].resample(freq, closed='left', label='left').last()
        fields['high']   = df['high'].resample(freq, closed='left', label='left').max()
        fields['low']    = df['low'].resample(freq, closed='left', label='left').min()
        fields['ret']    = np.log(fields['close']) - np.log(fields['open'])

        for vol in volumes:
            fields[vol] = df[vol].resample(freq, closed='left', label='left').sum()

        for trd in trades:
            fields[trd] = df[trd].resample(freq, closed='left', label='left').sum()

        for price in prices + ['tr']:
            fields[f'{price}_std']    = df[price].resample(freq, closed='left', label='left').std()
            fields[f'{price}_mean']   = df[price].resample(freq, closed='left', label='left').mean()
            fields[f'{price}_median'] = df[price].resample(freq, closed='left', label='left').median()
            fields[f'{price}_skew']   = df[price].resample(freq, closed='left', label='left').apply(lambda x: x.skew())

            for quantile in quantiles:
                fields[f'{price}_q{int(quantile*100):d}%'] = df[price].resample(freq, closed='left', label='left').quantile(quantile)
        
        return pd.DataFrame(fields)

    def window_calc(self, instrument: str, month: str, windows: list, prices: list, volumes: list, trades: list, quantiles: list):
        df = self.raw_data[instrument][month]
        df['tr'] = pd.DataFrame({
                                 'cl-lo': df['close'].shift() - df['low'],
                                 'hi-lo': df['high'] - df['close'].shift(),
                                 'hi-lo': df['high'] - df['low']
                                }).ffill().bfill().max(axis=1)

        df['buy_ratio'] = df['taker_buy_quote_volume'].mul(df['quote_volume'])
        df['volume_per_trade'] = df['quote_volume'].div(df['n_trades'])

        df['high-close'] = (df['high'] - df['close']).div(df['close'])
        df['close-low']  = (df['close'] - df['low']).div(df['close'])

        target = 30 * 4

        fields = defaultdict(pd.Series)

        for prc in ['open', 'close']:
            fields[f'{prc}_ret'] = np.log(df[prc]).diff()

        for lb in [15, 30, 60, 60 * 2, 60 * 4, 60 * 24]:
            quote_vol_fld = f'hist_{lb}_quote_vol'
            buy_vol_fld = f'hist_{lb}_buy_vol'
            buy_ratio_fld = f'hist_{lb}_buy_ratio'
            vol_per_trade_fld = f'hist_{lb}_vol_per_trade'
            trd_fld = f'hist_{lb}_trade'
            hc_fld = f'hist_{lb}_hc_median'
            cl_fld = f'hist_{lb}_cl_median'
            fields[quote_vol_fld] = df['quote_volume'].rolling(lb, min_periods=min(lb, 120)).sum()
            fields[buy_vol_fld] = df['taker_buy_quote_volume'].rolling(lb, min_periods=min(lb, 120)).sum()
            fields[trd_fld] = df['n_trades'].rolling(lb, min_periods=min(lb, 120)).sum()
            fields[buy_ratio_fld] = fields[buy_vol_fld].div(fields[quote_vol_fld])
            fields[vol_per_trade_fld] = fields[quote_vol_fld].div(fields[trd_fld])
            fields[hc_fld] = df['high-close'].rolling(lb, min_periods=min(lb, 120)).mean()
            fields[cl_fld] = df['close-low'].rolling(lb, min_periods=min(lb, 120)).mean()

            high_fld = f'hist_{lb}_high'
            low_fld  = f'hist_{lb}_low'
            mean_fld = f'hist_{lb}_mean'
            # fields[high_fld] = df['high'].rolling(lb, min_periods=min(lb, 120)).max()
            # fields[low_fld]  = df['low'].rolling(lb, min_periods=min(lb, 120)).min()
            # fields[mean_fld] = df['close'].rolling(lb, min_periods=min(lb, 120)).mean()
            fields[high_fld] = np.log(df['high']).diff(lb)
            fields[low_fld]  = np.log(df['low']).diff(lb)
            fields[mean_fld] = np.log(df['close']).diff(lb)

            h_c_fld = f'hist_{lb}_h_c'
            c_l_fld = f'hist_{lb}_c_l'
            fields[h_c_fld] = fields[high_fld] - fields[mean_fld]
            fields[c_l_fld] = fields[mean_fld] - fields[low_fld]

            for interval in range(0, 5):
                quantile = 0.1 + interval * 0.2
                fields[f'{quote_vol_fld}_{int(quantile*100):02d}%'] = fields[quote_vol_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[quote_vol_fld]) - 1.0
                fields[f'{buy_vol_fld}_{int(quantile*100):02d}%'] = fields[buy_vol_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[buy_vol_fld]) - 1.0
                fields[f'{trd_fld}_{int(quantile*100):02d}%'] = fields[trd_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[trd_fld]) - 1.0
                fields[f'{buy_ratio_fld}_{int(quantile*100):02d}%'] = fields[buy_ratio_fld].rolling(60 * 24, min_periods=120).quantile(quantile)
                fields[f'{vol_per_trade_fld}_{int(quantile*100):02d}%'] = fields[vol_per_trade_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[vol_per_trade_fld]) - 1.0
                fields[f'{hc_fld}_{int(quantile*100):02d}%'] = fields[hc_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[hc_fld]) - 1.0
                fields[f'{cl_fld}_{int(quantile*100):02d}%'] = fields[cl_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[cl_fld]) - 1.0

            for prc in ['open', 'close']:
                ret_fld = f'hist_{lb}_{prc}_ret'
                std_fld = f'hist_{lb}_{prc}_std'
                prc_fld = f'hist_{lb}_{prc}'
                fields[ret_fld] = np.log(df[prc]).diff(lb)
                fields[std_fld] = fields[f'{prc}_ret'].rolling(window=lb, min_periods=min(lb, 120)).std()
                for interval in range(0, 5):
                    quantile = 0.1 + interval * 0.2
                    fields[f'{ret_fld}_{int(quantile*100):02d}%'] = fields[ret_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[std_fld]) / lb
                    fields[f'{std_fld}_{int(quantile*100):02d}%'] = fields[std_fld].rolling(60 * 24, min_periods=120).quantile(quantile).div(fields[std_fld]) / lb
                    fields[f'{prc_fld}_{int(quantile*100):02d}%'] = df[prc].rolling(lb, min_periods=min(lb, 120)).quantile(quantile).div(df[prc]) - 1.0


        """
        for window in windows:
            for vol in volumes + trades + prices:
                _sum = df[vol].rolling(window=window).sum()
                _max = df[vol].rolling(window=window).max()
                _std = df[vol].rolling(window=window).std()
                _med = df[vol].rolling(window=window).median()
                _skw = df[vol].rolling(window=window).skew()
                fields[f'{vol}_{window}_sum'] = _sum
                fields[f'{vol}_{window}_std'] = _std
                fields[f'{vol}_{window}_med'] = _med
                fields[f'{vol}_{window}_skw'] = _skw
                fields[f'{vol}_{window}_max'] = _max * window / _sum


        ranks    = defaultdict(pd.Series)

        for col in fields:
            for window in windows:
                ranks[f'{col}_{window}_rank'] = fields[col].rolling(window=window).rank(ascending=False, pct=True)

        fields.update(ranks)

        for i in range(1, 10):
            fields[f'ret_{i}'] = fields['ret'].shift(-i)
        """

        for prc in ['open', 'close']:
            fields[f'{prc}_y'] = np.log(df[prc]).ffill().diff(target).shift(-target)

        return pd.DataFrame(fields).bfill().fillna(0)

    def window_all(self,
                   windows: list = [30, 60, 120],
                   prices: list = [
                       'open',
                       'high',
                       'low',
                       'close'],
                   volumes: list = [
                       'volume',
                       'quote_volume',
                       'taker_buy_base_volume',
                       'taker_buy_quote_volume'],
                   trades: list = [
                       'n_trades'],
                   quantiles: list = [0.05, 0.1, 0.5, 0.9, 0.95]):
        self.logger.info(f'Calculating window data ...')
        self.window_data = defaultdict(lambda: defaultdict(pd.DataFrame))
        with ThreadPoolExecutor(max_workers=self.max_concur) as executor:
            to_do_map = defaultdict(tuple)
            for inst in self.raw_data:
                for month in self.raw_data[inst]:
                    future = executor.submit(self.window_calc,
                                             inst,
                                             month,
                                             windows,
                                             prices,
                                             volumes,
                                             trades,
                                             quantiles)
                    to_do_map[future] = (inst, month)

            done_iter = tqdm.tqdm(as_completed(to_do_map), total=len(to_do_map))
            for future in done_iter:
                try:
                    result = future.result()
                except Exception as e:
                    self.logger.error(f'Windows calc unexpected error.')
                else:
                    inst, month = to_do_map[future]
                    self.window_data[inst][month] = result
                    self.logger.debug(f'Done window calculation for {inst} {month}.')


    def __init__(self):
        self.logger.info(f'Initiatlizing ...')

        self.raw_data = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.combined_data = defaultdict(pd.DataFrame)
        self.load_csv()

        self.transformed_data = self.transform_data()

    def dump_corr(self, path: str = './corr'):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for inst in self.window_data:
            for month in self.window_data[inst]:
                df = self.window_data[inst][month]
                filename = path / f'{inst}_{month}.csv'
                df.corr().round(4).sort_values('ret_1', ascending=True)[['ret'] + [f'ret_{i}' for i in range(1, 10)]].to_csv(filename, index=True)


if __name__ == '__main__':

    data = SpotMarketData()
    data.window_all()

    # features = defaultdict(lambda: defaultdict(list))

    # for sym in data.window_data:
    #     for month in sorted(data.window_data[sym]):
    #         print(f'{sym}  {month}')
    #         df = data.window_data[sym][month]
    #         # df_resampled = df.sample(len(df)//3, random_state=42)
    #         cr = df.corr()['close_y']
    #         fea = list(cr[cr.abs() > 0.03].index)
    #         fea.remove('open_y')
    #         fea.remove('close_y')
    #         for f in fea:
    #             features[sym][f] += month,


    # x = sorted([(len(v), i) for i, v in features['BTCUSDT'].items()])
    # y = sorted([(len(v), i) for i, v in features['ETHUSDT'].items()])

    cols = pd.read_csv('feature_cols.csv')['Features'].values.tolist()
    df = data.window_data['BTCUSDT']['2023-03'][cols + ['close_y']]