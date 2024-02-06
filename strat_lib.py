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
from data_lib import MetaData, SpotMarketData
from typing import Any, Optional

import warnings

warnings.filterwarnings('ignore')

class StrategyBase(metaclass=MetaData):
    logger: logging.Logger
    data: Any
    universe: list = ['BTCUSDT', 'ETHUSDT']

    def __init__(self, data: MetaData, universe: list = None, **kwargs) -> None:
        self.data = data
        if universe is not None:
            self.universe = universe

    def __repr__(self) -> str:
        universe = ', '.join(self.universe)
        return f'data type: {type(self.data)}  universe: {universe}'


class Strategy(StrategyBase):
    __trades: dict = None
    __signal: dict = None

    def __init__(self, data: MetaData, universe: list = None, **kwargs) -> None:
        super().__init__(data, universe)

    def __repr__(self) -> str:
        return super().__repr__()

    def strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def signal(self) -> dict[pd.DataFrame]:
        return self.__signal

    def simulate(self):
        pass


class TwoMovingAverage(Strategy):
    short: int
    long: int
    max_concur = 100

    def __init__(self, data: MetaData, universe: list = None, **kwargs) -> None:
        super().__init__(data, universe, **kwargs)
        self.__signal = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.__trades = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.simulate()
        self.analyze()

    def __repr__(self) -> str:
        return super().__repr__() + '\n' + f'short={self.short}, long={self.long}'

    def strategy(self, df: pd.DataFrame, field: str = 'hist_15_buy_ratio_90%') -> pd.DataFrame:
        df_strat = df[[field]]
        df_strat['sig_l'] = 0.
        df_strat['sig_s'] = 0.
        df_strat['mean'] = df_strat[field].rolling(200).mean()
        df_strat['std'] = df_strat[field].rolling(200).std()
        df_strat.loc[(df_strat[field] - df_strat['mean']) > 2.0 * df_strat['std'], 'sig_l']  = 1.0
        df_strat.loc[(df_strat[field] - df_strat['mean']) < -2.0 * df_strat['std'], 'sig_s'] = -1.0
        # df_strat.loc[df_strat[field] > 0.58, 'sig_l'] = 1.0
        # df_strat.loc[df_strat[field] < 0.55, 'sig_s'] = -1.0
        return df_strat[['sig_l', 'sig_s']]


    def simulate(self, delay: int = 1):
        self.calc_signals(delay)
        self.calc_trades()

    def calc_trades(self):
        self.logger.info('Simulating trades ...')
        raw_trd_l = defaultdict(dict)
        raw_trd_s = defaultdict(dict)

        with ThreadPoolExecutor(max_workers=self.max_concur) as executor:
            to_do_map = defaultdict(tuple)
            for inst in self.universe:
                for month in self.data.window_data[inst]:
                    future = executor.submit(self.gen_trades, 
                                             self.data.raw_data[inst][month],
                                             self.__signal[month]['L'][inst],
                                             self.__signal[month]['S'][inst])
                    to_do_map[future] = (inst, month)

            done_iter = tqdm.tqdm(as_completed(to_do_map), total=len(to_do_map))
            for future in done_iter:
                try:
                    long_trades, short_trades = future.result()
                except Exception as e:
                    self.logger.error(f'Error in generating signals.')
                else:
                    inst, month = to_do_map[future]
                    self.logger.debug(f'Done generating signals for {inst} {month}.')
                    long_trades['symbol']  = inst
                    short_trades['symbol'] = inst
                    raw_trd_l[month][inst] = long_trades
                    raw_trd_s[month][inst] = short_trades

        for month in raw_trd_l:
            self.__trades[month]['L'] = pd.concat(raw_trd_l[month].values(), axis=0)
            self.__trades[month]['S'] = pd.concat(raw_trd_s[month].values(), axis=0)

    def gen_trades(self, 
                   data, 
                   long_signal, 
                   short_signal, 
                   delay: int = 0,
                   slip: float = 2.0, 
                   capital: float = 1e5,
                   stoploss: float = -0.02,
                   drawdown: float = -0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
        long_entry, short_entry = [], []
        long_entry_price, short_entry_price = [], []

        long_exit, short_exit = [], []
        long_exit_price, short_exit_price = [], []

        long_entry_shares, short_entry_shares = [], []
        long_exit_reasons, short_exit_reasons = [], []

        long_entry_2, short_entry_2 = [], []
        long_entry_price_2, short_entry_price_2 = [], []

        long_exit_2, short_exit_2 = [], []
        long_exit_price_2, short_exit_price_2 = [], []

        long_entry_shares_2, short_entry_shares_2 = [], []
        long_exit_reasons_2, short_exit_reasons_2 = [], []

        delayed_long_signal  = long_signal.shift(delay)
        delayed_short_signal = short_signal.shift(delay)


        holding, entry, peak = False, None, 0
        holding_2, entry_2, peak_2 = False, None, 0
        for idx in long_signal.index:
            open_price  = data.loc[idx, 'open']
            close_price = data.loc[idx, 'close']
            
            if not holding and long_signal[idx] > 0:
                holding, entry, peak = True, close_price, open_price
                long_entry += idx,
                long_entry_price += entry * (1 + slip * 1e-4),
                long_entry_shares += capital / entry,

            if holding:
                if holding_2:
                    if np.log(open_price) - np.log(entry_2) < stoploss or np.log(open_price) - np.log(peak_2) < drawdown or short_signal[idx] < 0:
                        reasons_2 = []
                        if np.log(open_price) - np.log(entry_2) < stoploss:
                            reasons_2 += 'stoploss',
                        if np.log(open_price) - np.log(peak_2) < drawdown:
                            reasons_2 += 'drawdown',
                        if short_signal[idx] < 0:
                            reasons_2 += 'flip',
                        long_exit_reasons_2 += ','.join(reasons_2),

                        long_exit_2 += idx,
                        long_exit_price_2 += close_price * (1 - slip * 1e-4),
                        holding_2, entry_2, peak_2 = False, None, 0

                    peak_2 = max(open_price, peak_2)


                if np.log(open_price) - np.log(entry) < stoploss or np.log(open_price) - np.log(peak) < drawdown or short_signal[idx] < 0:
                    reasons = []
                    if np.log(open_price) - np.log(entry) < stoploss:
                        reasons += 'stoploss',

                    if np.log(open_price) - np.log(peak) < drawdown:
                        reasons += 'drawdown',

                    if short_signal[idx] < 0:
                        reasons += 'flip',

                    long_exit_reasons += ','.join(reasons),

                    long_exit += idx,
                    long_exit_price += close_price * (1 - slip * 1e-4),
                    holding, entry, peak = False, None, 0
                else:
                    if not holding_2 and delayed_long_signal[idx] > 0 and delay > 0:
                        holding_2, entry_2, peak_2 = True, close_price, open_price
                        long_entry_2 += idx,
                        long_entry_price_2 += entry_2 * (1 + slip * 1e-4),
                        long_entry_shares_2 += capital / entry_2,

                peak = max(open_price, peak)

        if len(long_exit) < len(long_entry) - 1:
            self.logger.critical(f'Wrong long entries and exits!')
            sys.exit(1)

        if len(long_exit_2) < len(long_entry_2) - 1:
            self.logger.critical(f'Wrong 2 long entries and exits!')
            sys.exit(1)

        if len(long_exit) == len(long_entry) - 1:
            long_exit += long_signal.index[-1],
            long_exit_price += close_price,
            long_exit_reasons += 'end', 

        if len(long_exit_2) == len(long_entry_2) - 1:
            long_exit_2 += long_signal.index[-1],
            long_exit_price_2 += close_price,
            long_exit_reasons_2 += 'end', 


        holding, entry, trough = False, None, float('inf')
        holding_2, entry_2, trough_2 = False, None, float('inf')
        for idx in short_signal.index:
            open_price  = data.loc[idx, 'open']
            close_price = data.loc[idx, 'close']
            
            if not holding and short_signal[idx] < 0:
                holding, entry, trough = True, close_price, open_price
                short_entry += idx,
                short_entry_price += entry * (1 - slip * 1e-4),
                short_entry_shares += 1.0 * capital / entry,

            if holding:
                if holding_2:
                    if np.log(open_price) - np.log(entry_2) > -stoploss or np.log(open_price) - np.log(trough_2) > -drawdown or long_signal[idx] > 0:
                        reasons_2 = []
                        if np.log(open_price) - np.log(entry_2) > -stoploss:
                            reasons_2 += 'stoploss',
                        if np.log(open_price) - np.log(trough_2) > -drawdown:
                            reasons_2 += 'drawdown',
                        if long_signal[idx] > 0:
                            reasons_2 += 'flip',

                        short_exit_reasons_2 += ','.join(reasons_2),

                        short_exit_2 += idx,
                        short_exit_price_2 += close_price * (1 - slip * 1e-4),
                        holding_2, entry_2, trough_2 = False, None, float('inf')

                    trough_2 = min(open_price, trough_2)

                if np.log(open_price) - np.log(entry) > -stoploss or np.log(open_price) - np.log(trough) > -drawdown or long_signal[idx] > 0:
                    reasons = []
                    if np.log(open_price) - np.log(entry) > -stoploss:
                        reasons += 'stoploss',

                    if np.log(open_price) - np.log(trough) > -drawdown:
                        reasons += 'drawdown',

                    if long_signal[idx] > 0:
                        reasons += 'flip',                    

                    short_exit_reasons += ','.join(reasons),

                    short_exit += idx,
                    short_exit_price += close_price * (1 - slip * 1e-4),
                    holding, entry, trough = False, None, float('inf')
                else:
                    if not holding_2 and delayed_short_signal[idx] < 0 and delay > 0:
                        holding_2, entry_2, trough_2 = True, close_price, open_price
                        short_entry_2 += idx,
                        short_entry_price_2 += entry_2 * (1 + slip * 1e-4),
                        short_entry_shares_2 += 1.0 * capital / entry_2,

                trough = min(open_price, peak)

        if len(short_exit) < len(short_entry) - 1:
            self.logger.critical(f'Wrong short entries and exits!')
            sys.exit(1)

        if len(short_exit_2) < len(short_entry_2) - 1:
            self.logger.critical(f'Wrong 2 short entries and exits!')
            sys.exit(1)

        if len(short_exit) == len(short_entry) - 1:
            short_exit += short_signal.index[-1],
            short_exit_price += close_price,
            short_exit_reasons += 'end', 

        if len(short_exit_2) == len(short_entry_2) - 1:
            short_exit_2 += short_signal.index[-1],
            short_exit_price_2 += close_price,
            short_exit_reasons_2 += 'end', 

        df_long = pd.DataFrame({
            'entry_time':    long_entry,
            'entry_price':   long_entry_price,
            'entry_shares':  long_entry_shares,
            'exit_time':     long_exit,
            'exit_price':    long_exit_price,
            'exit_reasons:': long_exit_reasons,
        })

        df_short = pd.DataFrame({
            'entry_time':    short_entry,
            'entry_price':   short_entry_price,
            'entry_shares':  short_entry_shares,
            'exit_time':     short_exit,
            'exit_price':    short_exit_price,
            'exit_reasons:': short_exit_reasons,
        })

        df_long_2 = pd.DataFrame({
            'entry_time':    long_entry_2,
            'entry_price':   long_entry_price_2,
            'entry_shares':  long_entry_shares_2,
            'exit_time':     long_exit_2,
            'exit_price':    long_exit_price_2,
            'exit_reasons:': long_exit_reasons_2,
        })

        df_short_2 = pd.DataFrame({
            'entry_time':    short_entry_2,
            'entry_price':   short_entry_price_2,
            'entry_shares':  short_entry_shares_2,
            'exit_time':     short_exit_2,
            'exit_price':    short_exit_price_2,
            'exit_reasons:': short_exit_reasons_2,
        })

        df_long['hands']    = '1st'
        df_long_2['hands']  = '2nd'
        df_short['hands']   = '1st'
        df_short_2['hands'] = '2nd'

        return pd.concat([df_long, df_long_2], axis=0).sort_values('entry_time'), pd.concat([df_short, df_short_2], axis=0).sort_values('entry_time')

    def calc_signals(self, delay: int = 1):
        self.logger.info('Simulating signals ...')
        raw_sig_l = defaultdict(dict)
        raw_sig_s = defaultdict(dict)

        with ThreadPoolExecutor(max_workers=self.max_concur) as executor:
            to_do_map = defaultdict(tuple)
            for inst in self.universe:
                for month in self.data.window_data[inst]:
                    future = executor.submit(self.strategy, self.data.window_data[inst][month])
                    to_do_map[future] = (inst, month)

            done_iter = tqdm.tqdm(as_completed(to_do_map), total=len(to_do_map))
            for future in done_iter:
                try:
                    result = future.result()
                except Exception as e:
                    self.logger.error(f'Error in generating signals.')
                else:
                    inst, month = to_do_map[future]
                    self.logger.debug(f'Done generating signals for {inst} {month}.')
                    raw_sig_l[month][inst] = result['sig_l']
                    raw_sig_s[month][inst] = result['sig_s']

        for month in raw_sig_l:
            self.__signal[month]['L'] = pd.DataFrame(raw_sig_l[month]).shift(delay).ffill().fillna(0)
            self.__signal[month]['S'] = pd.DataFrame(raw_sig_s[month]).shift(delay).ffill().fillna(0)

    def analyze(self, slip: float = -0., delay: int = 1):
        self.logger.info(f'Analyzing simulation results ...')
        self.pnl = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.pnl_net = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.dailyPnL = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.dailyPnLNet = defaultdict(lambda: defaultdict(pd.DataFrame))
        self.annualRet = defaultdict(lambda: defaultdict(float))
        self.annualRetNet = defaultdict(lambda: defaultdict(float))
        self.annualVol = defaultdict(lambda: defaultdict(float))
        self.annualVolNet = defaultdict(lambda: defaultdict(float))
        self.sharpe = defaultdict(lambda: defaultdict(float))
        self.sharpe_net = defaultdict(lambda: defaultdict(float))
        for month in self.trades:
            for side in ['L', 'S']:
                """
                df_sig  = self.signal[month][side]
                df_close = pd.DataFrame({
                    inst: self.data.raw_data[inst][month]['close']
                    for inst in df_sig.columns.to_list() })
                df_ret = np.log(df_close).diff().fillna(0)
                df_pnl = df_ret.mul(df_sig.shift(delay)).fillna(0)
                df_slip = df_sig.diff().abs().shift(delay) * slip * 1e-4
                dailyPnL = df_pnl.resample('1D').sum()
                dailyPnLNet = (df_pnl + df_slip).resample('1D').sum()
                self.pnl[month][side] = df_pnl
                self.pnl_net[month][side] = df_pnl + df_slip
                self.dailyPnL[month][side] = dailyPnL
                self.dailyPnLNet[month][side] = dailyPnLNet
                """

                df_trade = self.trades[month][side]
                df_trade['pnl'] = (df_trade['exit_price'] - df_trade['entry_price']).mul(df_trade['entry_shares'])
                df_trade.set_index('entry_time', inplace=True)
                df_pnl = df_trade.groupby('symbol')['pnl'].resample('1D').sum()
                dailyPnL = df_pnl.to_frame('pnl').reset_index().pivot(index='entry_time', columns='symbol', values='pnl')

                for inst in self.universe:
                    if not inst in dailyPnL.columns.to_list():
                        dailyPnL[inst] = 0.

                self.annualRet[month][side] = dailyPnL.mean(axis=0) * self.DAYS_IN_YEAR
                self.annualVol[month][side] = dailyPnL.std(axis=0) * np.sqrt(self.DAYS_IN_YEAR)
                self.sharpe[month][side] = self.annualRet[month][side].div(self.annualVol[month][side])
                self.dailyPnL[month][side] = dailyPnL

            self.dailyPnL[month]['T'] = self.dailyPnL[month]['L'] + self.dailyPnL[month]['S']
            self.annualRet[month]['T'] = self.dailyPnL[month]['T'].mean(axis=0) * self.DAYS_IN_YEAR
            self.annualVol[month]['T'] = self.dailyPnL[month]['T'].std(axis=0) * np.sqrt(self.DAYS_IN_YEAR)
            self.sharpe[month]['T'] = self.annualRet[month]['T'].div(self.annualVol[month]['T'])

        self.annualRet2 = self.convertMonthSide(self.annualRet)
        self.annualVol2 = self.convertMonthSide(self.annualVol)
        self.sharpe2 = self.convertMonthSide(self.sharpe)

    def convertMonthSide(self, data: dict) -> dict[pd.DataFrame]:
        L, S, T = defaultdict(list), defaultdict(list), defaultdict(list)
        for month in sorted(data):
            L['Month'] += month,
            S['Month'] += month,
            T['Month'] += month,
            for inst in self.universe:
                L[inst] += data[month]['L'][inst],
                S[inst] += data[month]['S'][inst],
                T[inst] += data[month]['T'][inst],

        result = defaultdict(pd.DataFrame)
        result['L'] = pd.DataFrame(L).set_index('Month')
        result['S'] = pd.DataFrame(S).set_index('Month')
        result['T'] = pd.DataFrame(T).set_index('Month')
        return result

    @property
    def signal(self) -> dict[pd.DataFrame]:
        return self.__signal

    @property
    def trades(self) -> dict[pd.DataFrame]:
        return self.__trades


if __name__ == '__main__':

    data = SpotMarketData()
    data.window_all()
    t = TwoMovingAverage(data, None)
    t.simulate()