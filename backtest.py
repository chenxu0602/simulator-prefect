import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Set, List, Dict, Union, Optional, Any
from prefect import task, flow, unmapped, get_run_logger
from pathlib import Path
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
from collections import defaultdict, Counter
import json, re, copy, string
import talib_signals
from random import randint 
import matplotlib.pyplot as plt

DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
MINUTES_IN_HOUR = 60

def convert_freq_to_hours(freq: str = "1H") -> float:
    if freq[-1] == 'H': return float(freq[:-1])
    else: return int(freq[:-1]) / 60.

def create_output_filename(alpha_id: str, param: frozenset, ext: str) -> str:
    filename, param_dict = f"{alpha_id}", dict(param)
    for k in sorted(param_dict):
        v = param_dict[k]
        filename += f"_{k}_{v}"
    filename += ext
    return filename

@task
def load_bar(data_dir: str, sym: str, start: str, end: str, 
             time_to_use: str = "open_time") -> pd.DataFrame:
    sym_dir = Path(data_dir) / sym / "1m"
    logger = get_run_logger()
    logger.info(f"Load data from {sym_dir} ...")
    data = []
    for f in sym_dir.iterdir():
        df_tmp = pd.read_csv(f, 
                             header=None, 
                             names=[
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
                             ])

        data.append(df_tmp)
    df = pd.concat(data)
    df.sort_values(time_to_use, inplace=True)
    df.drop_duplicates(subset=[time_to_use], keep="first", inplace=True)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms").dt.tz_localize(None)
    df.drop(labels=["open_time", "close_time", "ignore"], axis=1, inplace=True)
    df.set_index("datetime", inplace=True)
    df = df.loc[(df.index >= start) & (df.index < end)]
    return df.sort_index()
        
@task 
def transform_dataframe(data: Dict[str, pd.DataFrame],
                       columns: List[str] = [
                            "open", 
                            "high", 
                            "low", 
                            "close", 
                            "volume",
                            "quote_volume",
                            "n_trades",
                            "taker_buy_base_volume",
                            "taker_buy_quote_volume"]) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Transforming data frames ...")
    symbols = sorted(data)
    level_1 = []
    for col in columns:
        tmp_res = {}
        for sym in symbols:
            tmp_res[sym] = data[sym][col]
        level_1.append(pd.DataFrame(tmp_res))
    return pd.concat(level_1, axis=1, keys=columns)

@task 
def resample(df: pd.DataFrame, 
             freq: str = "1H",
             columns: List[str] = [
                "open", 
                "high", 
                "low", 
                "close", 
                "volume",
                "quote_volume",
                "n_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume"]) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Resample data to {freq} ...")

    res = {}
    for col in columns:
        if freq == "1T":
            res[col] = df[col]
        else:
            if col == "open":
                res[col] = df[col].resample(freq, closed="left", label="left").first()
            elif col == "high":
                res[col] = df[col].resample(freq, closed="left", label="left").max()
            elif col == "low":
                res[col] = df[col].resample(freq, closed="left", label="left").min()
            elif col == "close":
                res[col] = df[col].resample(freq, closed="left", label="left").last()
            elif col in ["n_trades"] or "volume" in col:
                res[col] = df[col].resample(freq, closed="left", label="left").sum()

        if col in ["open", "high", "low", "close"]:
            res[col] = res[col].fillna(method="ffill")
        elif col in ["n_trades"] or "volume" in col:
            res[col] = res[col].fillna(0)

    res["ret"] = np.log(res["close"]).diff()

    columns += "ret",
    df_res = pd.concat([res[k] for k in columns], keys=columns, axis=1)
    return df_res

@flow(task_runner=SequentialTaskRunner(), flow_run_name="data")
def data_flow(data_dir: str, univ_file: str, start: str, end: str, freq: str):
    results, symbols = {}, []
    with open(univ_file, 'r') as f:
        for line in f:
            sym = line.strip('\n').strip('')
            symbols.append(sym)

    logger = get_run_logger()
    logger.info("Universe: {}".format(', '.join(symbols)))

    data = load_bar.map(data_dir=unmapped(data_dir),
                        sym=symbols,
                        start=unmapped(start),
                        end=unmapped(end))

    for sym, d in zip(symbols, data):
        results[sym] = d

    return resample(transform_dataframe(results), freq)

@task
def load_params(filename: str) -> Tuple[Dict[str, List[Any]]]:
    logger = get_run_logger()
    logger.info(f"Param file: {filename}")
    res, comb = defaultdict(list), defaultdict(list)
    with open(filename, 'r') as f:
        raw_data = json.load(f)
        for par in raw_data:
            if "alpha_id" in par:
                res[par["alpha_id"]] += par["params"]
            elif "comb_id" in par:
                res[par["comb_id"]] += par["params"]
    return res, comb

@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="talib")
def talib_flow(data: Any, config: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Generate Ta-Lib signals ...")

    params, combs = load_params(config)

    signals = defaultdict(lambda: defaultdict(pd.DataFrame))
    signal_wait = defaultdict(lambda: defaultdict(pd.DataFrame))

    for alpha_id in sorted(params):
        for param in params[alpha_id]:
            func = getattr(talib_signals, alpha_id)
            fut = func.submit(data, param)
            signal_wait[alpha_id][frozenset(param.items())] = fut

    for alpha_id in sorted(signal_wait):
        for param in signal_wait[alpha_id]:
            sig = signal_wait[alpha_id][param].wait().result()
            signals[alpha_id][param] = sig

    return signals

@task
def calc_positions(sig: pd.DataFrame, delay: int = 0, max_pos: float = 1.0) -> pd.DataFrame:
    return sig.shift(delay) * max_pos

@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="position")
def pos_flow(signals: Dict[str, Any], delay: int = 0):
    positions = defaultdict(lambda: defaultdict(pd.DataFrame))
    position_wait = defaultdict(lambda: defaultdict(pd.DataFrame))

    for alpha_id in sorted(signals):
        for param in signals[alpha_id]:
            sig = signals[alpha_id][param]
            fut = calc_positions.submit(sig, 0)
            position_wait[alpha_id][param] = fut

    for alpha_id in sorted(position_wait):
        for param in position_wait[alpha_id]:
            pos = position_wait[alpha_id][param].wait().result()
            positions[alpha_id][param] = pos

    return positions

@task
def calc_slippage(data: Any, pos: pd.DataFrame, rate: float = 5e-4) -> Dict[str, pd.DataFrame]:
    posL = copy.deepcopy(pos)
    posS = copy.deepcopy(pos)
    posL[posL < 0] = 0
    posS[posS > 0] = 0

    t =  pos.diff().abs() * rate * -1.0
    l = posL.diff().abs() * rate * -1.0
    s = posS.diff().abs() * rate * -1.0

    return {'T': t, 'L': l, 'S': s} 

@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="slip")
def slip_flow(data: Any, positions: Dict[str, Any]) -> Dict[str, Any]:
    slip = defaultdict(lambda: defaultdict(lambda: defaultdict(pd.DataFrame)))
    slip_wait = defaultdict(lambda: defaultdict(pd.DataFrame))

    for alpha_id in sorted(positions):
        for param in positions[alpha_id]:
            pos = positions[alpha_id][param]
            fut = calc_slippage.submit(data, pos)
            slip_wait[alpha_id][param] = fut

    for alpha_id in sorted(slip_wait):
        for param in slip_wait[alpha_id]:
            s = slip_wait[alpha_id][param].wait().result()
            slip[alpha_id][param] = s

    return slip

@task
def calc_pnls(data: Any, pos: pd.DataFrame, slip: Any) -> Dict[str, pd.DataFrame]:
    posL = copy.deepcopy(pos)
    posS = copy.deepcopy(pos)
    posL[posL < 0] = 0
    posS[posS > 0] = 0

    ret = data["ret"]
    pnl  = ret.mul(pos.shift(1))
    pnlL = ret.mul(posL.shift(1))
    pnlS = ret.mul(posS.shift(1))

    return {'T': pnl, 'N': pnl + slip['T'], 'L': pnlL + slip['L'], 'S': pnlS + slip['S']}

@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="pnl")
def pnl_flow(data: Any, positions: Dict[str, Any], slippage: Any) -> Dict[str, Any]:
    pnls = defaultdict(lambda: defaultdict(lambda: defaultdict(pd.DataFrame)))
    pnl_wait = defaultdict(lambda: defaultdict(pd.DataFrame))

    for alpha_id in sorted(positions):
        for param in positions[alpha_id]:
            pos  = positions[alpha_id][param]
            slip = slippage[alpha_id][param]
            fut  = calc_pnls.submit(data, pos, slip)
            pnl_wait[alpha_id][param] = fut
            
    for alpha_id in sorted(pnl_wait):
        for param in pnl_wait[alpha_id]:
            p = pnl_wait[alpha_id][param].wait().result()
            pnls[alpha_id][param] = p

    return pnls

@task
def calc_trades(pnls: Dict[str, Any], pos: pd.DataFrame) -> pd.DataFrame:
    if pos.empty: return pd.DataFrame()

    df_trd = pd.DataFrame(index=pos.index, columns=pos.columns, dtype=np.str_)
    ppos = pos.shift(1)
    pchg = (pos - ppos).div(ppos.abs() + pos.abs()) * 2.0
    chg_thr = 0.05

    df_trd[(ppos <= 0) & (pos > 0)] = "Long Open"
    df_trd[(ppos >= 0) & (pos < 0)] = "Short Open"
    df_trd[(pchg.abs() < chg_thr) & ~(pos == 0)] = "Hold"
    df_trd[(pchg.abs() < chg_thr) & (pos == 0)] = "Empty"
    df_trd[(ppos > 0) & (pos == 0)] = "Long Close"
    df_trd[(ppos < 0) & (pos == 0)] = "Short Close"
    df_trd[(ppos > 0) & (pos > 0) & (pchg >= chg_thr)]  = "Long Incr"
    df_trd[(ppos > 0) & (pos > 0) & (pchg <= -chg_thr)] = "Long Decr"
    df_trd[(ppos < 0) & (pos < 0) & (pchg <= -chg_thr)] = "Short Decr"
    df_trd[(ppos < 0) & (pos < 0) & (pchg >= chg_thr)]  = "Short Decr"

    df_trd.columns = pd.MultiIndex.from_arrays([df_trd.columns, ["trades"] * len(df_trd.columns)])

    df_pnl = copy.deepcopy(pnls['N'])
    df_pnl.columns = pd.MultiIndex.from_arrays([df_pnl.columns, ["pnl"] * len(df_pnl.columns)])

    df_tmp = pd.concat([df_trd, df_pnl], axis=1)
    df_tmp.loc[:, (slice(None), "trades")] = df_tmp.loc[:, (slice(None), "trades")] \
        .applymap(lambda x: randint(1, 10**8) if x in ["Long Open", "Short Open"] else 0 if x in ["Long Close", "Short Close", "Empty"] else np.nan) \
        .fillna(method="ffill")

    df_tmp = df_tmp.stack(0)
    df_tmp.reset_index(inplace=True)
    df_tmp.columns = ["datetime", "symbol", "pnl", "trade_id"]
    df_tmp = df_tmp.loc[df_tmp["trade_id"] > 0]
    df_tmp["month"] = df_tmp["datetime"].dt.strftime("%Y-%m")

    results = df_tmp.groupby(["symbol", "month", "trade_id"]).agg(
        start=pd.NamedAgg(column="datetime", aggfunc="first"),
        end=pd.NamedAgg(column="datetime", aggfunc="last"),
        pnl=pd.NamedAgg(column="pnl", aggfunc="sum"))

    df_res = results.reset_index().drop("trade_id", axis=1)
    df_res["start"] = df_res["start"].dt.tz_localize(None)
    df_res["end"] = df_res["end"].dt.tz_localize(None)
    return df_res.sort_values(by="end")


@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="trade")
def trade_flow(pnls: Dict[str, Any], positions: Dict[str, Any]) -> Dict[str, Any]:
    trades = defaultdict(lambda: defaultdict(pd.DataFrame))
    trade_wait = defaultdict(lambda: defaultdict(pd.DataFrame))

    for alpha_id in sorted(pnls):
        for param in pnls[alpha_id]:
            pos = positions[alpha_id][param]
            pnl = pnls[alpha_id][param]
            fut = calc_trades.submit(pnl, pos)
            trade_wait[alpha_id][param] = fut

    for alpha_id in sorted(trade_wait):
        for param in trade_wait[alpha_id]:
            st = trade_wait[alpha_id][param].wait().result()
            trades[alpha_id][param] = st

    return trades

@task
def count_trades(df: pd.DataFrame) -> Tuple[List[int], List[float], Dict[str, int], Dict[str, float], Dict[str, int], Dict[str, float]]:
    wins = df.loc[df["pnl"] > 0, "pnl"]
    loss = df.loc[df["pnl"] < 0, "pnl"]

    winTrades, lossTrades = map(len, (wins, loss))
    winPnL  = wins.sum()
    lossPnL = loss.sum()

    tradesBySymbol, pnlBySymbol = {}, {}
    df_by_sym = df.groupby("symbol").agg(
        tradesBySymbol=pd.NamedAgg(column="pnl", aggfunc="count"),
        pnlBySymbol=pd.NamedAgg(column="pnl", aggfunc="sum"))

    for r in df_by_sym.itertuples():
        tradesBySymbol[r.Index] = r.tradesBySymbol
        pnlBySymbol[r.Index] = r.pnlBySymbol

    tradesByMonth, pnlByMonth = {}, {}
    df_by_month = df.groupby("month").agg(
        tradesByMonth=pd.NamedAgg(column="pnl", aggfunc="count"),
        pnlByMonth=pd.NamedAgg(column="pnl", aggfunc="sum"))

    for r in df_by_month.itertuples():
        tradesByMonth[r.Index] = r.tradesByMonth
        pnlByMonth[r.Index] = r.pnlByMonth

    return [winTrades, lossTrades], [winPnL, lossPnL], tradesBySymbol, pnlBySymbol, tradesByMonth, pnlByMonth

@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="count_trades")
def count_trades_flow(trades: Dict[str, Any]) -> Dict[str, Any]:
    count_wait = defaultdict(lambda: defaultdict(pd.DataFrame))

    winTrades  = defaultdict(lambda: defaultdict(int))
    lossTrades = defaultdict(lambda: defaultdict(int))
    winPnL     = defaultdict(lambda: defaultdict(float))
    lossPnL    = defaultdict(lambda: defaultdict(float))

    tradesBySymbol = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    tradesByMonth  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pnlBySymbol    = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    pnlByMonth     = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for alpha_id in sorted(trades):
        for param in trades[alpha_id]:
            trd = trades[alpha_id][param]
            fut = count_trades.submit(trd)
            count_wait[alpha_id][param] = fut

    for alpha_id in sorted(count_wait):
        for param in trades[alpha_id]:
            cnts, pnls, trBySym, pnlBySym, trByMon, pnlByMon = count_wait[alpha_id][param].wait().result()

            winTrades[alpha_id][param]  = cnts[0]
            lossTrades[alpha_id][param] = cnts[1]
            winPnL[alpha_id][param]     = pnls[0]
            lossPnL[alpha_id][param]    = pnls[1]

            tradesBySymbol[alpha_id][param] = trBySym
            pnlBySymbol[alpha_id][param]    = pnlBySym
            tradesByMonth[alpha_id][param]  = trByMon
            pnlByMonth[alpha_id][param]     = pnlByMon

    return winTrades, lossTrades, winPnL, lossPnL, tradesBySymbol, pnlBySymbol, tradesByMonth, pnlByMonth

def calc_tvr(pos: pd.DataFrame, day_factor: float) -> float:
    return pos.diff().abs().sum(axis=1).mean() / pos.abs().sum(axis=1).mean() * day_factor

def calc_tvr2(pos: pd.DataFrame, day_factor: float) -> float:
    return pos.diff().abs().sum(axis=1).mean() / len(pos.columns) * day_factor

@task
def calc_summary(
    pnls: Dict[str, Any],
    pos: Dict[str, Any],
    winTrades: int,
    lossTrades: int,
    winPnL: float,
    lossPnL: float,
    param: frozenset,
    alpha_id: str,
    freq: str,
    start: str,
    end: str) -> pd.DataFrame: 

    res = defaultdict(list)
    pnl = pnls['N'].sum(axis=1, skipna=True)

    funding = pos.abs().sum(axis=1)

    pnl = pnl.loc[(pnl.index >= start) & (pnl.index < end)]
    funding = funding.loc[(funding.index >= start) & (funding.index < end)]

    param_dict = dict(param)
    for k in sorted(param_dict):
        res[k] += param_dict[k],

    res["bar_freq"]  += freq,
    res["alpha_id"]  += alpha_id,
    res["total_pnl"] += pnl.sum(),
    res["ave_pos"]   += pos.abs().sum(axis=1).mean(),

    tot_trades = winTrades + lossTrades
    res["tot_trades"] += tot_trades,

    ave_pnl = (winPnL + lossPnL) / tot_trades if tot_trades > 0 else 0.
    res["ave_pnl"] += ave_pnl,

    ave_win_pnl = winPnL / winTrades if winTrades > 0 else 0.
    res["ave_win_pnl"] += ave_win_pnl,

    ave_loss_pnl = lossPnL / lossTrades if lossTrades > 0 else 0.
    res["ave_loss_pnl"] += ave_loss_pnl,

    res["max_funding"] += funding.max(),

    maxdd = (pnl.cumsum().cummax() - pnl.cumsum()).max()
    res["maxdd"] += maxdd,

    day_factor  = HOURS_IN_DAY / convert_freq_to_hours(freq)
    year_factor = DAYS_IN_YEAR * day_factor

    ret, vol = pnl.mean(), pnl.std()
    ann_ret = ret * year_factor
    ann_vol = vol * np.sqrt(year_factor)

    calmer = pnl.sum() / maxdd if maxdd > 0 else 0.
    res["calmer"] += calmer,

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.
    res["ann_ret"] += ann_ret,
    res["ann_vol"] += ann_vol,
    res["sharpe"]  += sharpe,

    res["tvr"]  += calc_tvr(pos, day_factor),
    res["tvr2"] += calc_tvr2(pos, day_factor),

    res["start"] += pnl.index.min().strftime("%Y-%m-%d"),
    res["end"]   += pnl.index.max().strftime("%Y-%m-%d"),

    df = pd.DataFrame(res)
    df.set_index(sorted(param_dict.keys()), inplace=True)
    return df

@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="summary")
def summary_flow(
    pnls: Dict[str, Any],
    positions: Dict[str, Any],
    winTrades: Dict[str, Any],
    lossTrades: Dict[str, Any],
    winPnL: Dict[str, Any],
    lossPnL: Dict[str, Any],
    freq: str,
    start: str,
    end: str,
    output: str = "Results") -> Dict[str, Any]:

    logger = get_run_logger()

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"summary.xlsx"
    summary_file.unlink(missing_ok=True)

    def style_negative(v, props=""):
        if pd.isna(v) or isinstance(v, str):
            return None
        return props if v < 0 else None

    def convert_idx_to_letter(idx):
        res = ""
        while idx:
            idx, m = divmod(idx - 1, 26)
            res = string.ascii_letters[m] + res
        return res

    summary_wait = defaultdict(lambda: defaultdict(pd.DataFrame))
    summaries = defaultdict(lambda: defaultdict(pd.DataFrame))

    for alpha_id in sorted(pnls):
        for param in pnls[alpha_id]:
            pnl = pnls[alpha_id][param]
            pos = positions[alpha_id][param]
            wt  = winTrades[alpha_id][param]
            lt  = lossTrades[alpha_id][param]
            wp  = winPnL[alpha_id][param]
            lp  = lossPnL[alpha_id][param]
            fut = calc_summary.submit(pnl, pos, wt, lt, wp, lp, param, alpha_id, freq, start, end)
            summary_wait[alpha_id][param] = fut

    summary_by_aid = defaultdict(lambda: pd.DataFrame)

    with pd.ExcelWriter(summary_file, engine="xlsxwriter") as writer:
        workbook = writer.book

        format1 = workbook.add_format({"num_format": "#,##0.00"})
        format2 = workbook.add_format({"num_format": "0.00%"})
        
        for alpha_id in sorted(summary_wait):
            all_summaries = []
            for param in summary_wait[alpha_id]:
                summary = summary_wait[alpha_id][param].wait().result()
                summaries[alpha_id][param] = summary
                all_summaries += summary,

            logger.info(f"Writing {alpha_id} to {summary_file} ...")
            summary_alpha = pd.concat(all_summaries).sort_index()
            summary_by_aid[alpha_id] = summary_alpha

            ds = summary_alpha.round(2).style.applymap(style_negative, props="color:red;")
            ds.to_excel(writer, sheet_name=alpha_id, index=True)

    return summaries, summary_by_aid



@flow(task_runner=ConcurrentTaskRunner(), flow_run_name="sim")
def sim_flow(data_dir: str,
             univ_file: str,
             start: str,
             end: str,
             freq: str,
             exec: str,
             config: str,
             output: str):
    
    logger = get_run_logger()
    data = data_flow(data_dir, univ_file, start, end, freq)
    signals = talib_flow(data, config)
    positions = pos_flow(signals, delay=0)
    slippage = slip_flow(data, positions)
    pnls = pnl_flow(data, positions, slippage)
    trades = trade_flow(pnls, positions)
    winTrades, lossTrades, winPnL, lossPnL, tradesBySymbol, pnlBySymbol, tradesByMonth, pnlByMonth = count_trades_flow(trades)
    summaries, summary_by_aid = summary_flow(pnls, positions, winTrades, lossTrades, winPnL, lossPnL, freq, start, end, output)

    return data, signals, positions, slippage, pnls, trades, \
        winTrades, lossTrades, winPnL, lossPnL, tradesBySymbol, pnlBySymbol, tradesByMonth, pnlByMonth, \
        summaries, summary_by_aid



if __name__ == "__main__":
    data, signals, positions, slippage, pnls, trades,\
        winTrades, lossTrades, winPnL, lossPnL, tradesBySymbol, pnlBySymbol, tradesByMonth, pnlByMonth, \
        summaries, summary_by_aid  = sim_flow(
        data_dir="/Users/chenxu/Work/Crypto/spot/monthly/klines",
        univ_file="univ.csv",
        start="2017-01-01", 
        end="2023-01-01", 
        freq="15T",
        exec="close",
        config="talib.json",
        output="Test")
