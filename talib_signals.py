import os, sys
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any
from pathlib import Path 
from collections import defaultdict
import re, copy, json
import talib
import matplotlib.pyplot as plt
from datetime import datetime
from prefect import flow, task

def tsrank(array):
    s = pd.Series(array)
    return s.rank(ascending=True, pct=True).iloc[-1]

@task
def BBANDS(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        u, m, l = talib.BBANDS(c[sym], 
                               timeperiod=params["timeperiod"], 
                               nbdevup=params["std"],
                               nbdevdn=params["std"],
                               matype=0)

        df_bb = pd.DataFrame({"u": u, "m": m, "l": l, "c": c[sym]})
        df_bb["L"] = np.nan
        df_bb["S"] = np.nan

        df_bb.loc[df_bb.c > df_bb.u, "L"] = 1.
        df_bb.loc[df_bb.c < df_bb.m, "L"] = 0.
        df_bb.loc[df_bb.c < df_bb.l, "S"] = -1.
        df_bb.loc[df_bb.c > df_bb.m, "S"] = 0.

        df_bb["L"] = df_bb["L"].fillna(method="ffill")
        df_bb["S"] = df_bb["S"].fillna(method="ffill")
        df_bb["pos"] = df_bb["L"].fillna(0) + df_bb["S"].fillna(0)
        res[sym] = df_bb["pos"]

    return pd.DataFrame(res)

@task
def BBANDS_ST(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        u, m, l = talib.BBANDS(c[sym], 
                               timeperiod=params["timeperiod"], 
                               nbdevup=params["std"],
                               nbdevdn=params["std"],
                               matype=0)

        df_bb = pd.DataFrame({"u": u, "m": m, "l": l, "c": c[sym]})
        df_bb["L"] = np.nan
        df_bb["S"] = np.nan

        df_bb.loc[df_bb.c < df_bb.l, "L"] = 1.
        df_bb.loc[df_bb.c > df_bb.m, "L"] = 0.
        df_bb.loc[df_bb.c > df_bb.u, "S"] = -1.
        df_bb.loc[df_bb.c < df_bb.m, "S"] = 0.

        df_bb["L"] = df_bb["L"].fillna(method="ffill")
        df_bb["S"] = df_bb["S"].fillna(method="ffill")
        df_bb["pos"] = df_bb["L"].fillna(0) + df_bb["S"].fillna(0)
        res[sym] = df_bb["pos"]

    return pd.DataFrame(res)

@task
def RSI(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        df_rsi = talib.RSI(c[sym], timeperiod=params["timeperiod"])
        df = df_rsi.to_frame("rsi")
        df['L'] = np.nan
        df['S'] = np.nan
        df.loc[df.rsi >= params["lentry"], 'L'] = 1.0
        df.loc[df.rsi < params["lexit"],   'L'] = 0.0
        df.loc[df.rsi <= params["sentry"], 'S'] = -1.0
        df.loc[df.rsi > params["sexit"],   'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def RSI_ST(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        df_rsi = talib.RSI(c[sym], timeperiod=params["timeperiod1"]).rolling(window=params["timeperiod2"]).apply(tsrank)
        df = df_rsi.to_frame("rsi")
        df['L'] = np.nan
        df['S'] = np.nan
        df.loc[df.rsi <= params["lentry"], 'L'] = 1.0
        df.loc[df.rsi > params["lexit"],   'L'] = 0.0
        df.loc[df.rsi >= params["sentry"], 'S'] = -1.0
        df.loc[df.rsi < params["sexit"],   'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def RSI_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        rsi = talib.RSI(c[sym], timeperiod=params["timeperiod1"])
        ma  = rsi.rolling(window=params["timeperiod2"]).mean()
        sd  = rsi.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "rsi": rsi})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.rsi >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.rsi <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.rsi <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.rsi >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def ADX_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    h = copy.deepcopy(df["high"])
    l = copy.deepcopy(df["low"])
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        adx = talib.ADX(h[sym], l[sym], c[sym], timeperiod=params["timeperiod1"])
        ma  = adx.rolling(window=params["timeperiod2"]).mean()
        sd  = adx.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "adx": adx})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.adx <= df.LOW2_CUT, 'L'] = 1.0
        df.loc[df.adx >= df.HGH2_CUT, 'L'] = 0.0
        df.loc[df.adx >= df.HGH_CUT,  'S'] = -1.0
        df.loc[df.adx <= df.LOW_CUT,  'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def APO_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        apo = talib.APO(c[sym], fastperiod=params["timeperiod1"], slowperiod=params["timeperiod2"], matype=0)
        ma  = apo.rolling(window=params["timeperiod3"]).mean()
        sd  = apo.rolling(window=params["timeperiod3"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "apo": apo})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.apo >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.apo <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.apo <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.apo >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def PPO_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        ppo = talib.PPO(c[sym], fastperiod=params["timeperiod1"], slowperiod=params["timeperiod2"], matype=0)
        ma  = ppo.rolling(window=params["timeperiod3"]).mean()
        sd  = ppo.rolling(window=params["timeperiod3"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "ppo": ppo})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.ppo >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.ppo <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.ppo <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.ppo >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def CCI_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    o = copy.deepcopy(df["open"])
    h = copy.deepcopy(df["high"])
    l = copy.deepcopy(df["low"])
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        cci = talib.CCI(h[sym], l[sym], c[sym], timeperiod=params["timeperiod1"])
        ma  = cci.rolling(window=params["timeperiod2"]).mean()
        sd  = cci.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "cci": cci})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.cci >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.cci <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.cci <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.cci >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def CMO_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        cmo = talib.CMO(c[sym], timeperiod=params["timeperiod1"])
        ma  = cmo.rolling(window=params["timeperiod2"]).mean()
        sd  = cmo.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "cmo": cmo})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.cmo >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.cmo <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.cmo <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.cmo >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def MFI_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    o = copy.deepcopy(df["open"])
    h = copy.deepcopy(df["high"])
    l = copy.deepcopy(df["low"])
    c = copy.deepcopy(df["close"])
    v = copy.deepcopy(df["volume"])
    res = {}
    for sym in c.columns:
        mfi = talib.MFI(h[sym], l[sym], c[sym], v[sym], timeperiod=params["timeperiod1"])
        ma  = mfi.rolling(window=params["timeperiod2"]).mean()
        sd  = mfi.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "mfi": mfi})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.mfi >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.mfi <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.mfi <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.mfi >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def MACD_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        macd, _, _ = talib.MACD(c[sym], 
                                fastperiod=params["timeperiod1"], 
                                slowperiod=params["timeperiod2"], 
                                signalperiod=params["timeperiod3"])
        ma  = macd.rolling(window=params["timeperiod4"]).mean()
        sd  = macd.rolling(window=params["timeperiod4"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "macd": macd})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.macd >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.macd <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.macd <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.macd >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def MOM_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        mom = talib.MOM(c[sym], timeperiod=params["timeperiod1"])
        ma  = mom.rolling(window=params["timeperiod2"]).mean()
        sd  = mom.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "mom": mom})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.mom >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.mom <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.mom <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.mom >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def PLUS_DI_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    h = copy.deepcopy(df["high"])
    l = copy.deepcopy(df["low"])
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        plus_di = talib.PLUS_DI(h[sym], l[sym], c[sym], timeperiod=params["timeperiod1"])
        ma  = plus_di.rolling(window=params["timeperiod2"]).mean()
        sd  = plus_di.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "plus_di": plus_di})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.plus_di >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.plus_di <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.plus_di <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.plus_di >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def TRIX_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        trix = talib.TRIX(c[sym], timeperiod=params["timeperiod1"])
        ma  = trix.rolling(window=params["timeperiod2"]).mean()
        sd  = trix.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "trix": trix})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.trix >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.trix <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.trix <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.trix >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def ULTOSC_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    h = copy.deepcopy(df["high"])
    l = copy.deepcopy(df["low"])
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        ultosc = talib.ULTOSC(h[sym], l[sym], c[sym], 
                                timeperiod1=params["timeperiod1"], 
                                timeperiod2=params["timeperiod2"], 
                                timeperiod3=params["timeperiod3"])
        ma  = ultosc.rolling(window=params["timeperiod4"]).mean()
        sd  = ultosc.rolling(window=params["timeperiod4"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "ultosc": ultosc})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.ultosc <= df.LOW2_CUT, 'L'] = 1.0
        df.loc[df.ultosc >= df.HGH2_CUT, 'L'] = 0.0
        df.loc[df.ultosc >= df.HGH_CUT,  'S'] = -1.0
        df.loc[df.ultosc <= df.LOW_CUT,  'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def WILLR_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    h = copy.deepcopy(df["high"])
    l = copy.deepcopy(df["low"])
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        willr = talib.WILLR(h[sym], l[sym], c[sym], timeperiod=params["timeperiod1"])
        ma  = willr.rolling(window=params["timeperiod2"]).mean()
        sd  = willr.rolling(window=params["timeperiod2"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "willr": willr})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.willr >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.willr <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.willr <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.willr >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)

@task
def ADOSC_MA(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    o = copy.deepcopy(df["open"])
    h = copy.deepcopy(df["high"])
    l = copy.deepcopy(df["low"])
    c = copy.deepcopy(df["close"])
    v = copy.deepcopy(df["volume"])
    res = {}
    for sym in c.columns:
        adosc = talib.ADOSC(h[sym],
                          l[sym],
                          c[sym], 
                          v[sym],
                          fastperiod=params["timeperiod1"], 
                          slowperiod=params["timeperiod2"])
        ma  = adosc.rolling(window=params["timeperiod3"]).mean()
        sd  = adosc.rolling(window=params["timeperiod3"]).std()
        df = pd.DataFrame({"ma": ma, "sd": sd, "adosc": adosc})

        df["HGH_CUT"]  = ma + 2 * sd
        df["LOW_CUT"]  = ma - 1 * sd
        df["HGH2_CUT"] = ma + 1 * sd
        df["LOW2_CUT"] = ma - 2 * sd

        df['L'] = np.nan
        df['S'] = np.nan

        df.loc[df.adosc >= df.HGH_CUT,  'L'] = 1.0
        df.loc[df.adosc <= df.LOW_CUT,  'L'] = 0.0
        df.loc[df.adosc <= df.LOW2_CUT, 'S'] = -1.0
        df.loc[df.adosc >= df.HGH2_CUT, 'S'] = 0.0

        df['L'] = df['L'].fillna(method="ffill")
        df['S'] = df['S'].fillna(method="ffill")
        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)
