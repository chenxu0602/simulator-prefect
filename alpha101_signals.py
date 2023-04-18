import os, sys
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any
from pathlib import Path 
from collections import defaultdict
import re, copy, json
import matplotlib.pyplot as plt
from datetime import datetime
from prefect import flow, task
from utils import *

def tsrank(array):
    s = pd.Series(array)
    return s.rank(ascending=True, pct=True).iloc[-1]

# @task
# def RSI(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
#     c = copy.deepcopy(df["close"])
#     res = {}
#     for sym in c.columns:
#         df_rsi = talib.RSI(c[sym], timeperiod=params["timeperiod"])
#         df = df_rsi.to_frame("rsi")
#         df['L'] = np.nan
#         df['S'] = np.nan
#         df.loc[df.rsi >= params["lentry"], 'L'] = 1.0
#         df.loc[df.rsi < params["lexit"],   'L'] = 0.0
#         df.loc[df.rsi <= params["sentry"], 'S'] = -1.0
#         df.loc[df.rsi > params["sexit"],   'S'] = 0.0

#         df['L'] = df['L'].fillna(method="ffill")
#         df['S'] = df['S'].fillna(method="ffill")
#         df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
#         res[sym] = df["pos"]

#     return pd.DataFrame(res)

@task
def alpha101_002(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    df1 = (np.log(v)).diff(params["lookback"]).rank(axis=1, pct=True)
    df2 = (np.log(c) - np.log(o)).rank(axis=1, pct=True)

    res = {}
    for sym in c.columns:
        res[sym] = df1[sym].rolling(params["lookback"] * 3).corr(df2[sym])

    return pd.DataFrame(res)

@task
def alpha101_006(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]
    res = {}
    for sym in c.columns:
        res[sym] = o[sym].rolling(params["lookback"]).corr(v[sym])

    return pd.DataFrame(res)



