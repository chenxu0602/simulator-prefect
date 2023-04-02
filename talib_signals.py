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