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

@task
def two_mvavg_simple(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    c = copy.deepcopy(df["close"])
    res = {}
    for sym in c.columns:
        df = pd.DataFrame({
            "fast": c[sym].rolling(window=params["fast"]).mean(),
            "slow": c[sym].rolling(window=params["slow"]).mean()})

        df['L'] = 0
        df['S'] = 0

        df.loc[df.fast > df.slow, 'L'] = 1.0
        df.loc[df.fast < df.slow, 'S'] = -1.0

        df["pos"] = df['L'].fillna(0) + df['S'].fillna(0)
        res[sym] = df["pos"]

    return pd.DataFrame(res)