import numpy as np
import pandas as pd
from scipy.stats import rankdata

def get_vwap(df: pd.DataFrame):
    v = df["volume"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    return ((h + l + c) / 3. * v).cumsum().div(v.cumsum())

def get_return(close):
    return close.pct_change().fillna(0.001)

def stdev(df, window=10):
    return df.rolling(window).std().fillna(method="bfill")

def ts_max(df, window=10):
    return df.rolling(window).max().fillna(method="bfill")

def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax).fillna(method="bfill")

def rank(df):
    return df.rank(axis=1, pct=True)

def delta(df, period=1):
    return df.diff(period).fillna(method="bfill")

def correlation(x1, x2, window=10):
    return (x1.rolling(window).corr(x2)).fillna(method="bfill")

def covariance(x1, x2, window=10):
    return (x1.rolling(window).cov(x2)).fillna(method="bfill")

def ts_rank(df, window=10):
    def _rolling_rank(na):
        return randdata(na)[-1]
    return (df.rolling(window).apply(_rolling_rank)).fillna(method="bfill")

def ts_sum(df, window=10):
    return df.rolling(window).mean().fillna(method="bfill")

def delay(df, period=1):
    return df.shift(period)

def ts_min(df, window=10):
    return df.rolling(window).min().fillna(method="bfill")

def ts_max(df, window=10):
    return df.rolling(window).max().fillna(method="bfill")

def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum())

def decaly_linear_pn(df, period=10):
    print(np.shape(df))
    if df.isnull().values.any():
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        df.fillna(0, inplace=True)

    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.values

    divisor = period * (period + 1) / 2.
    y = (np.arange(period) + 1) * 1.0 / divisor

    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1:row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))

    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)