import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

N = 20
start_year, end_year = 2022, 2023

data_dir = Path("/Users/chenxu/Work/Crypto/spot/monthly/klines")

volumes_by_month = defaultdict(list)

stable_coins = [
    "USDCUSDT",
    "BUSDUSDT",
    "USTUSDT",
    "TUSDUSDT",
    "UNIUSDT",
    "LUNAUSDT",
    "FDUSDUSDT",
]

def get_data(filename):
    df = pd.read_csv(filename, 
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

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.set_index("datetime")

data = defaultdict(lambda: defaultdict(pd.DataFrame))

for year in range(start_year - 1, end_year + 1):
    for month in range(1, 13):

        if year == start_year - 1 and month < 12:
            continue

        year_month = f"{year:04d}-{month:02d}"
        print(year_month)

        for f in data_dir.iterdir():
            sym = f.name

            if sym in stable_coins: continue

            filename = f / "1m" / f"{sym}-1m-{year_month}.csv"
            
            if not filename.exists(): continue

            df = get_data(filename)
            data[year_month][sym] = df.copy()


for ym in data:
    df_btc = data[ym]['BTCUSDT']
    df_eth = data[ym]['ETHUSDT']
    df_daily_volume_btc = df_btc["taker_buy_quote_volume"].resample("1D", closed="left", label="left").sum()
    df_daily_volume_eth = df_eth["taker_buy_quote_volume"].resample("1D", closed="left", label="left").sum()
    df_prc_btc = df_btc['open'].resample('1H', closed='left', label='left').first().ffill().bfill()
    df_prc_eth = df_eth['open'].resample('1H', closed='left', label='left').first().ffill().bfill()
    df_ret_btc = np.log(df_prc_btc).diff()
    df_ret_eth = np.log(df_prc_eth).diff()

    for sym in data[ym]:
        if sym in ['BTCUSDT', 'ETHUSDT']: continue
        df = data[ym][sym]
        df_daily_volume = df["taker_buy_quote_volume"].resample("1D", closed="left", label="left").sum()
        df_prc = df['open'].resample('1H', closed='left', label='left').first().ffill().bfill()
        df_ret = np.log(df_prc).diff()
        df_ret_tmp = pd.DataFrame({
            'BTCUSDT': df_ret_btc,
            'ETHUSDT': df_ret_eth,
            sym: df_ret}).fillna(0)
        btc_corr = df_ret_tmp.corr().iloc[0, 2] 
        eth_corr = df_ret_tmp.corr().iloc[1, 2] 

        vol_btc = df_daily_volume.median() / df_daily_volume_btc.median()
        vol_eth = df_daily_volume.median() / df_daily_volume_eth.median()
        volumes_by_month[ym].append((-df_daily_volume.median(), vol_btc, vol_eth, sym[:-4], btc_corr, eth_corr))


months = sorted(volumes_by_month.keys())
results = defaultdict(list)

all_symbols = set()

for i in range(0, len(months)):
    # univ_month = sorted(volumes_by_month[months[i - 1]])
    univ_month = sorted(volumes_by_month[months[i]])
    vols_b  = [x[1] for x in univ_month]
    vols_e  = [x[2] for x in univ_month]
    symbols = [x[3] for x in univ_month]
    btc_cor = [x[4] for x in univ_month]
    eth_cor = [x[5] for x in univ_month]
    univ = []
    for j in range(len(univ_month)):
        if j == N: break
        univ += f'{symbols[j]}|{vols_b[j]:.2f}|{vols_e[j]:.2f}|{btc_cor[j]:.2f}|{eth_cor[j]:.2f}',

    if len(univ) < N: univ += [''] * (N - len(univ))

    all_symbols |= set(symbols[:N])
    results[months[i] + "-01"] = univ

df = pd.DataFrame(results)
df.to_csv("univ_by_month_taker.csv", index=False)

with open("univ.csv", 'w') as f:
    for sym in all_symbols:
        if not sym == "":
            f.write(f"{sym[:-4]}\n")
