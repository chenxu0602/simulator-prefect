import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

N = 20
start_year, end_year = 2018, 2023

data_dir = Path("/Users/chenxu/Work/Crypto/spot/monthly/klines")

volumes_by_month = defaultdict(list)

stable_coins = [
    "USDCUSDT",
    "BUSDUSDT",
    "USTUSDT",
    "TUSDUSDT",
    "UNIUSDT",
    "LUNAUSDT",
]

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
            df.set_index("datetime", inplace=True)

            df_daily_volume = df["quote_volume"].resample("1D", closed="left", label="left").sum()

            volumes_by_month[year_month].append((-df_daily_volume.median(), sym))

months = sorted(volumes_by_month.keys())
results = defaultdict(list)

all_symbols = set()

for i in range(1, len(months)):
    univ_month = sorted(volumes_by_month[months[i - 1]])
    symbols = [x[1] for x in univ_month]
    if len(symbols) >= N:
        univ = symbols[:N]
    else:
        univ = symbols + [""] * (N - len(symbols))

    all_symbols |= set(univ)
    results[months[i] + "-01"] = univ

df = pd.DataFrame(results)
df.to_csv("univ_by_month.csv", index=False)

with open("univ.csv", 'w') as f:
    for sym in all_symbols:
        if not sym == "":
            f.write(f"{sym}\n")
