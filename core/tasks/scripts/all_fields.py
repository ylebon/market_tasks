import time

import numpy as np
import pandas as pd

from core.task_step import TaskStep


class Task(TaskStep):
    """Create dataset from dataframe
    """

    def df_minimum_length(self, rolling_intervall, shift):
        """dataframe minimum length"""
        return int(rolling_intervall.replace('s', '')) + shift

    def run(self, df, shift=600, chrono=False, rolling_intervall="7200s", pct_profit=0.001):
        """execute strategy"""
        self.log.info(f"msg='creating dataset' shift={shift} rolling_intervall={rolling_intervall}")
        df = df.resample('1s').last().bfill()

        # Check length
        if len(df) < 3600:
            return pd.DataFrame()

        start_time = time.time()

        # Next Profit
        def fn(x):
            return 1 if x > 0 else 0

        df['bid_price.rolling_mean'] = df.bid_price.rolling(rolling_intervall).mean()
        df['ask_price.rolling_mean'] = df.ask_price.rolling(rolling_intervall).mean()

        df['bid_qty.rolling_mean'] = df.bid_qty.rolling(rolling_intervall).mean()
        df['ask_qty.rolling_mean'] = df.ask_qty.rolling(rolling_intervall).mean()

        df['bid_price.bid_price_mean.diff'] = df.bid_price - df['bid_price.rolling_mean']
        df['ask_price.ask_price_mean.diff'] = df.ask_price - df['ask_price.rolling_mean']

        df['bid_qty.bid_qty_mean.diff'] = df.bid_qty - df['bid_qty.rolling_mean']
        df['ask_qty.ask_qty_mean.diff'] = df.ask_qty - df['ask_qty.rolling_mean']

        target_close = df.ask_price + (df.ask_price * pct_profit)
        df['profit'] = (df.bid_price.shift(-shift) - target_close).apply(fn)

        # Dropna
        df.dropna(inplace=True)
        if df.empty:
            return pd.DataFrame()

        df['profit'] = np.nan_to_num(np.vstack(df['profit']))

        # Dropna in place
        df.dropna(inplace=True)

        # Duration
        duration = time.time() - start_time
        self.log.info("msg='execution duration' duration='{0}'".format(duration))

        return df
