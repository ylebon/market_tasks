import time

import numpy as np
import pandas as pd

from core.task_step import TaskStep


class Task(TaskStep):
    """Simple Cross Moving Average
    Only for Forex
    """

    def df_minimum_length(self, rolling_intervall, shift):
        """dataframe minimum length"""
        return int(rolling_intervall.replace('s', '')) + shift

    def run(self, df, max_rolling_intervall='3600s', min_rolling_intervall='1200s', shift=1200, chrono=False,
            pct_profit=0.001):
        """execute strategy"""
        self.log.info("msg='creating dataset' max_rolling_intervall={} min_rolling_intervall={} shift={}".format(
            max_rolling_intervall, min_rolling_intervall, shift
        ))

        # Dataframe
        df = df[['bid_price', 'ask_price']].resample('1s').last().bfill()

        # Check length
        if len(df) < self.df_minimum_length(max_rolling_intervall, shift):
            return pd.DataFrame()

        start_time = time.time()

        # Rolling data
        min_rolling_data = df['bid_price'].rolling(min_rolling_intervall)
        max_rolling_data = df['bid_price'].rolling(max_rolling_intervall)

        # Mean rolling
        target_close = df.ask_price + (df.ask_price * pct_profit)
        df['signal'] = min_rolling_data.mean() - max_rolling_data.mean()
        df['profit'] = (df.bid_price.shift(-shift) - target_close).apply(lambda x: 1 if x > 0 else 0)
        df['profit'] = np.nan_to_num(np.vstack(df['profit']))

        df.dropna(inplace=True)
        duration = time.time() - start_time
        self.log.info("msg='execution duration' duration='{0}'".format(duration))

        return df
