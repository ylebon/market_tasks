import time

import numpy as np
import pandas as pd

from core.task_step import TaskStep


def create_signal(series):
    """Create signal"""
    candle_size = 300
    mean = series.mean()
    return mean - x[-candle_size]


class Task(TaskStep):
    """Simple Cross Moving Average

    """

    def df_minimum_length(self, rolling_intervall, shift):
        """dataframe minimum length"""
        return int(rolling_intervall.replace('s', '')) + shift

    def run(self, df, rolling_intervall='3600s', shift=1200, chrono=False):
        """Execute strategy"""
        self.log.info(
            f"msg='creating dataset' rolling_intervall={rolling_intervall} shift={shift}".format(rolling_intervall,
                                                                                                 shift))

        # Dataframe
        df = df[['bid_price', 'ask_price']].resample('1s').last().bfill()

        # Check length
        if len(df) < self.df_minimum_length(rolling_intervall, shift):
            return pd.DataFrame()

        # Start Time
        if chrono:
            start_time = time.time()

        # Rolling data
        rolling_data = df['bid_price'].rolling(rolling_intervall, min_periods=3600)
        df.dropna(inplace=True)

        # Aggregated
        df_agg = rolling_data.agg({
            'signal': create_signal
        })

        df['signal'] = df_agg['signal']
        df['profit'] = (df.bid_price.shift(-shift) - df.ask_price).apply(lambda x: 1 if x > 0 else 0)

        y = np.nan_to_num(np.vstack(df['profit']))
        df['profit'] = y

        df.dropna(inplace=True)

        if chrono:
            duration = time.time() - start_time
            self.log.info("msg='execution duration' duration='{0}'".format(duration))

        return df
