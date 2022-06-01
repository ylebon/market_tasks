import time

import numpy as np
import pandas as pd
from sklearn import preprocessing

from core.task_step import TaskStep


def create_pattern(x):
    """Create signal"""
    c_1 = int(x[-1] > x[-300])
    c_2 = int(x[-300] > x[-600])
    c_3 = int(x[-600] > x[-900])
    c_4 = int(x[-900] > x[-1200])
    c_5 = int(x[-1200] > x[-1500])
    c_6 = int(x[-1500] > x[-1800])
    c_7 = int(x[-1800] > x[-2100])
    c_8 = int(x[-2100] > x[-2300])
    pattern = int("".join([str(c_1), str(c_2), str(c_3), str(c_4), str(c_5), str(c_6), str(c_7), str(c_8)]))
    return pattern


class Task(TaskStep):
    """Moving Average Pattern

    """
    patterns = [int(format(x, '08b')) for x in range(256)]
    x_encoder = preprocessing.LabelEncoder().fit(patterns)

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
            'pattern': create_pattern
        })

        df_agg.dropna(inplace=True)
        df.dropna(inplace=True)

        df['signal'] = self.x_encoder.transform(df_agg['pattern'])
        df['profit'] = (df.bid_price.shift(-shift) - df.ask_price).apply(lambda x: 1 if x > 0 else 0)

        y = np.nan_to_num(np.vstack(df['profit']))
        df['profit'] = y

        df.dropna(inplace=True)

        if chrono:
            duration = time.time() - start_time
            self.log.info("msg='execution duration' duration='{0}'".format(duration))

        return df
