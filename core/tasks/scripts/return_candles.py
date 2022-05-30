import time

import numpy as np
import pandas as pd

from tasks.core.task_step import TaskStep


def remove_outliers(x):
    """remove outliers"""
    gap = 4 * np.std(x)
    return x[((np.mean(x) - gap) < x) & (x < (np.mean(x) + gap))]


def remove_outliers_pct(x):
    return x[(x.quantile(0.01) < x) & (x < x.quantile(0.99))]


def open_r(x):
    """open"""
    try:
        return remove_outliers_pct(x)[0]
    except IndexError:
        return np.nan


def close_r(x):
    """close"""
    try:
        return remove_outliers_pct(x)[-1]
    except IndexError:
        return np.nan


def open(x):
    """open"""
    try:
        return x[0]
    except IndexError:
        return np.nan


def close(x):
    """close"""
    try:
        return x[-1]
    except IndexError:
        return np.nan


def min(x):
    return np.min(x)


def median(x):
    return x.quantile(0.5)


def pct99(x):
    return x.quantile(0.99)


def pct01(x):
    return x.quantile(0.01)


def max(x):
    return np.max(x)


def std(x):
    return np.std(x)


def sum(x):
    return np.sum(x)


def count(x):
    return x.size


def diff(x):
    return np.diff(x)


def pct_change(x):
    if x.size > 1:
        return (x[-1] - x[0]) / x[0]
    else:
        return np.nan


def ratio(x):
    if len(x):
        return np.count_nonzero(x) > (len(x) / 2)
    else:
        return 0


class Task(TaskStep):
    """Create dataset from dataframe

    1) resample to 1s with the last value
    2) rolling with minimum periods equal to resample intervall
    3) resample to 60s
    4) open (first) close (last) how to deal with outliers
    """

    def df_minimum_length(self, rolling_intervall, shift):
        """dataframe minimum length"""
        return int(rolling_intervall.replace('s', '')) + shift

    def run(self, df, shift=None, chrono=False):
        """execute strategy"""
        self.log.info(f"msg='creating dataset' shif={shift}")
        df = df[['bid_price', 'ask_price']].resample('1s').last().bfill()

        # Check length
        if len(df) < 9600:
            return pd.DataFrame()

        df['bid_price.candle_1'] = (df.bid_price - df.bid_price.shift(1200)) / df.bid_price.shift(1200)
        df['bid_price.candle_2'] = (df.bid_price.shift(1200) - df.ask_price.shift(2400)) / df.ask_price.shift(2400)
        df['bid_price.candle_3'] = (df.bid_price.shift(2400) - df.ask_price.shift(3600)) / df.ask_price.shift(3600)
        df['bid_price.candle_4'] = (df.bid_price.shift(3600) - df.ask_price.shift(4800)) / df.ask_price.shift(4800)
        df['bid_price.candle_5'] = (df.bid_price.shift(4800) - df.ask_price.shift(6000)) / df.ask_price.shift(6000)
        df['bid_price.candle_6'] = (df.bid_price.shift(6000) - df.ask_price.shift(7200)) / df.ask_price.shift(7200)
        df['bid_price.candle_7'] = (df.bid_price.shift(7200) - df.ask_price.shift(8400)) / df.ask_price.shift(8400)
        df['bid_price.candle_8'] = (df.bid_price.shift(8400) - df.ask_price.shift(9600)) / df.ask_price.shift(9600)

        def create_y(value):
            if value > 0:
                return 1
            else:
                return 0

        df['next_profit'] = ((df.bid_price.shift(-1200) - df.ask_price) / df.ask_price).apply(create_y)

        # Dropna
        df.dropna(inplace=True)
        if df.empty:
            return pd.DataFrame()

        y = np.nan_to_num(np.vstack(df['next_profit']))
        df['next_profit'] = y

        df.dropna(inplace=True)

        if chrono:
            duration = time.time() - start_time
            self.log.info("msg='execution duration' duration='{0}'".format(duration))

        return df
