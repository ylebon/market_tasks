import time

import numpy as np
import pandas as pd

from core.task_step import TaskStep


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

    def run(self, df, rolling_intervall='3600s', shift=1200, buy_threshold=0.01, sell_threshold=-0.01,
                 chrono=False):
        """execute strategy"""
        self.log.info("msg='creating dataset' rolling_intervall={} shift={} buy_threshold={} sell_threshold={}".format(
            rolling_intervall,
            shift,
            buy_threshold,
            sell_threshold
        ))
        # Re-sample data to 1s
        df = df[['bid_price', 'ask_price']].resample('1s').last().bfill()

        # Check length
        if len(df) < self.df_minimum_length(rolling_intervall, shift):
            return pd.DataFrame()

        # Start Time
        if chrono:
            start_time = time.time()

        # Rolling and minimum periods
        rolling_data = df['bid_price'].rolling(rolling_intervall)
        rolling_q_25 = rolling_data.quantile(0.25)
        rolling_q_75 = rolling_data.quantile(0.75)
        rolling_mean = rolling_data.mean()
        rolling_var = rolling_data.var()
        rolling_coeff = (rolling_q_75 - rolling_q_25) / (rolling_q_75 + rolling_q_25)
        df['bid_price.open'] = df['bid_price'].shift(shift)
        df['bid_price.close'] = df['bid_price'].shift(-shift)

        df['ask_price.open'] = df['ask_price'].shift(shift)
        df['ask_price.close'] = df['ask_price'].shift(-shift)

        def create_y(value):
            if value > 0:
                return 1
            else:
                return 0

        # Dataset
        df['bid_price.rolling_mavg'] = rolling_mean
        df['bid_price.rolling_var'] = rolling_var
        df['bid_price.rolling_coeff'] = rolling_coeff
        df['bid_price.actual_pct_change'] = (df.bid_price - df['bid_price.open']) / df['bid_price.open']
        df['ask_price.actual_pct_change'] = (df.ask_price - df['ask_price.open']) / df['ask_price.open']
        df['bid_price.next_pct_change'] = (df['bid_price.close'] - df.ask_price).apply(create_y)

        # Dropna
        df.dropna(inplace=True)
        if df.empty:
            return pd.DataFrame()

        y = np.nan_to_num(np.vstack(df['bid_price.next_pct_change']))
        df['bid_price.next_pct_change'] = y

        df.dropna(inplace=True)

        if chrono:
            duration = time.time() - start_time
            self.log.info("msg='execution duration' duration='{0}'".format(duration))

        return df
