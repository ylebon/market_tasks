import numpy as np
import pandas as pd

from core.task_step import TaskStep


class Task(TaskStep):
    """Create dataset from dataframe
    """

    def df_minimum_length(self, rolling_intervall, shift):
        """dataframe minimum length"""
        return int(rolling_intervall.replace('s', '')) + shift

    def run(self, df, rolling_intervall, shift, pct_profit=0.001):
        df = df[['bid_price', 'ask_price']].resample('1s').last().bfill()

        if len(df) < self.df_minimum_length(rolling_intervall, shift):
            return pd.DataFrame()

        rolling_data = df['bid_price'].rolling(rolling_intervall)

        df['open'] = rolling_data.apply(lambda x: x[0])
        df['close'] = rolling_data.apply(lambda x: x[-1])
        df['low'] = rolling_data.min()
        df['high'] = rolling_data.max()
        df['rolling_mean'] = rolling_data.mean()
        df['rolling_var'] = rolling_data.var()
        rolling_q_25 = rolling_data.quantile(0.25)
        rolling_q_75 = rolling_data.quantile(0.75)
        df['rolling_coeff'] = (rolling_q_75 - rolling_q_25) / (rolling_q_75 + rolling_q_25)

        target_close_price = df.ask_price + (df.ask_price * pct_profit)
        df['profit'] = (df.bid_price.shift(-shift) - target_close_price).apply(lambda x: 1 if x > 0 else 0)
        df['profit'] = np.nan_to_num(np.vstack(df['profit']))

        df.dropna(inplace=True)
        return df[['open', 'close', 'low', 'high', 'profit', 'rolling_mean', 'rolling_var', 'rolling_coeff']]
