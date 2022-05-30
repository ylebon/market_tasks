import pandas as pd
import time

import pandas as pd

from tasks.core.task_step import TaskStep


class Pattern(TaskStep):
    """Pattern

    """
    indexes = []
    rolling_series = None
    buy_patterns = []

    def load_config(self, config):
        """load config"""
        # load parameters
        parquet_file = config.get("parquet")
        params = config.get("params")
        nbr_samples = params.get("nbr_samples")
        step = params.get("step")
        size = int(params.get("rolling_intervall", "3600s").replace("s", ""))

        # create indexes
        self.create_indexes(nbr_samples=nbr_samples, step=step)

        # create rolling series
        self.rolling_series = RollingResampleList(max_len=size)

        # read parquet file
        df = pd.read_parquet(parquet_file)

        # filter BUY patterns
        self.buy_patterns = df[df.score == 1]['pattern'].values
        self.log.info(f"msg='buy patterns' patterns={self.buy_patterns}")

    def get_signal(self, element):
        """get signal"""
        self.rolling_series.update(element)
        if self.rolling_series.is_full():
            start_time = time.time()
            values = self.rolling_series.get_values()
            pattern = self.create_pattern(values)
            duration = time.time() - start_time
            self.log.info(f"msg='read pattern' pattern='{pattern}' duration='{duration}'")
            return int(pattern in self.buy_patterns)

    def create_pattern(self, x):
        """create pattern"""
        pattern_list = ["0000", "0000", "0000", "0000"]
        # pattern bin
        pattern_bin = ".".join(pattern_list)
        # convert to hex
        pattern_hex = hex(int((pattern_bin.replace(".", "")), 2))
        # convert to int
        return int(pattern_hex, 0)

    def create_indexes(self, nbr_samples=None, step=None, **kwargs):
        """return indexes"""
        self.indexes = [-(i + 1) * step for i in range(nbr_samples)]
        self.indexes = [-1] + self.indexes

    def train(self, df, rolling_intervall='3600s', shift=600, step=60, nbr_samples=30, pct_profit=0, **kwargs):
        """Create pattern"""
        # resample to 1S
        df = df[['bid_price', 'ask_price']].resample('1s').last().bfill()

        # set indexes
        self.create_indexes(nbr_samples=nbr_samples, step=step)

        # Rolling data
        rolling_data = df['bid_price'].rolling(rolling_intervall, min_periods=int(rolling_intervall.replace("s", "")))

        # Aggregated
        df_agg = rolling_data.agg({
            'pattern': self.create_pattern
        })

        # Drop
        profit = df.ask_price + (df.ask_price * pct_profit)
        df['pattern'] = df_agg['pattern']
        df['profit'] = (df.bid_price.shift(-shift) - profit).apply(lambda x: 1 if x > 0 else 0)
        df.dropna(inplace=True)

        return df
