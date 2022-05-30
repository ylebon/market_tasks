import os
from glob import glob

import dask.dataframe as dd
import pandas as pd

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep
from utils import time_util


class Task(TaskStep):
    """
    Load feed

    """

    def run(self, instrument_id, start_date=None, columns=None, end_date=None, date=None, fields=None):
        config_services = ConfigServices.create()
        parquet_directory = config_services.get_value("PARQUET.LOCAL_DIRECTORY")
        parquet_directory = os.path.expanduser(parquet_directory)

        self.log.info(
            f"msg='convert parquet files to dataframe parquet='{parquet_directory}' start_date='{start_date}' end_date='{end_date}'")

        # Load parquet
        parquet_files = list()
        exchange, symbol = instrument_id.split("_", 1)
        if start_date and end_date:
            dates = time_util.get_dates(start_date, end_date)
            for date in dates:
                pattern = "{}/exchange={}/symbol={}/date={}/*.parquet".format(parquet_directory, exchange, symbol,
                                                                              date.strftime("%Y%m%d"))
                parquet_files.extend(glob(pattern))
        elif date:
            pattern = "{}/exchange={}/symbol={}/date={}/*.parquet".format(parquet_directory, exchange, symbol, date)
            parquet_files.extend(glob(pattern))
        else:
            pattern = "{}/exchange={}/symbol={}/date=*/*.parquet".format(parquet_directory, exchange, symbol)
            parquet_files.extend(glob(pattern))

        # parquet files loaded
        self.log.info(f"msg='total parquet files to load' pattern='{pattern}' size={len(parquet_files)}")

        # pandas dataframe list
        df_list = list()
        all_columns = columns + ['seq', 'time']
        for parquet_file in parquet_files:
            self.log.info(f"msg='loading parquet file' parquet_file='{parquet_file}'")
            df = dd.read_parquet(parquet_file, columns=all_columns, engine='pyarrow', index=False)
            df_list.append(df)
        df = dd.concat(df_list, axis=0).compute()

        # drop duplicates
        df = df.drop_duplicates('seq', keep='last')
        df.set_index("time", inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms', errors='raise')
        df['date'] = df.index
        df['exchange'] = exchange
        df['symbol'] = symbol
        df['exchange'] = df.exchange.astype('category')
        df['symbol'] = df.symbol.astype('category')

        # replace qty
        if 'ask_qty' in all_columns:
            df[['ask_qty']] = df[['ask_qty']].fillna(0)
        if 'bid_qty' in all_columns:
            df[['bid_qty']] = df[['bid_qty']].fillna(0)

        self.log.info(
            f"msg='instrument dataframe loaded' instrument_id={instrument_id} length='{len(df)}' start_date='{df.index[0]}' end_date='{df.index[-1]}'"
        )
        return df.sort_index()


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    task = Task("load_feed")
    df = task.run("CEX_BTC_USD")
