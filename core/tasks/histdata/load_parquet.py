import os
from glob import glob

import dask.dataframe as dd
import pandas as pd

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep
from utils import time_util


class Task(TaskStep):
    """Load instrument feed data from local parquet directory
    """

    def run(self, instrument_id, start_date, end_date, start_hour="00:00:00.000", end_hour="23:59:59.999",
            parquet_directory=None, columns=[]):

        # services config
        services_config = ConfigServices.create()

        # check instrument id
        if isinstance(instrument_id, str):
            instrument_id = instrument_id.split("_", 1)

        self.log.info("msg='load symbol prices' instrument_id={instrument_id}".format(instrument_id=instrument_id))
        # Exchange and symbol
        exchange, symbol = instrument_id

        # Parquet directory
        parquet_directory = parquet_directory or services_config.get_value("PARQUET.LOCAL_DIRECTORY")
        parquet_directory = os.path.expanduser(parquet_directory)

        # Set start and end date
        start_date = time_util.string_to_date(" ".join((start_date, start_hour)))
        end_date = time_util.string_to_date(" ".join((end_date, end_hour)))

        # Load parquet
        parquet_files = list()
        for date in time_util.get_dates(start_date, end_date):
            pattern = "{}/exchange={}/symbol={}/date={}/*.parquet".format(parquet_directory, exchange, symbol,
                                                                          date.strftime("%Y%m%d"))
            parquet_files.extend(glob(pattern))

        # check parquet files
        if len(parquet_files) == 0:
            return pd.DataFrame()

        # Load dataframe
        dask_df = dd.read_parquet(parquet_files, engine='pyarrow')
        if columns:
            df = dask_df[columns].compute()
        else:
            df = dask_df.compute()
        df = df.drop_duplicates(keep='last')
        df.index = pd.to_datetime(df.index, utc=True)
        df['exchange'] = exchange
        df['symbol'] = symbol
        df['exchange'] = df.exchange.astype('category')
        df['symbol'] = df.symbol.astype('category')
        df['date'] = df.index

        # log info
        self.log.info(
            f"msg='instrument dataframe loaded' instrument_id='{instrument_id}' length='{len(df)}' start_date='{df.index[0]}' end_date='{df.index[-1]}'"
        )

        # sort index
        return df.sort_index()
