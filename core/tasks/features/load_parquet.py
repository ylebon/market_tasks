import os
from glob import glob

import dask.dataframe as dd
import pandas as pd

from config.core.config_services import ConfigServices
from core.task_step import TaskStep
from utils import time_util


class Task(TaskStep):
    """
    Load feed

    """

    def run(self, instrument_id, start_date=None, end_date=None, date=None, fields=None, feature_list=None):
        config_services = ConfigServices.create()
        parquet_directory = config_services.get_value("FEATURES.LOCAL_DIRECTORY")
        parquet_directory = os.path.expanduser(parquet_directory)

        # feature list
        if feature_list is None:
            feature_list = list()

        self.log.info(
            f"msg='convert parquet files to dataframe' parquet='{parquet_directory}' feature_total='{len(feature_list)}'"
        )

        # Load parquet
        exchange, symbol = instrument_id.split("_", 1)

        df_list = list()

        for feature in feature_list:
            parquet_files = list()
            if start_date and end_date:
                dates = time_util.get_dates(start_date, end_date)
                for date in dates:
                    pattern = f"{parquet_directory}/exchange={exchange}/symbol={symbol}/feature={feature}/date={date.strftime('%Y%m%d')}/*.parquet"
                    parquet_files.extend(glob(pattern))
            elif date:
                pattern = f"{parquet_directory}/exchange={exchange}/symbol={symbol}/feature={feature}/date={date}/*.parquet"
                parquet_files.extend(glob(pattern))
            else:
                pattern = f"{parquet_directory}/exchange={exchange}/symbol={symbol}/feature={feature}/date=*/*.parquet"
                parquet_files.extend(glob(pattern))

            # parquet files loaded
            self.log.info(f"msg='parquet files found' feature='{feature}' pattern='{pattern}' size={len(parquet_files)}")

            # load dataframe
            dask_df = dd.read_parquet(parquet_files, engine='pyarrow')
            df = dask_df.compute()

            # drop duplicates
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)

            # dataframe
            self.log.info(
                f"msg='instrument features dataframe loaded' feature='{feature}' instrument_id='{instrument_id}' length='{len(df)}' start_date='{df.index[0]}' end_date='{df.index[-1]}'"
            )
            df = df.rename(columns={'value': feature})
            df_list.append(df)

        df_all = pd.concat(df_list, axis=1, sort=True)
        self.log.info(
            f"msg='features dataframe loaded' instrument_id={instrument_id} length='{len(df_all)}' start_date='{df_all.index[0]}' end_date='{df_all.index[-1]}'"
        )
        return df_all


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    task = Task("load_feed")
    df = task.run("BINANCE_BTC_USDT")
    mem = df.memory_usage()
    print(mem)
