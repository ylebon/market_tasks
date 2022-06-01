import os
import time

import dask.dataframe as dd

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Store dataframe to influxdb

    """

    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, df):
        start_time = time.time()
        services_config = ConfigServices.create()

        storage_options = dict(
            key=services_config.get_value("AMAZON.ACCESS_KEY"),
            secret=services_config.get_value("AMAZON.SECRET_KEY")
        )
        bucket = services_config.get_value("PARQUET.S3_BUCKET")

        self.log.info(f"msg='uploading dataframe to s3 bucket' bucket='{bucket}'")
        if len(df.index):
            dd.to_parquet(df, bucket, engine="auto", write_index=True, partition_on=["exchange", "symbol", "date"],
                          storage_options=storage_options)
        else:
            self.log.warn("msg='empty dataframe'")

        # dataframe uploaded to s3
        self.log.info(f"msg='dataframe uploaded to s3' duration='{time.time()-start_time} secs'")
