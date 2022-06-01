import os

import dask.dataframe as dd

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Store dataframe to influxdb

    """

    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, df):
        services_config = ConfigServices.create()

        storage_options = dict(
            key=services_config.get_value("AMAZON.ACCESS_KEY"),
            secret=services_config.get_value("AMAZON.SECRET_KEY"),
        )
        bucket = services_config.get_value("FEATURES.S3_BUCKET")

        self.log.info(f"msg='uploading features dataframe to s3' bucket='{bucket}'")
        if len(df.index):
            dd.to_parquet(df, bucket, engine="pyarrow", write_index=True,
                          partition_on=["exchange", "symbol", "feature", "date"],
                          storage_options=storage_options
                          )
            self.log.info(f"msg='dataframe uploaded to S3 bucket' bucket='{bucket}'")
        else:
            self.log.warn("msg='empty dataframe'")
