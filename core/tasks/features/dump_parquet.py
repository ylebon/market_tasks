import os

import dask.dataframe as dd

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Store dataframe to influxdb

    """

    def run(self, df):
        services_config = ConfigServices.create()

        local_dir = os.path.expanduser(services_config.get_value("FEATURES.LOCAL_DIRECTORY"))

        os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

        self.log.info(f"msg='dumping features dataframe to local directory' local='{local_dir}'")
        if len(df.index):
            dd.to_parquet(df, local_dir, engine="pyarrow", write_index=True,
                          partition_on=["exchange", "symbol", "feature", "date"],
                          )
        else:
            self.log.warn("msg='empty dataframe'")
