import os

import dask.dataframe as dd

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    def run(self, parquet_directory=None):
        """sync amazon s3 data"""
        parquet_directory = parquet_directory or ConfigServices.create().get_value("PARQUET.LOCAL_DIRECTORY")
        parquet_directory = os.path.expanduser(parquet_directory)
        self.log.info("msg='convert parquet files to dataframe parquet={0}".format(parquet_directory))
        dataframe = dd.read_parquet(parquet_directory, engine='pyarrow')
        return dataframe


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    task = Task("load_directory")
    df = task.run()
