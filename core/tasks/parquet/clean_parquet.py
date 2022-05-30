import os
import shutil
from glob import glob

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Cleaning parquet data for this date

    """

    def run(self, exchange, symbol, date, parquet_directory=None, confirm=True):
        """sync amazon s3 data"""
        parquet_directory = parquet_directory or ConfigServices.create().get_value("PARQUET.LOCAL_DIRECTORY")
        parquet_directory = os.path.expanduser(parquet_directory)
        self.log.info("msg='cleaning parquet data' parquet_directory='{}' date='{}'".format(parquet_directory, date))
        pattern = "{}/exchange={}/symbol={}/date={}".format(parquet_directory, exchange, symbol, date)
        directories = glob(pattern)
        for directory in directories:
            print(directory)
        if confirm:
            c = input('Do you want to delete?')
            if c in ["y", "yes"]:
                for directory in directories:
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
