import os

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Save dataframe to local parquet

    """

    def run(self, df):
        services_config = ConfigServices.create()
        parquet_local_directory = os.path.abspath(
            os.path.expanduser(services_config.get_value('PARQUET.LOCAL_DIRECTORY'))
        )

        self.log.info(f"msg='converting dataframe to parquet' directory='{parquet_local_directory}'")
        df.to_parquet(
            parquet_local_directory,
            engine="auto",
            write_index=True,
            partition_on=["exchange", "symbol", "date"]
        )
