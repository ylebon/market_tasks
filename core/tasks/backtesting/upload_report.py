import time

from sqlalchemy import create_engine

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Upload backtesting report

    """

    def run(self, df):
        """
        Create bucket

        """
        services_config = ConfigServices.create()
        database_url = services_config.get_value("BACKTESTING_DATABASE.URL")

        # create database
        engine = create_engine(database_url)

        # created at
        df['created_at'] = time.time()
        try:
            df.to_sql('backtesting', engine, if_exists='append', index=True)
        except Exception:
            df.to_sql('backtesting', engine, if_exists='replace', index=True)


if __name__ == "__main__":
    t = Task("create_client")
    t.run("oks")
