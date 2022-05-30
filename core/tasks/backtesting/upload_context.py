from sqlalchemy import create_engine, MetaData, Table, Column, Integer, JSON, String

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Upload backtesting context

    """
    table_name = 'run_context'

    def run(self, run_id, run_context):
        """
        Upload backesting run id context

        """
        services_config = ConfigServices.create()
        database_url = services_config.get_value("BACKTESTING_DATABASE.URL")

        # create database
        engine = create_engine(database_url)

        # created env context
        res = engine.dialect.has_table(engine, self.table_name)
        meta = MetaData()

        table = Table(
            self.table_name, meta,
            Column('id', Integer, primary_key=True),
            Column('run_id', String, unique=True),
            Column('context', JSON),
        )
        if not res:
            meta.create_all(engine)

        with engine.connect() as conn:
            conn.execute(
                table.insert(),
                run_id=run_id,
                context=run_context
            )


if __name__ == "__main__":
    t = Task("upload_context")
    t.run("ookokok", {"ij": 3, "ji": 5})
