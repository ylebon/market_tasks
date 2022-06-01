import asyncpg
import asyncio
from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Load Features

    """


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    task = Task("load_feed")
    df = task.run("SELECT * FROM features ORDER BY id DESC LIMIT 1000;")
