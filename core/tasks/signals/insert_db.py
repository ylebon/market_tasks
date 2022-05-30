from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Signal database insert

    """
    database_client = None

    def run(self, signal):
        """
        Insert signal

        """

        if not self.__class__.database_client:
            services_config = ConfigServices()
