import os

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Return config

    """

    def run(self, config):
        services_config = ConfigServices.create()
        conf_directory = services_config.get_value("PATTERN.CONF_DIRECTORY")
        return os.path.join(conf_directory, f"{config}.toml")
