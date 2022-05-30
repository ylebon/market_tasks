import os
import toml

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Load pipeline config

    """

    def run(self, config_file):
        services_config = ConfigServices.create()
        pipeline_config_directory = services_config.get_value("PIPELINE.CONFIG_DIRECTORY")
        self.log.info("msg='reading pipeline config' config='{0}'".format(config_file))
        config_file = os.path.join(pipeline_config_directory, config_file)
        return toml.load(config_file)
