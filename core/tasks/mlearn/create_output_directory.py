import os

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Create output directory

    """

    def run(self, dataset_name, feed, start_date, end_date):
        output_directory = os.path.join(
            ConfigServices.create().get_value("MODEL.LOCAL_DIRECTORY"),
            f"{dataset_name.upper()}_{feed}_{start_date}_{end_date}"
        )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        return output_directory
