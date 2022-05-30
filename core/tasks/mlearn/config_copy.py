import shutil

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Copy configuration file

    """
    def run(self, config_file, output_directory):
        shutil.copy(config_file, output_directory)
