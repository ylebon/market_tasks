import toml

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Read patters config file

    """

    def run(self, config_file):
        with open(config_file, "r") as fr:
            config = toml.loads(fr.read())
            return config
