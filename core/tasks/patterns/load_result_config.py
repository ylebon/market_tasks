import os
import toml

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Read patters config file

    """

    def run(self, config_file):
        with open(config_file, "r") as fr:
            config = toml.loads(fr.read())
            config['dir'] = os.path.dirname(config_file)
            config['parquet'] = os.path.abspath(os.path.join(config["dir"], config["parquet"]))
            return config
