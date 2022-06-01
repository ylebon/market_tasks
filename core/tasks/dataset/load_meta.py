import os
import toml

from core.task_step import TaskStep


class Task(TaskStep):
    """Load params

    """

    def run(self, directory, instrument):
        self.log.info("msg='loading meta information from directory' directory={}".format(directory))
        meta_file = os.path.join(directory, f"{instrument}.meta")
        return toml.load(meta_file)