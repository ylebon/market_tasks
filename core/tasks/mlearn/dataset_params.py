from core.task_step import TaskStep


class Task(TaskStep):
    """Return dataset parameters

    """

    def run(self, config):
        return config["learning"]["dataset"]["params"]