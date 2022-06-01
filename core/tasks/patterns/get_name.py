from core.task_step import TaskStep


class Task(TaskStep):
    """Read patterns name

    """

    def run(self, config):
        return config["name"]
