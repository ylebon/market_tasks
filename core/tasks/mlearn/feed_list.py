from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Return feeds

    """

    def run(self, config):
        return config["learning"]["data"]["feeds"]