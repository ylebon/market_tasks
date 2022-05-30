from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Return estimator X

    """

    def run(self, config):
        return config["learning"]["dataset"]["name"]