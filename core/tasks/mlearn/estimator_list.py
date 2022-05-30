from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Return estimators

    """

    def run(self, config):
        return config["learning"]["estimator"]