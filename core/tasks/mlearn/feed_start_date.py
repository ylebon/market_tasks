from core.task_step import TaskStep


class Task(TaskStep):
    """Get start date

    """

    def run(self, config):
        return config["learning"]["data"]["start_date"]