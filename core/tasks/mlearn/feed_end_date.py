from core.task_step import TaskStep


class Task(TaskStep):
    """Return MLearning end date

    """

    def run(self, config):
        return config["learning"]["data"]["end_date"]