from core.task_step import TaskStep


class Task(TaskStep):

    def run(self, config):
        return config["learning"]["dataset"]["train_test"]["x"]