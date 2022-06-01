import joblib

from core.task_step import TaskStep


class Task(TaskStep):
    """Dump model

    """

    def run(self, model, model_file):
        joblib.dump(model, model_file)