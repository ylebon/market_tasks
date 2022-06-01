import joblib

from core.task_step import TaskStep


class Task(TaskStep):
    """Dump scaler

    """

    def run(self, scaler, scaler_file):
        joblib.dump(model, scaler_file)