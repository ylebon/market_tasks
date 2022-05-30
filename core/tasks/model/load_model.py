import os
from glob import glob
import joblib
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Load model from directory

    """

    def run(self, directory, model_name=None):
        self.log.info("msg='loading model from directory' directory={}".format(directory))
        if model_name is None:
            models = glob(os.path.join(directory, "*.model"))
            model_name = models[0].split(".")[0]
        model_file = os.path.join(directory, "{0}.model".format(model_name))
        model = joblib.load(model_file)
        return model
