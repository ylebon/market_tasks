import os
import pandas as pd

from core.task_step import TaskStep


class Task(TaskStep):
    """Read patterns CSV

    """

    def run(self, config):
        csv_file = os.path.join(config["dir"], config["csv"])
        patterns_df = pd.read_csv(csv_file)
        return patterns_df
