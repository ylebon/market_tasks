import time

import numpy as np

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    def __init__(self, name):
        TaskStep.__init__(self, name)

    def create_pattern(self, series):
        """create pattern"""
        pattern_list = list()
        start_time = time.time()
        if isinstance(series, list):
            mean = np.mean(series)
        else:
            mean = np.mean(series.values)
        for index in self.indexes:
            s = f"{int(series[index] > mean)}111"
            pattern_list.append(s)
        # pattern bin
        pattern_bin = ".".join(pattern_list)
        # convert to hex
        pattern_hex = hex(int((pattern_bin.replace(".", "")), 2))
        # convert to int
        pattern_int = int(pattern_hex, 0)
        duration = time.time() - start_time
        self.log.debug(f"msg='pattern generated' pattern={pattern_int} duration='{duration}'")
        return pattern_int
