from core.task_step import TaskStep


class Task(TaskStep):
    def __init__(self, name):
        TaskStep.__init__(self, name)

    def create_pattern(self, series):
        """create pattern"""
        pattern_list = list()
        median = series.quantile(0.5)
        for index in self.indexes:
            s = f"{int(series[index] > median)}111"
            pattern_list.append(s)
        # pattern bin
        pattern_bin = ".".join(pattern_list)
        # convert to hex
        pattern_hex = hex(int((pattern_bin.replace(".", "")), 2))
        # convert to int
        return int(pattern_hex, 0)
