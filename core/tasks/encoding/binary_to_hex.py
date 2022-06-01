import string

from core.task_step import TaskStep


class Task(TaskStep):
    """
    Binary to HEX
    """

    def run(self, s):
        assert isinstance(s, str), f"{s} is not a string"
        assert isinstance(len(s), 4), f"{s} length is not equal to 4"
        hex(string.atoi(s.replace(" ", ""), 2))
