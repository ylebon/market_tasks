import itertools

from core.task_step import TaskStep


class Task(TaskStep):
    """
    Return parameters combination

    """

    def run(self, params):
        x = []
        keys = []
        result = []
        for key, values in params.items():
            x.append(values)
            keys.append(key)
        params = list(itertools.product(*x))
        for values in params:
            result.append(dict(zip(keys, values)))
        return result