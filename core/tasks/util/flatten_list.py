from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Flatten list

    """

    def run(self, data):
        """
        Run Task

        """
        result = list()
        for d in data:
            for i in d:
                if i not in result:
                    result.append(i)
        return result
