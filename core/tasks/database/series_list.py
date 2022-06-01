from core.task_step import TaskStep


class Task(TaskStep):
    """
    List database series

    """

    def run(self, db_client):
        result = db_client.query("SHOW MEASUREMENTS")
        series = [x['name'] for x in list(result.get_points())]
        return series
