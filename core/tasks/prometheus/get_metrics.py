import requests

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Get prometheus metrics

    """

    def run(self, pattern, start_date="2020-01-02T00:00:00Z", end_date="2020-01-02T23:59:59Z"):
        services_config = ConfigServices.create()
        url = services_config.get_value('PROMETHEUS.URL')
        self.log.info(f"msg='loading metrics' pattern='{pattern}'")
        query_url = f'{url}/api/v1/series?match[]={{__name__=~"{pattern}"}}&start={start_date}&end_date={end_date}'
        response = requests.get(query_url)
        data = response.json()
        metrics = list(set(["_".join(x['__name__'].split('_')[:3]) for x in data['data']]))
        return metrics


if __name__ == "__main__":
    t = Task("get_metrics")
    r = t.run("OANDA.*")
