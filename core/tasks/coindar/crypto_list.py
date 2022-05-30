import requests

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Create database client
    """
    url = "https://coindar.org/api/v2/coins?access_token={token}"
    token = ConfigServices.create().get_value('COINDAR.TOKEN')

    def run(self, token=None):
        token = token or self.__class__.token
        result = requests.get(URL.format(token=token))
        json = result.json()
        return json
