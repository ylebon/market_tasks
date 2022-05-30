import re
from bs4 import BeautifulSoup
from urllib.request import urlopen

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    CoinMarketCap Top volume

    """

    def run(self, topic):
        """
        Get reddit comments

        """
        services_config = ConfigServices.create()
        URL = services_config.get_value('REDDIT.URL')
