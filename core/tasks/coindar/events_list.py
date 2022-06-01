import requests
from datetime import datetime

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """Events list
    """
    url = "https://coindar.org/api/v2/events?access_token={token}" \
          "&page=1&page_size={page_size}" \
          "&filter_date_start={filter_date_start}" \
          "&filter_date_end={filter_date_end}" \
          "&filter_coins={filter_coins}" \
          "&filter_tags={filter_tags}" \
          "&sort_by=views" \
          "&order_by=1"
    token = ConfigServices.create().get_value('COINDAR.TOKEN')

    def run(self, token=None, page_size=30, filter_date_start=None, filter_date_end=None, filter_tags=None,
            filter_coins=None):
        """Filter events list"""
        token = token or self.__class__.token
        filter_date_start = filter_date_start or datetime.now().strftime("%Y-%m-%d")
        filter_date_end = filter_date_end or datetime.now().strftime("%Y-%m-%d")
        url = self.url.format(
            token=token, page_size=page_size,
            filter_date_start=filter_date_start,
            filter_date_end=filter_date_end,
            filter_coins=filter_coins,
            filter_tags=filter_tags
        )
        result = requests.get(url)
        json = result.json()
        return json
