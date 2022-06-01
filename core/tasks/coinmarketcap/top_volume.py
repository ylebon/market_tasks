import re
from bs4 import BeautifulSoup
from urllib.request import urlopen

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    CoinMarketCap Top volume

    """

    def normalize(self, exchange, pair):
        """
        Normalize

        """
        return exchange.upper() + '_' + pair.replace('/', '_')

    def run(self, top, exchange="binance", pattern="ETH"):
        """
        Run top volume task

        """
        services_config = ConfigServices.create()
        base_url = services_config.get_value('COINMARKETCAP.URL')
        html = urlopen(f"{base_url}/exchanges/{exchange}")
        soup = BeautifulSoup(html.read(), 'html.parser')
        div = soup.find(id="exchange-markets")
        rows = div.find("tbody").find_all('tr')
        result = []
        for row in rows:
            cols = row.find_all('td')
            pair, volume = self.normalize(exchange, cols[2].text), cols[3].text
            if re.match(pattern, pair):
                result.append(pair)
            if len(result) == top:
                break

        return result
