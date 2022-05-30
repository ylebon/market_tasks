from referencedata.core.referencedata import Referencedata
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Top Volume

    """

    def run(self, exchange, minimum_volume=1_000_000):
        """Exchange"""
        referencedata = Referencedata.from_url()
        volume = referencedata.get_volume(exchange, update=True)
        volume = filter(lambda x: x['volume'] > minimum_volume, volume)
        volume = map(lambda x: f"{exchange.upper()}_{x['symbol']}", volume)
        return list(volume)
