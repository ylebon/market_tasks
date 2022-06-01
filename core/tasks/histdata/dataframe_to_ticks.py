from events.exchange.market_bbo import MarketBBO
from referencedata.core.referencedata import Referencedata
from core.task_step import TaskStep


class Task(TaskStep):
    """Convert dataframe to ticks"""

    def run(self, df):
        self.log.info(f"msg='converting dataframe to ticks'")
        ticks = []
        for index, row in df.iterrows():
            instrument_id = row['exchange'].upper(), tick['symbol'].upper()
            tick['instrument'] = Referencedata().get_instrument(instrument_id)
            tick['strategy_timestamp'] = time.time()
            tick_event = BBO(**tick)
            ticks.append(tick_event)
        return ticks
