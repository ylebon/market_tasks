import pandas as pd

from core.task_step import TaskStep


class Task(TaskStep):
    """Replay parquet"""

    def run(self, feeds, start_date, end_date=None, cb=lambda x: x):
        end_date = end_date or start_date
        dfs = list()
        for feed in feeds:
            df = self.tasks.histdata.load_parquet.run(feed, start_date, end_date)
            dfs.append(df)
        df = pd.concat(dfs)
        ticks = self.tasks.histdata.dataframe_to_ticks.run(df)
        for tick in ticks:
            cb(tick)
