import time

import pandas as pd
from pytz import timezone
from tzlocal import get_localzone

from core.task_step import TaskStep


class Task(TaskStep):
    timezone = timezone(get_localzone().zone)

    def run(self, points):
        """convert points to dataframe"""
        self.log.info("msg='converting points to dataframe'")
        df = pd.DataFrame(points)
        if len(df):
            # Use UTC time
            df['time'] = pd.to_datetime(df.time, utc=True)
            df['exchange'] = df['exchange'].str.upper()
            df['date'] = df['time'].dt.strftime('%Y%m%d')
            df['symbol'] = df['symbol'].str.upper()
            df['database_timestamp'] = df['time'].apply(self.to_timestamp)
            df = df.set_index('time', drop=True)
            df = df.sort_index()
            return df
        else:
            return pd.DataFrame()

    def to_timestamp(self, x):
        """Transform to timestamp"""
        return time.mktime(x.timetuple())
