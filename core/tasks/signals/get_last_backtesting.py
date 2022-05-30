import pandas as pd
import sqlalchemy as db

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Return last sharpe ratio

    """

    engine = None

    def run(self, signal_name, instrument_id):
        """
        Return last sharpe ratio

        """
        if not self.engine:
            services_config = ConfigServices.create()
            url = services_config.get_value("BACKTESTING_DATABASE.URL")
            self.engine = db.create_engine(url)
        df = pd.read_sql_table(signal_name, self.engine)
        df = df[df.instrument_id == instrument_id]
        df['created_at'] = pd.to_datetime(df.created_at, unit='s')
        df.sort_values(by='created_at', inplace=True)
        if len(df.index):
            result = df.iloc[-1].to_dict()
            return result
        else:
            return dict()


if __name__ == '__main__':
    t = Task("get_sharpe_ratio")
    r = t.run("bollinger_bands__percentile", "OANDA_EUR_GBP")
    print(r)
