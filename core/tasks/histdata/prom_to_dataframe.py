import pandas as pd

from core.task_step import TaskStep


class Task(TaskStep):
    """
    Prometheus to dataframe

    """

    def run(self, data):
        """
        Convert metrics values to dataframe

        """
        self.log.info("msg='converting metrics to dataframe'")
        df = pd.DataFrame(data).transpose()
        df.index = pd.to_datetime(df.index, unit='ms')
        df['date'] = df.index.strftime('%Y%m%d')
        return df


if __name__ == "__main__":
    from core.tasks.histdata.load_prom import Task as LoadPromTask

    t = LoadPromTask("load_prom")
    data = t.run("OANDA_WTICO_USD", "2020-01-05", "2020-01-05")

    t_1 = Task("prom_to_dataframe")
    df = t_1.run(data)
    print(df)
