import pandas as pd
from datetime import timedelta

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Predict from dataset

    """

    def run(self, df, element, max_size):
        row = pd.Series(element.__dict__)
        index = element.marketdata_date
        try:
            df.loc[index] = row
        except ValueError:
            df = pd.DataFrame(columns=element.__dict__.keys())
            df.loc[index] = row
        except AttributeError:
            df = pd.DataFrame(columns=element.__dict__.keys())
            df.loc[index] = row
        finally:
            begin_index = df.index[-1] - timedelta(seconds=max_size)
            df = df[begin_index < df.index]
            return df
