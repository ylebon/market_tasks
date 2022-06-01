from core.task_step import TaskStep


class Task(TaskStep):
    def run(self, df):
        """dask dataframe"""
        df['seq_previous'] = df.shift(1).seq
        df["gap"] = (df.seq - 1) != df.seq_previous
        return df[df.gap]

