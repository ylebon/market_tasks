import pandas as pd

from tasks.core.task_step import TaskStep


class Task(TaskStep):

    def run(self, df, **kwargs):
        self.log.info("name='create dataset' name='rolling_data'")
        # parameters
        features = kwargs.get("features")
        rolling_window = kwargs.get("rolling_window")
        target = kwargs.get("target", None)
        shift = kwargs.get("shift", None)

        # prepare columns
        columns = list(set([x.split(".")[0] for x in features]))

        # resample
        df = df[columns].resample('1s').last().bfill()
        self.log.info("msg='resampling done!'")
        # rolling

        rolling_data = df.rolling(rolling_window, 600)
        rolling_mean = rolling_data.mean()

        rolling_q_25 = rolling_data.quantile(0.25)
        rolling_q_75 = rolling_data.quantile(0.75)
        rolling_coeff = (rolling_q_75 - rolling_q_25) / (rolling_q_75 + rolling_q_25)

        rolling_var = rolling_data.var()

        rolling_min = rolling_data.min()
        rolling_max = rolling_data.max()
        rolling_open = rolling_data.apply(lambda x: x[0])
        rolling_close = rolling_data.apply(lambda x: x[-1])

        # mean
        dataset = pd.DataFrame()
        for feature in features:
            try:
                column, metric = feature.split(".")
            except ValueError:
                dataset[feature] = df[feature]
                continue

            if metric == "rolling_mean":
                dataset['{0}.{1}'.format(column, metric)] = rolling_mean[column]

            elif metric == "rolling_var":
                dataset['{0}.{1}'.format(column, metric)] = rolling_var[column]

            elif metric == "rolling_coeff":
                dataset['{0}.{1}'.format(column, metric)] = rolling_coeff[column]

            elif metric == "rolling_min":
                dataset['{0}.{1}'.format(column, metric)] = rolling_min[column]

            elif metric == "rolling_max":
                dataset['{0}.{1}'.format(column, metric)] = rolling_max[column]

            elif metric == "rolling_open":
                dataset['{0}.{1}'.format(column, metric)] = rolling_open[column]

            elif metric == "rolling_close":
                dataset['{0}.{1}'.format(column, metric)] = rolling_close[column]

        if target:
            dataset['profit.pct'] = (df.bid_price.shift(-shift) - (
                    df.ask_price + (df.ask_price * 0.001))) / df.ask_price
            dataset['profit.sign'] = dataset['profit.pct'].apply(lambda x: int(x > 0))

        self.log.debug("msg='list of columns' columns={}".format(list(dataset)))

        dataset = dataset.dropna()
        return dataset
