import pandas as pd

from core.task_step import TaskStep


class Task(TaskStep):

    def run(self, df, **kwargs):
        self.log.info("name='create dataset' name='rolling_data'")
        # parameters
        if isinstance(df, list):
            df = df[0]
        rolling_window = kwargs.get("rolling_window")
        shift = kwargs.get("shift", None)
        profit_pct = kwargs.get("profit_pct", None)

        # prepare columns
        columns = ["bid_price", "ask_price", "bid_qty", "ask_qty"]
        metrics = [
            'rolling_mean', 'rolling_var', 'rolling_coeff', 'rolling_min', "rolling_max", "rolling_open",
            "rolling_close"
        ]
        features = list()
        for column in columns:
            for metric in metrics:
                features.append("{0}.{1}".format(column, metric))

        # resample
        df = df[columns].resample('1s').last().bfill()

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

        dataset['profit.pct'] = (df.bid_price.shift(-shift) - (
                df.ask_price + (df.ask_price * profit_pct))) / df.ask_price
        dataset['profit.sign'] = dataset['profit.pct'].apply(lambda x: int(x > 0))

        self.log.debug("msg='list of columns' columns='{}'".format(list(dataset)))

        dataset = dataset.dropna()
        X = dataset[features]
        y = dataset[['profit.sign']]
        return X, y
