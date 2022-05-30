import traceback

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Predict from dataset

    """
    def __init__(self, name):
        TaskStep.__init__(self, name)
        self.last_index = None

    def run(self, model, dataset, estimator_x=None, x_scaler=None):
        self.log.info("msg='predict from dataset")
        if estimator_x:
            dataset = dataset[estimator_x]
        if len(dataset) > 1 and dataset.index[-1] != self.last_index:
            self.last_index = dataset.index[-1]
            x = dataset.iloc[-2].values.reshape(1, -1)
            try:
                if x_scaler:
                    x = x_scaler.transform(x)
                prediction = model.predict(x)[0]
                self.log.info(
                    f"[PREDICTION!!!] prediction={prediction} x={x}  size={len(dataset)} date={self.last_index}"
                )
                return prediction
            except Exception as error:
                self.log.error("msg='failed to predict' x={} error={}".format(x, error))
                self.log.error(traceback.format_exc())
                return 0
