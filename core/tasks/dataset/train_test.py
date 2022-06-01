from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from core.task_step import TaskStep


@dataclass
class Dataset:
    x_train: list
    x_test: list
    y_train: list
    y_test: list


class Task(TaskStep):
    """Train & Test

    """

    def run(self, dataset, test_size=0.3, random_state=0):
        self.log.info("msg='creating dataset from pure dataframe'")
        X, y = dataset
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.ravel(), test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test
