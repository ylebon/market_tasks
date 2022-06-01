import pandas as pd
from tabulate import tabulate

from core.task_step import TaskStep


class Task(TaskStep):
    """Return classification report

    """

    def run(self, clf, X_test, y_test):
        print("Detailed confusion matrix:")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        confusion_matrix = pd.crosstab(
            y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True
        )
        data = tabulate(confusion_matrix, headers=confusion_matrix.columns.values.tolist(), tablefmt="grid")
        print(data)
        return confusion_matrix
