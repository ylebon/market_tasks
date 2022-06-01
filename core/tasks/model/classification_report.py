import pandas as pd
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report

from core.task_step import TaskStep


class Task(TaskStep):
    """Return classification report

    """

    def run(self, clf, X_test, y_test):
        print("Detailed classification report:")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        clf_report = classification_report_imbalanced(y_true, y_pred)
        print(clf_report)
        return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
