from tasks.core.task_step import TaskStep


class Task(TaskStep):

    def run(self):
        return {
            'n_estimators': [1, 5, 10, 50],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 5],
            'criterion': ["gini", "entropy"],
            'max_features': ["sqrt", "log2"],
            'max_depth': [1, 5, 10, None]
        }