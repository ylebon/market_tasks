from sklearn.feature_selection import SelectKBest, chi2

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Select features
    """

    # chi2, f_classif, mutual_info_classif

    def run(self, dataset, score_func="chi2", number_features=5):
        self.log.info("msg='selecting the best features'")
        X, y = dataset
        select_k_best_classifier = SelectKBest(chi2, k=number_features).fit(X, y)
        mask = select_k_best_classifier.get_support()
        new_features = X.columns[mask]
        return X[new_features], y
