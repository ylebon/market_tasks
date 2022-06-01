from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.task_step import TaskStep


class Task(TaskStep):
    """
    Training task

    """

    ESTIMATORS = {
        'nearestcentroid': NearestCentroid,
        'randomforest': RandomForestClassifier,
        'logicregression': LogisticRegression,
        'mlpclassifier': MLPClassifier,
        'kneighbors': KNeighborsClassifier,
        'adaboost': AdaBoostClassifier,
        'bagging': BaggingClassifier,
        'guassiannb': GaussianNB,
        'sgdclassifier': SGDClassifier,
        'decisiontree': DecisionTreeClassifier,
        'gradientboostclassifier': GradientBoostingClassifier,
        'gaussianprocess': GaussianProcessClassifier,
        'linearregression': LinearRegression,
        'svc': SVC,
        'quadratic': QuadraticDiscriminantAnalysis,
    }

    PARAMETERS = {
        'linearregression': {

        },
        'randomforest': {
            'n_estimators': [1, 5, 10, 50],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 5],
            'criterion': ["gini", "entropy"],
            'max_features': ["sqrt", "log2"],
            'max_depth': [1, 5, 10, None]
        }
    }

    def run(self, name, X, y, split_ratio=0.3, shuffle=False):
        self.log.info(f"msg='running training' name='{name}''")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=0, shuffle=shuffle
        )

        self.log.info(f"msg='dataset size' training_len='{len(X_train)}' testing_len='{len(X_test)}'")

        # grid search model
        grid_search = GridSearchCV(Task.ESTIMATORS[name](), Task.PARAMETERS[name])
        model = grid_search.fit(X_train, y_train.ravel())

        # return model
        return dict(
            model=model,
            X_train=X_train,
            X_test=X_test,
            X_scaler=None,
            y_train=y_train,
            y_test=y_test,
            y_scaler=None,
            estimator=name,
            params={}
        )
