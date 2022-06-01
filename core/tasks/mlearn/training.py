from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from mlearn.core.mlearn_model import MLearnModel
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Training task

    """
    ESTIMATORS = {
        'randomforestclassifier': RandomForestClassifier,
        'kneighborsclassifier': KNeighborsClassifier,
        'gradientboostclassifier': GradientBoostingClassifier,
        'gaussianprocessclassifier': GaussianProcessClassifier,
        'mlpclassifier': MLPClassifier,
        'adaboostclassifier': AdaBoostClassifier,
        'baggingclassifier': BaggingClassifier,
        'sgdclassifier': SGDClassifier,
        'decisiontreeclassifier': DecisionTreeClassifier,
        'logisticregression': LogisticRegression,
        'nearestcentroid': NearestCentroid,
        'guassiannb': GaussianNB,
        'svc': SVC,
        'quadratic': QuadraticDiscriminantAnalysis,
        'linearregression': LinearRegression,
        'svr': SVR,
        'randomforestregressor': RandomForestRegressor,
        'adaboostregressor': AdaBoostRegressor
    }

    def run(self, estimator_name, X, y, split_ratio=0.3, shuffle=False, params={}):
        """
        Run training

        """
        self.log.info(f"msg='running training' name='{estimator_name}' params={params}")

        X_scaler = None
        y_scaler = None

        # split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=0, shuffle=shuffle
        )

        self.log.info(f"training_length='{len(X_train)}' testing_length='{len(X_test)}'")

        # grid search cv needs an estimator name
        grid_search_cv = False
        if estimator_name == "gridsearchcv":
            grid_search_cv = True
            estimator_name = params['name']
            del params['name']
            estimator = GridSearchCV(Task.ESTIMATORS[estimator_name](), params)
        else:
            estimator = Task.ESTIMATORS[estimator_name](**params)

        # fit estimator
        model = estimator.fit(X_train, y_train.ravel())

        # y prediction
        y_pred = model.predict(X_test)

        # return model
        return MLearnModel.create(
            estimator_name,
            params,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            y_pred,
            X_scaler,
            y_scaler,
            grid_search_cv=grid_search_cv
        )
