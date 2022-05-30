from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tasks.core.task_step import TaskStep


class Task(TaskStep):
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
        'svc': SVC,
        'quadratic': QuadraticDiscriminantAnalysis,
    }

    def run(self, name):
        return self.__class__.ESTIMATORS[name]()
