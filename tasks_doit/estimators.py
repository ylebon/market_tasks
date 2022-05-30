from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from dataclasses import dataclass
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier, XGBRegressor
from dask_ml.linear_model import LinearRegression
from dask_ml.linear_model import PoissonRegression

ESTIMATORS = {
    'LogisticRegression': (LogisticRegression, {

    }, 'ScikitClassifier'),
    'RandomForestClassifier': (RandomForestClassifier, {
        'n_estimators': [5, 10, 20],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2']
    }, 'ScikitClassifier'),
    'GradientBoostingClassifier': (GradientBoostingClassifier, {
        'loss': ["deviance", "exponential"],
        'n_estimators': [100, 200, 300, 400, 500]
    }, 'ScikitClassifier'),
    'BaggingClassifier': (BaggingClassifier, {

    }, 'ScikitClassifier'),
    'GaussianProcessClassifier': (GaussianProcessClassifier, {

    }, 'ScikitClassifier'),
    'GaussianNB': (GaussianNB, {

    }, 'ScikitClassifier'),
    'KNeighborsClassifier': (KNeighborsClassifier, {
         'n_neighbors': [5, 10, 15],
         'algorithm': ["auto", "ball_tree", "kd_tree"]
    }, 'ScikitClassifier'),
    'AdaBoostClassifier': (AdaBoostClassifier, {
        'algorithm': ["SAMME", "SAMME.R"],
        'n_estimators': [10, 50],
        'learning_rate': [0.1, 0.5, 1.0]
    }, 'ScikitClassifier'),
    'ExtraTreesClassifier': (ExtraTreesClassifier, {
        'n_estimators': [10, 50],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]

    }, 'ScikitClassifier'),

    'NearestCentroid': (NearestCentroid, {

    }, 'ScikitClassifier'),

    'MultinomialNB': (MultinomialNB, {

    }, 'ScikitClassifier')
    ,

    'DecisionTreeClassifier': (DecisionTreeClassifier, {

    }, 'ScikitClassifier'),

    'MLPClassifier': (MLPClassifier(), {
        'hidden_layer_sizes': [20, 50, 100],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }, 'ScikitClassifier'),

    'SVC': (SVC, {
        #'gamma': ['rbf', 'poly', 'sigmoid', 'scale'],
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

    }, 'ScikitClassifier'),
    'SGDClassifier': (SGDClassifier, {

    }, 'ScikitClassifier'),
    'RandomForestRegressor': (RandomForestRegressor, {

    }, 'ScikitClassifier'),

    # ARIMA autoregression
    'AutoRegression': (AR, {}, 'ARIMA'),
    'ARMA': (ARMA, {'order': (0, 1)}, 'ARIMA'),
    'ARIMA': (ARIMA, {'order': (1, 1, 1)}, 'ARIMA'),
    'SARIMAX': (SARIMAX, {'order': (1, 1, 1)}, 'ARIMA'),
    'SARIMAX_SEASONAL': (SARIMAX, {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 1)}, 'ARIMA'),
    'XGBClassifier': (XGBClassifier, {
        #'min_child_weight': [1.0, 5.0, 10.0],
        #'gamma': [0.5, 1.0, 1.5, 2, 5],
        #'subsample': [0.6, 0.8, 1.0],
        #'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }, 'ScikitClassifier'),
    'LinearRegression': (LinearRegression, {}, 'ScikitLinear'),
    'PoissonRegression': (PoissonRegression, {}, 'ScikitLinear'),
}

@dataclass
class Estimator(object):
    """
    Estimator

    """
    clf: int = None
    name: int = None
    category: str = None
    params: int = None

    @classmethod
    def from_name(cls, name):
        """
        Create from name

        """
        estimator_cls, estimator_params, estimator_cat = ESTIMATORS[name]
        estimator = Estimator(clf=estimator_cls, name=name, category=estimator_cat, params=estimator_params,)
        return estimator



