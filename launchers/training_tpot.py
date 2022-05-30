import copy
import os
import sys

import time
from joblib import dump
from logbook import StreamHandler
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from varatra_features.core.feature_executor import FeatureExecutor
from varatra_features.core.feature_loader import FeatureLoader
from varatra_tasks.core.tasks.features import load_parquet
from varatra_tasks.core.tasks.features import sync_s3
from varatra_tasks.core.tasks.model import classification_report, confusion_matrix
from varatra_tasks.core.tasks.model import dump_to_s3

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()
from datetime import datetime
from dask.distributed import Client
from logbook import Logger
from tabulate import tabulate
import numpy as np

ESTIMATORS_CONFIG = {
    # Classifiers
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },

    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    'lightgbm.LGBMClassifier': {
        'boosting_type': ['gbdt', 'dart'],
        'min_child_samples': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
        'num_leaves': [2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250, 500],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'subsample': [0.7, 0.9, 1.0],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000]
    }

}

DEFAULT_CONFIG = {
    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }
}


def launch(instrument_id, target, env_context=None, features=None, use_dask=False, estimators=None, shuffle=False):
    """
    Run machine learner

    """
    # logger
    log = Logger("training_tpot")

    # start date
    start_date = time.time()

    # parameters
    shuffle = os.getenv("SHUFFLE", shuffle)
    tpot_config = os.getenv("TPOT_CONFIG", None)
    use_dask = use_dask or os.getenv("USE_DASK", "false") == "true"
    n_jobs = int(os.getenv("N_JOBS", 1))

    # use dask
    if use_dask:
        client = Client()

    # estimators
    if estimators is None:
        estimators = list()

    if not tpot_config:
        tpot_config = copy.deepcopy(DEFAULT_CONFIG)
        if not estimators:
            tpot_config.update(ESTIMATORS_CONFIG)
        else:
            for key, value in ESTIMATORS_CONFIG.items():
                if key in estimators:
                    tpot_config[key] = value

    if env_context:
        os.environ.update(env_context)

    # select features
    features_dir = FeatureExecutor.get_features_dir()
    features = FeatureLoader.filter_features(features_dir, features)
    features_and_target = features + [target]

    # sync parquet
    log.info(f"msg='loading features and target' features_target='{features_and_target}'")
    t_sync_s3 = sync_s3.Task("sync_s3")
    exchange, symbol = instrument_id.split("_", 1)
    parquet_filter = {'exchange': exchange, 'symbol': symbol, 'features': features_and_target}
    t_sync_s3.run(parquet_filter=parquet_filter)

    # drop nan values from dataframe
    t_load_parquet = load_parquet.Task("load_parquet")
    df = t_load_parquet.run(instrument_id, feature_list=features_and_target)

    # clean data
    log.info(f"msg='dataset length after loading' length='{len(df)}'")
    len_after_load = len(df)
    # print nan
    df_count = df.count().sort_values(ascending=False).to_frame()
    print(tabulate(df_count, headers=['count']))
    # drop nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    log.info(f"msg='dataset length after dropping nan' length='{len(df)}' dropped='{len_after_load - len(df)}'")

    # create X, y
    features_valid = [c for c in df.columns if c in features]
    X = df[features_valid]
    y = df[target]

    # create train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42,
                                                        shuffle=shuffle)
    log.info(
        f"msg='dataset splitted to train/test' training_length='{len(X_train)}' testing_length='{len(X_test)}' shuffle='{shuffle}'")

    tpot = TPOTClassifier(
        generations=5,
        population_size=5,
        verbosity=2,
        scoring='precision',
        random_state=42,
        max_eval_time_mins=10,
        config_dict=tpot_config,
        use_dask=use_dask,
        n_jobs=n_jobs,
        memory=None
    )
    tpot.fit(X_train, y_train)

    # run id
    tmp_run_dir = f"/tmp/{datetime.now().strftime('%Y%M%d_%H%M%S')}"
    os.makedirs(tmp_run_dir)
    output_file = os.path.join(tmp_run_dir, f"script.py")
    model_file = os.path.join(tmp_run_dir, f"model.joblib")

    # dump output
    tpot.export(output_file)
    dump(tpot.fitted_pipeline_, model_file)

    # features importances
    try:
        exctracted_best_model = tpot.fitted_pipeline_.steps[-1][1]
        exctracted_best_model.fit(X_train, y_train)
        feature_importance = list(zip(list(features_valid), exctracted_best_model.feature_importances_))
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    except Exception as error:
        log.error(f"msg='feature_importance' error='{error}'")
        feature_importance = dict(error=error)

    # classification report
    t_cls_report = classification_report.Task("classification_report")
    clf_report = t_cls_report.run(tpot.fitted_pipeline_, X_test, y_test)

    # classification report
    t_confusion_matrix = confusion_matrix.Task("confusion_matrix")
    confusion_matrix_report = t_confusion_matrix.run(tpot.fitted_pipeline_, X_test, y_test)

    # script content
    with open(output_file, 'r') as fr:
        script_content = fr.read()

    # duration
    duration = time.time() - start_date
    meta = {
        'features': features_valid,
        'target': target,
        'estimators': estimators,
        'duration': duration,
        'feature_importance': feature_importance,
        'dataset': {
            'start_date': str(X.index[0]),
            'end_date': str(X.index[-1]),
            'size': len(X),
            'shuffle': shuffle,
            'training': {
                'start_date': str(X_train.index[0]),
                'end_date': str(X_train.index[-1]),
                'size': len(X_train),
            },
            'testing': {
                'start_date': str(X_test.index[0]),
                'end_date': str(X_test.index[-1]),
                'size': len(X_test),
            }
        },
        'classification_report': clf_report.to_dict(),
        'confusion_matrix': confusion_matrix_report.to_dict(),
        'script_content': script_content
    }

    # dump to s3
    t_dump_to_s3 = dump_to_s3.Task("dump_to_s3")
    t_dump_to_s3.run(instrument_id, features_valid, target, 'TPOT', 'CLASSIFIER', model_file, meta)


if __name__ == "__main__":
    launch("BINANCE_BTC_USDT", 'profit__binary__10M')
