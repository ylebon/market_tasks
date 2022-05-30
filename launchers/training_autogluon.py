import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
from logbook import Logger
from logbook import StreamHandler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from varatra_features.core.feature_executor import FeatureExecutor
from varatra_features.core.feature_loader import FeatureLoader
from varatra_tasks.core.tasks.features import load_parquet
from varatra_tasks.core.tasks.features import select_importances
from varatra_tasks.core.tasks.features import sync_s3
from varatra_tasks.core.tasks.model import classification_report, confusion_matrix
from varatra_tasks.core.tasks.model import dump_to_s3

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()
from datetime import datetime

DEFAULT_CONFIG = {
    'NN': dict(),
    'GBM': dict(),
    'CAT': dict(),
    'RF': dict(),
    'XT': dict()
}


def launch(instrument_id, target, env_context=None, features=None, shuffle=False, selector=None, estimators=None):
    """
    Run machine learner

    """
    from autogluon import TabularPrediction as task

    # shuffle
    shuffle = os.getenv("SHUFFLE", shuffle)

    # selector
    selector = os.getenv("SELECTOR", selector)

    # selector
    encode_label = os.getenv("ENCODE_LABEL", "false") == "true"

    # estimators
    if estimators is None:
        estimators = list()

    log = Logger("training_autogluon")
    start_time = time.time()
    if env_context:
        os.environ.update(env_context)

    # select features
    features_dir = FeatureExecutor.get_features_dir()
    features = FeatureLoader.filter_features(features_dir, features)
    features_and_target = features + [target]

    # sync parquet
    t_sync_s3 = sync_s3.Task("sync_s3")
    exchange, symbol = instrument_id.split("_", 1)
    parquet_filter = {'exchange': exchange, 'symbol': symbol, 'features': features_and_target}
    t_sync_s3.run(parquet_filter=parquet_filter)

    # drop nan values from dataframe
    t_load_parquet = load_parquet.Task("load_parquet")
    df = t_load_parquet.run(instrument_id, feature_list=features_and_target)

    if encode_label:
        label_encoder = preprocessing.LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target])

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

    # selector
    if selector:
        X = df[[c for c in df.columns if c in features]]
        y = df[target]
        t_select_importances = select_importances.Task("select_importances")
        result = t_select_importances.run(X, y, limit=None, selectors=[selector])
        data = result[selector]
        feature_df = pd.DataFrame(data, columns=["feature", "score"])
        feature_df = feature_df.replace(0, np.nan)
        feature_df.dropna(inplace=True)
        features = feature_df.feature.values.tolist()
        log.info(f"msg='features selection is finished' name='{selector}' length='{len(features)}'")

    # create X, y
    features_valid = [c for c in df.columns if c in features]

    # classifier
    run_id = datetime.now().strftime('%Y%M%d_%H%M%S')
    tmp_run_dir = os.path.join("/tmp", run_id)
    model_file = os.path.join(tmp_run_dir, f"model.joblib")

    # create train and test data
    features_valid_and_target = features_valid + [target]
    df_train, df_test = train_test_split(df[features_valid_and_target], train_size=0.9, test_size=0.1,
                                         shuffle=shuffle)
    log.info(f"msg='dataset splitted to train/test' training_length='{len(df_train)}' testing_length='{len(df_test)}'")

    # disable shuffling
    auto_stack = os.getenv("AUTO_STACK", "false") == "true"
    if auto_stack:
        hyper_parameter_tune = False
    else:
        hyper_parameter_tune = os.getenv("HYPER_PARAMETER_TUNE", "false") == "true"

    # estimators
    log.info(
        f"msg='training started' shuffle='{shuffle}' auto_stack='{auto_stack}' output_directory='{tmp_run_dir}' dataset_length='{len(df)}'"
    )
    if estimators:
        log.info("msg='running with hyperpameters")
        hyperparameters = {key: value for key, value in DEFAULT_CONFIG.items() if key in estimators}
        automl = task.fit(train_data=df_train, auto_stack=auto_stack, label=target, output_directory=tmp_run_dir,
                          eval_metric='precision', hyperparameter_tune=hyper_parameter_tune,
                          hyperparameters=hyperparameters)
    else:
        automl = task.fit(train_data=df_train, auto_stack=auto_stack, label=target, output_directory=tmp_run_dir,
                          eval_metric='precision')
    dump(automl, model_file)

    X_test = df_test[features_valid]
    y_test = df_test[target]

    # classification report
    t_cls_report = classification_report.Task("classification_report")
    try:
        clf_report = t_cls_report.run(automl, X_test, y_test)
        clf_report_dict = clf_report.to_dict()
    except Exception as error:
        log.error(f"msg='failed to generate the classification report' error='{error}'")
        clf_report_dict = dict()

    # confusion matrix
    t_confusion_matrix = confusion_matrix.Task("confusion_matrix")
    try:
        confusion_matrix_report = t_confusion_matrix.run(automl, X_test, y_test)
        confusion_matrix_dict = confusion_matrix_report.to_dict()
    except Exception as error:
        log.error(f"msg='failed to generate the confusion matrix' error='{error}'")
        confusion_matrix_dict = dict()

    # get feature importance
    try:
        feature_importance = automl.feature_importance()
        feature_importance = feature_importance.to_json()
    except Exception as error:
        log.error(f"msg='failed to get feature importance' error='{error}'")

    # get leaderbord
    try:
        feature_importance = automl.feature_importance()
        feature_importance = feature_importance.to_dict()
    except Exception as error:
        log.error(f"msg='failed to get feature importance' error='{error}'")
        feature_importance = dict(error=error)

    # get data
    target_data = FeatureLoader.get_data(features_dir, target)
    target_data = target_data or dict()

    meta = {
        'features': features_valid,
        'target': [{'name': target, 'data': target_data}],
        'duration': time.time() - start_time,
        'feature_importance': feature_importance,
        'dataset': {
            'start_date': str(df.index[0]),
            'end_date': str(df.index[-1]),
            'size': len(df),
            'shuffle': shuffle,
            'training': {
                'start_date': str(df_train.index[0]),
                'end_date': str(df_train.index[-1]),
                'size': len(df_train),
            },
            'testing': {
                'start_date': str(df_test.index[0]),
                'end_date': str(df_test.index[-1]),
                'size': len(df_test),
            }
        },
        'classification_report': clf_report_dict,
        'confusion_matrix': confusion_matrix_dict
    }
    # remove utils directory
    utils_tmp_dir = os.path.join(tmp_run_dir, "utils")
    if os.path.exists(utils_tmp_dir):
        shutil.rmtree(utils_tmp_dir)

    # dump to s3
    t_dump_to_s3 = dump_to_s3.Task("dump_to_s3")
    t_dump_to_s3.run(instrument_id, features_valid, target, 'AUTOGLUON', 'CLASSIFIER', model_file, meta,
                     zip_dir=tmp_run_dir)


if __name__ == "__main__":
    launch("BITSTAMP_BTC_EUR", 'profit__binary__10M')
