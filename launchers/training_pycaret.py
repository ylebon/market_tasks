import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from logbook import Logger
from logbook import StreamHandler
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from features.core.feature_executor import FeatureExecutor
from features.core.feature_loader import FeatureLoader
from core.tasks.features import load_parquet
from core.tasks.features import select_importances
from core.tasks.features import sync_s3
from core.tasks.model import classification_report, confusion_matrix
from core.tasks.model import dump_to_s3

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()


def launch(instrument_id, target, env_context=None, features=None, shuffle=False, selector=None, estimators=None):
    """
    Run machine learner

    """

    # estimators
    if estimators is None:
        estimators = list()

    log = Logger("training_pycaret")
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
    df_train, df_test = train_test_split(df[features_valid_and_target], train_size=0.75, test_size=0.25,
                                         shuffle=shuffle)
    log.info(f"msg='dataset splitted to train/test' training_length='{len(df_train)}' testing_length='{len(df_test)}'")

    # estimators
    log.info(
        f"msg='training started' shuffle='{shuffle}' output_directory='{tmp_run_dir}' dataset_length='{len(df)}'"
    )

    # automl
    automl = setup(df_train, target=target)
    print(automl)

    # models
    models = compare_models(sort='Precision')
    print(models)

    # estimators
    for estimator in estimators:
        tuned_clf = tune_model(estimator)
        print(tuned_clf)
        prediction = predict_model(tuned_clf)
        print(prediction)

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

    # get leaderbord
    try:
        feature_importance = automl.feature_importance()
        feature_importance = feature_importance.to_dict()
    except Exception as error:
        log.error(f"msg='failed to get feature importance' error='{error}'")
        feature_importance = dict(error=error)

    meta = {
        'features': features_valid,
        'target': target,
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
    t_dump_to_s3.run(instrument_id, features_valid, target, 'PYCARET', 'CLASSIFIER', model_file, meta,
                     zip_dir=tmp_run_dir)


if __name__ == "__main__":
    launch("BITSTAMP_BTC_EUR", 'profit__binary__10M')
