import os
import sys

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))

sys.path.append(BASE_DIR)

from tabulate import tabulate
import numpy as np
import os
import tpot
import subprocess
from varatra_features.core.feature_executor import FeatureExecutor
import pandas as pd
from sklearn.model_selection import train_test_split as scikit_train_test_split
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from varatra_mlearn.core.mlearn_storage import MLearnStorage
from varatra_backtesting.core.backesting import SignalBackTesting
from doit.tools import config_changed

from varatra_utils import time_util
from doit import get_var
from datetime import datetime, timedelta

from imblearn.metrics import classification_report_imbalanced
from xgboost import XGBClassifier
from dask_ml import preprocessing as dask_preprocessing

import toml
import ast
import re
from varatra_tasks.core.tasks_factory import TasksFactory
from varatra_features.core.features_loader import FeaturesLoader
from varatra_features.core.feature import Feature
import dask.dataframe as dd
import multiprocessing as mp

from varatra_patterns.recursionlimit import recursionlimit
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

DEFINITIONS_FILE = os.path.join(BASE_DIR, "varatra_features", "data")
OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "varatra_logs", "auto_tpot")

from dask.distributed import Client

DEFAULT_PARAMETERS = {
    "STORE": True,
    "INSTRUMENTS": "OANDA_EUR_USD",
    "OUTPUT_DIRECTORY": OUTPUT_DIRECTORY,
    "DSET_START_DATE": -1,
    "DSET_END_DATE": 0,
    "BTEST_START_DATE": -1,
    "BTEST_END_DATE": 0,
    "DATABASE_NAME": "TEST",
    "DATABASE_HOST": "51.15.106.57:5432",
    "WINDOW": 600,
    # "PATTERN_FEATURES": ".*profit.*,.*coeff.*,.*rolling_mean_diff.*,.*rolling_median_diff.*",
    "PATTERN_FEATURES": ".*profit.*,.*std.*,.*coeff.*",
    "TARGET_LINEAR": ["profit_range_resampled_backfill_{window}"],
    "TARGET_CLASSIFIER": ["profit_range_resampled_backfill_{window}_range"],
    "BTEST_LIFETIME": "{window}",
    "FEATURES_COUNT": 10,
    "BUCKET": 'varatra-models-dev',
    "MODE": "DEV",
    "DASK_SCHEDULER": "51.15.54.161:8786"
}

PARAMETERS = dict()
CONTEXT = dict()


def start_logging():
    from logbook import FileHandler
    log_file = os.path.join(OUTPUT_DIRECTORY, "auto_tpot.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    FileHandler(log_file, level='INFO').push_application()


def create_feature(parquet_file, feature):
    ddf = dd.read_parquet(parquet_file, engine='pyarrow')
    df = ddf[list(feature.feeds.keys())]
    feature_data = feature.script.create_dataset(df).compute()
    return feature.alias, feature_data


def load_parameters():
    """Setup parameters

    """

    # filter feeds
    tasks = TasksFactory.create()

    PARAMETERS['DATABASE_NAME'] = get_var(
        'db_name',
        os.environ.get('DATABASE_NAME', DEFAULT_PARAMETERS["DATABASE_NAME"])
    )

    PARAMETERS['DATABASE_HOST'] = get_var(
        'db_host',
        DEFAULT_PARAMETERS["DATABASE_HOST"]
    )

    PARAMETERS['OUTPUT_DIRECTORY'] = get_var(
        'output_directory',
        DEFAULT_PARAMETERS["OUTPUT_DIRECTORY"]
    )

    # DASK SCHEDULER
    PARAMETERS['DASK_SCHEDULER'] = get_var(
        'dask_scheduler',
        os.environ.get('DASK_SCHEDULER', DEFAULT_PARAMETERS["DASK_SCHEDULER"])
    )

    # INSTRUMENT
    PARAMETERS['INSTRUMENTS'] = get_var(
        'instruments',
        os.environ.get('INSTRUMENTS', DEFAULT_PARAMETERS["INSTRUMENTS"])
    )

    # STORE MODEL
    PARAMETERS['STORE'] = get_var(
        'store',
        DEFAULT_PARAMETERS["STORE"]
    )

    # WINDOW
    PARAMETERS['WINDOW'] = int(get_var(
        'window',
        os.environ.get('WINDOW', DEFAULT_PARAMETERS["WINDOW"])
    )
    )

    # NUMBER OF FEATURES
    PARAMETERS['FEATURES_COUNT'] = int(get_var(
        'feat_count',
        DEFAULT_PARAMETERS["FEATURES_COUNT"])
    )

    # PARAMETERS MODE
    PARAMETERS["MODE"] = get_var('mode', os.environ.get('MODE', DEFAULT_PARAMETERS["MODE"]))

    # DATASET REPORT NAME
    PARAMETERS['REPORT_NAME'] = get_var(
        'report_name',
        os.environ.get(
            'REPORT_NAME',
            'auto_learning_tpot_classifier_{mode}'.format(mode=PARAMETERS['MODE'].lower())
        )
    )

    # DATASET
    PARAMETERS['DSET_START_DATE'] = time_util.get_date(int(get_var(
        'dset_start_date',
        os.environ.get('DSET_START_DATE', DEFAULT_PARAMETERS["DSET_START_DATE"])))
    )
    PARAMETERS['DSET_END_DATE'] = time_util.get_date(int(get_var(
        'dset_end_date',
        os.environ.get('DSET_END_DATE', DEFAULT_PARAMETERS["DSET_END_DATE"])))
    )

    # BACKTEST
    PARAMETERS['BTEST_START_DATE'] = time_util.get_date(int(get_var(
        'btest_start_date',
        os.environ.get('BTEST_START_DATE', DEFAULT_PARAMETERS["BTEST_START_DATE"]))
    )
    )
    PARAMETERS['BTEST_END_DATE'] = time_util.get_date(int(
        get_var(
            'btest_end_date',
            os.environ.get('BTEST_END_DATE', DEFAULT_PARAMETERS["BTEST_END_DATE"])))
    )
    PARAMETERS['BTEST_LIFETIME'] = int(
        DEFAULT_PARAMETERS["BTEST_LIFETIME"].format(window=PARAMETERS['WINDOW'] + 1)
    )

    # CHECK BTEST LAST DURATION
    PARAMETERS['BTEST_LAST_DURATION'] = int(get_var(
        'btest_last_duration',
        os.environ.get('BTEST_LAST_DURATION', 0)
    ))

    # CHECK DSET LAST DURATION
    PARAMETERS['DSET_LAST_DURATION'] = int(get_var(
        'dset_last_duration',
        os.environ.get('DSET_LAST_DURATION', 0)
    ))

    # CHECK BTEST
    now = datetime.now()
    fmt = '%Y-%m-%d %H:%M:%S'

    if PARAMETERS['BTEST_LAST_DURATION']:
        PARAMETERS['BTEST_START_DATE'] = (now - timedelta(hours=PARAMETERS['BTEST_LAST_DURATION'])).strftime(fmt)
        PARAMETERS['BTEST_END_DATE'] = now.strftime(fmt)

    # CHECK DSET
    if PARAMETERS['DSET_LAST_DURATION']:
        PARAMETERS['DSET_START_DATE'] = (now - timedelta(hours=PARAMETERS['DSET_LAST_DURATION'])).strftime(fmt)
        PARAMETERS['DSET_END_DATE'] = now.strftime(fmt)

    # TARGET
    PARAMETERS['TARGET_CLASSIFIER'] = [x.format(window=PARAMETERS['WINDOW']) for x in
                                       DEFAULT_PARAMETERS["TARGET_CLASSIFIER"]]
    PARAMETERS['TARGET_LINEAR'] = [x.format(window=PARAMETERS['WINDOW']) for x in DEFAULT_PARAMETERS["TARGET_LINEAR"]]

    # PATTERN_FEATURES
    PARAMETERS["PATTERN_FEATURES"] = get_var(
        'pattern_features',
        os.environ.get('PATTERN_FEATURES', DEFAULT_PARAMETERS["PATTERN_FEATURES"])
    ).split(",")

    # BUCKET
    PARAMETERS["BUCKET"] = 'varatra-models-{}'.format(PARAMETERS['MODE'].lower())

    # INSTRUMENTS
    if not get_var('pattern'):
        PARAMETERS['INSTRUMENTS'] = get_var(
            'instruments',
            os.environ.get('INSTRUMENTS', DEFAULT_PARAMETERS["INSTRUMENTS"])
        ).split(",")

    else:
        PARAMETERS["INSTRUMENTS"] = list()
        db_client = tasks.database.create_client.run()
        pattern = get_var('pattern', '.*')
        for feed in tasks.database.series_list.run(db_client):
            if re.match(pattern, feed):
                PARAMETERS["INSTRUMENTS"].append(feed)
            else:
                pass

    print("-" * 20)
    print("{:<30} {:<15}".format('Name', 'Value'))
    for k, v in PARAMETERS.items():
        print("{:<30} {:<15}".format(k, str(v)))
    print("-" * 20)


# start logging
start_logging()

# load parameters
load_parameters()


def task_create_directory():
    """
    Create output directory

    """

    def create_directory(instrument, targets):
        # create directory
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # open started txt
        with open(targets[0], 'w') as fw:
            fw.write("")

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f"{instrument}",
            'actions': [(create_directory, [instrument])],
            'targets': [
                f'{output_directory}/CREATED.txt'
            ],
        }


def task_create_features_params():
    """
    Create output directory

    """

    def create_features_params(instrument, features_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # create directory
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # create directory
        with open(f'{output_directory}/features_params.toml', "w") as out_file:
            toml.dump(features_params, out_file)

        # dataset parameters
        dataset_params = {
            'TARGET_CLASSIFIER': PARAMETERS['TARGET_CLASSIFIER'],
            'TARGET_LINEAR': PARAMETERS['TARGET_LINEAR']
        }
        with open(f'{output_directory}/dataset_params.toml', "w") as out_file:
            toml.dump(dataset_params, out_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # features loader
        features_loader = FeaturesLoader.from_file(DEFINITIONS_FILE)
        features = features_loader.filter(PARAMETERS["PATTERN_FEATURES"])

        features_params = {
            'START_DATE': PARAMETERS['DSET_START_DATE'],
            'END_DATE': PARAMETERS['DSET_END_DATE'],
            'FEED': instrument,
            'PATTERNS': PARAMETERS["PATTERN_FEATURES"],
            'FEATURES': features,
            'COUNT': PARAMETERS['FEATURES_COUNT'],
        }

        yield {
            'name': f"{instrument}_feaures_params",
            'actions': [(create_features_params, [instrument, features_params])],
            'file_dep': [
                f'{output_directory}/CREATED.txt'
            ],
            'uptodate': [config_changed(features_params)],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/features_params.toml",
                f"{output_directory}/dataset_params.toml",
            ],
        }


def task_create_training_params():
    """
    Create output directory

    """

    def create_training_params(instrument, training_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        with open(f'{output_directory}/training_params.toml', "w") as out_file:
            toml.dump(training_params, out_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        # training parameters
        training_params = {
            'NAME': f"{instrument}_training_params",
            'PARAMS': {},
            'MODEL': f"{output_directory}/{instrument}_tpot.model",
        }
        yield {
            'name': f"{instrument}_training_params",
            'actions': [(create_training_params, [instrument, training_params])],
            'file_dep': [
                f'{output_directory}/CREATED.txt'
            ],
            'uptodate': [config_changed(training_params)],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/training_params.toml",
            ],
        }


def task_create_storage_params():
    """
    Create storage parameters

    """

    def create_storage_params(instrument, storage_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        with open(f'{output_directory}/storage_params.toml', "w") as out_file:
            toml.dump(storage_params, out_file)

    # storage parameters
    storage_params = {
        'BUCKET': PARAMETERS['BUCKET']
    }

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f'{instrument}_storage_params',
            'actions': [(create_storage_params, [instrument, storage_params])],
            'file_dep': [
                f'{output_directory}/CREATED.txt'
            ],
            'uptodate': [config_changed(storage_params)],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/storage_params.toml",
            ],
        }


def task_create_report_params():
    """
    Create report parameters

    """

    def create_report_params(instrument, report_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        with open(f'{output_directory}/report_params.toml', "w") as out_file:
            toml.dump(report_params, out_file)

    # storage parameters
    report_params = {
        'REPORT_NAME': PARAMETERS['REPORT_NAME']
    }

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f'{instrument}_report_params',
            'actions': [(create_report_params, [instrument, report_params])],
            'file_dep': [
                f'{output_directory}/CREATED.txt'
            ],
            'uptodate': [config_changed(report_params)],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/report_params.toml",
            ],
        }


def task_create_testing_params():
    """
    Create output directory

    """

    def create_testing_params(instrument, testing_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        with open(f'{output_directory}/testing_params.toml', "w") as out_file:
            toml.dump(testing_params, out_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        # testing parameters
        testing_params = {
            'MODEL': f"{output_directory}/{instrument}_tpot.model",
            'Y_PRED': f"{output_directory}/{instrument}_y_pred.csv",
        }
        yield {
            'name': f"{instrument}",
            'actions': [(create_testing_params, [instrument, testing_params])],
            'file_dep': [
                f'{output_directory}/CREATED.txt'
            ],
            'uptodate': [config_changed(testing_params)],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/testing_params.toml",
            ],
        }


def task_create_backtesting_params():
    """
    Create output directory

    """

    def create_params(instrument, backtesting_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        with open(f'{output_directory}/backtesting_params.toml', "w") as out_file:
            toml.dump(backtesting_params, out_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # backtesting parameters
        model_file = f"{instrument}_tpot.model"
        features_file = f"{instrument}_tpot.features"
        scaler_file = f"{instrument}.scaler"

        backtesting_params = {
            'START_DATE': PARAMETERS['BTEST_START_DATE'],
            'END_DATE': PARAMETERS['BTEST_END_DATE'],
            'FEEDS': [instrument],
            'MODEL_FILE': model_file,
            'FEATURES_FILE': features_file,
            'SCALER_FILE': scaler_file,
            'LIFETIME': PARAMETERS['BTEST_LIFETIME'],
        }

        yield {
            'name': f"{instrument}",
            'actions': [(create_params, [instrument, backtesting_params])],
            'file_dep': [
                f'{output_directory}/CREATED.txt'
            ],
            'uptodate': [config_changed(backtesting_params)],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/backtesting_params.toml",
            ],
        }


def task_load_data():
    """
    Task load data

    """

    tasks = TasksFactory.create()

    def load_data(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # load parameters
        features_params = toml.load(f"{output_directory}/features_params.toml")
        features_names = features_params.get("FEATURES")
        start_date = features_params.get("START_DATE")
        end_date = features_params.get("END_DATE")

        features_loader = FeaturesLoader.from_file(DEFINITIONS_FILE)
        features = list()
        data = dict()
        for feature_name in features_names:
            params = {'feed': instrument}
            feature = Feature.from_name(feature_name, definitions=features_loader, params=params)
            features.append(feature)
            for feed_alias, feed in feature.feeds.items():
                if feed not in data:
                    feed_source, feed_field = feed.split(".")
                    points = tasks.histdata.load_influx.run(
                        feed_source,
                        start_date,
                        end_date,
                        fields=feed_field.lower()
                    )
                    # create series
                    series = pd.DataFrame(list(points)).set_index('time')
                    series.index = pd.to_datetime(series.index)

                    # save series
                    data[feed] = series

        # dump features
        joblib.dump(features, targets[0])

        # resample
        df = pd.concat(data.values(), axis=1, sort=False)
        df = df.resample("1s").bfill().dropna()

        # dump parquet file
        df.to_parquet(targets[1], engine='pyarrow')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f'{instrument}_data',
            'actions': [
                (load_data, [instrument])
            ],
            'targets': [
                f"{output_directory}/features.joblib",
                f"{output_directory}/data.parquet"
            ],
            'file_dep': [
                f"{output_directory}/features_params.toml"
            ],
            'verbosity': 2
        }


def task_create_features():
    """
    Create features

    """

    def create_features(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # load parameters
        features_params = toml.load(f"{output_directory}/features_params.toml")
        parquet_file = f"{output_directory}/data.parquet"

        # feature
        features = joblib.load(f"{output_directory}/features.joblib")

        #  run features engineering
        pool = mp.Pool()
        results = [pool.apply_async(create_feature, (parquet_file, f)) for f in features]
        pool.close()
        pool.join()

        # dataset
        dataset = pd.concat([r.get()[1] for r in results], axis=1)
        dataset.columns = [r.get()[0] for r in results]
        dataset.dropna(inplace=True)

        # create features parquet file
        parquet_file = os.path.join(output_directory, 'features.parquet')
        dataset.to_parquet(parquet_file, engine='pyarrow')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f'{instrument}_features',
            'actions': [
                (create_features, [instrument])
            ],
            'targets': [
                f"{output_directory}/features.parquet"
            ],
            'file_dep': [
                f"{output_directory}/features_params.toml",
                f"{output_directory}/features.joblib",
                f"{output_directory}/data.parquet"
            ],
            'verbosity': 2
        }


def task_create_dataset():
    """
    Create dataset

    """

    def create_dataset(instrument, targets):
        # load parameters
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        dataset_params = toml.load(f"{output_directory}/dataset_params.toml")
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        target_classifier = dataset_params['TARGET_CLASSIFIER']
        target_linear = dataset_params['TARGET_LINEAR']

        df = pd.read_parquet(f"{output_directory}/features.parquet", engine='pyarrow')
        all_features = list(df)
        X = df[[x for x in all_features if 'profit' not in x]]
        y_classifier = df[target_classifier]
        y_linear = df[target_linear]

        X.to_parquet(targets[0], engine='pyarrow')
        y_classifier.to_parquet(targets[1], engine='pyarrow')
        y_linear.to_parquet(targets[2], engine='pyarrow')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f'{instrument}_dataset',
            'actions': [(create_dataset, [instrument])],
            'file_dep': [
                f"{output_directory}/dataset_params.toml",
                f"{output_directory}/features.parquet"
            ],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/X.parquet",
                f"{output_directory}/y_classifier.parquet",
                f"{output_directory}/y_linear.parquet"
            ],
        }


def task_scale_x():
    """
    Scale X dataset

    """

    def scale_x(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        df = pd.read_parquet(f"{output_directory}/X.parquet", engine='pyarrow')

        # create scaler
        scaler = dask_preprocessing.MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

        # dump targets
        df.to_parquet(targets[0], engine='pyarrow')

        # dump scaler
        joblib.dump(scaler, targets[1])

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f"{instrument}_scale_x",
            'actions': [(scale_x, [instrument])],
            'file_dep': [
                f"{output_directory}/X.parquet"
            ],
            'targets': [
                f"{output_directory}/X_scaled.parquet",
                f"{output_directory}/X.scaler"
            ],
            'verbosity': 2,
        }


def task_select_xgb_features():
    """
    Select features

    """

    def select_xgb_features(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # load parameters
        features_params = toml.load(f'{output_directory}/features_params.toml')

        # load dataset, should be selected from SCALED
        X = pd.read_parquet(f"{output_directory}/X_scaled.parquet", engine='pyarrow')
        y = pd.read_parquet(f"{output_directory}/y_classifier.parquet", engine='pyarrow')

        # select features
        nbr_of_features = features_params['COUNT']

        model = XGBClassifier()
        model.fit(X.values, y.values.ravel())

        # plot features importances
        x = model.get_booster().get_score(importance_type='weight')

        # select N important features
        features_importances = list(zip(list(X), model.feature_importances_))
        features_importances = sorted(features_importances, key=lambda x: x[1], reverse=True)
        selected_features = [x[0] for x in features_importances[:nbr_of_features]]

        # selected features
        print(f"  -> xgb features: {selected_features}")
        with open(f"{output_directory}/xgb_features.txt", 'w') as fw:
            fw.write(str(selected_features))

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f'{instrument}_select_xgb_features',
            'actions': [(select_xgb_features, [instrument])],
            'file_dep': [
                f"{output_directory}/X_scaled.parquet",
                f"{output_directory}/features_params.toml"
            ],
            'targets': [
                f"{output_directory}/xgb_features.txt",
            ],
            'verbosity': 2,
        }


def task_select_kbest_features():
    """
    Select features

    """

    def select_kbest_features(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # load parameters
        features_params = toml.load(f'{output_directory}/features_params.toml')

        # load dataset, should be selected from SCALED
        X = pd.read_parquet(f"{output_directory}/X_scaled.parquet", engine='pyarrow')
        y = pd.read_parquet(f"{output_directory}/y_classifier.parquet", engine='pyarrow')

        # select features
        nbr_of_features = features_params['COUNT']

        features = SelectKBest(score_func=chi2, k=int(nbr_of_features)).fit(X, y)

        # select N important features
        features_importances = list(zip(list(X), features.scores_))
        features_importances = sorted(features_importances, key=lambda x: x[1], reverse=True)
        selected_features = [x[0] for x in features_importances[:nbr_of_features]]

        # selected features
        print(f"  -> kbest features: {selected_features}")
        with open(f"{output_directory}/kbest_features.txt", 'w') as fw:
            fw.write(str(selected_features))

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f'{instrument}_select_kbest_features',
            'actions': [(select_kbest_features, [instrument])],
            'file_dep': [
                f"{output_directory}/X_scaled.parquet",
                f"{output_directory}/features_params.toml"
            ],
            'targets': [
                f"{output_directory}/kbest_features.txt",
            ],
            'verbosity': 2,
        }


def task_select_rfe_features():
    """
    Select features

    """

    def select_rfe_features(instrument, targets):
        """
        Select RFE features

        """
        try:
            output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

            # load parameters
            features_params = toml.load(f'{output_directory}/features_params.toml')

            # load dataset, should be selected from SCALED
            X = pd.read_parquet(f"{output_directory}/X_scaled.parquet", engine='pyarrow')
            y = pd.read_parquet(f"{output_directory}/y_classifier.parquet", engine='pyarrow')

            # select features
            nbr_of_features = features_params['COUNT']

            model = LogisticRegression()
            rfe = RFE(model, int(nbr_of_features)).fit(X, y)

            # select N important features
            features_importances = list(zip(list(X), rfe.ranking_))
            features_importances = sorted(features_importances, key=lambda x: x[1], reverse=False)
            selected_features = [x[0] for x in features_importances[:nbr_of_features]]

        except Exception as error:
            print("Error: RFE features selection {0}".format(error))
            selected_features = []
        finally:
            # selected features
            print(f"  -> rfe features: {selected_features}")
            with open(f"{output_directory}/rfe_features.txt", 'w') as fw:
                fw.write(str(selected_features))

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f'{instrument}_select_rfe_features',
            'actions': [(select_rfe_features, [instrument])],
            'file_dep': [
                f"{output_directory}/X_scaled.parquet",
                f"{output_directory}/features_params.toml"
            ],
            'targets': [
                f"{output_directory}/rfe_features.txt",
                # f"{output_directory}/features_importances.png"
            ],
            'verbosity': 2,
        }


def task_select_features():
    """
    Select features

    """

    def select_features(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # load parameters
        features_params = toml.load(f'{output_directory}/features_params.toml')

        # select features
        nbr_of_features = features_params['COUNT']

        with open(f"{output_directory}/xgb_features.txt", "r") as fr:
            xgb_features = ast.literal_eval(fr.read())

        with open(f"{output_directory}/kbest_features.txt", "r") as fr:
            kbest_features = ast.literal_eval(fr.read())

        with open(f"{output_directory}/rfe_features.txt", "r") as fr:
            rfe_features = ast.literal_eval(fr.read())

        # selected_features = reduce(np.intersect1d, (xgb_features, rfe_features))

        selected_features = xgb_features

        # selected features
        print(f"  -> features selected: {selected_features}")
        X = pd.read_parquet(f"{output_directory}/X.parquet", engine='pyarrow')
        try:
            X[selected_features].to_parquet(targets[0], engine='pyarrow')
        except Exception as error:
            print("Error: {}".format(error))

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f'{instrument}_select_features',
            'actions': [(select_features, [instrument])],
            'file_dep': [
                f"{output_directory}/X.parquet",
                f"{output_directory}/features_params.toml",
                f"{output_directory}/rfe_features.txt",
                f"{output_directory}/kbest_features.txt",
                f"{output_directory}/xgb_features.txt"
            ],
            'targets': [
                f"{output_directory}/X_selected.parquet",
            ],
            'verbosity': 2,
        }


def task_split_train_test():
    """
    Split dataset

    """

    def split_train_test(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        if os.path.join(f"{output_directory}/X_selected.parquet"):
            X = pd.read_parquet(f"{output_directory}/X_selected.parquet", engine='pyarrow')
            y_classifier = pd.read_parquet(f"{output_directory}/y_classifier.parquet", engine='pyarrow')
            y_linear = pd.read_parquet(f"{output_directory}/y_linear.parquet", engine='pyarrow')

            X_train, X_test, y_train_classifier, y_test_classifier = scikit_train_test_split(
                X, y_classifier, test_size=0.3, random_state=0, shuffle=False
            )

            X_train, X_test, y_train_linear, y_test_linear = scikit_train_test_split(
                X, y_linear, test_size=0.3, random_state=0, shuffle=False
            )

            X_train.to_parquet(targets[0], engine='pyarrow')
            X_test.to_parquet(targets[1], engine='pyarrow')

            y_train_classifier.to_parquet(targets[2], engine='pyarrow')
            y_test_classifier.to_parquet(targets[3], engine='pyarrow')

            y_train_linear.to_parquet(targets[4], engine='pyarrow')
            y_test_linear.to_parquet(targets[5], engine='pyarrow')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f'{instrument}_split_train_test',
            'actions': [(split_train_test, [instrument])],
            'verbosity': 2,
            'file_dep': [f"{output_directory}/X_selected.parquet"],
            'targets': [
                f"{output_directory}/X_train.parquet",
                f"{output_directory}/X_test.parquet",
                f"{output_directory}/y_train_classifier.parquet",
                f"{output_directory}/y_test_classifier.parquet",
                f"{output_directory}/y_train_linear.parquet",
                f"{output_directory}/y_test_linear.parquet"
            ]
        }


def task_training():
    """
    Train model

    """

    def training(instrument, targets):
        # output directory
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # load parameters
        training_params = toml.load(f'{output_directory}/training_params.toml')

        X_train = pd.read_parquet(f"{output_directory}/X_train.parquet", engine='pyarrow')
        y_train = pd.read_parquet(f"{output_directory}/y_train_classifier.parquet", engine='pyarrow')

        X_values = X_train.astype(np.float32).values
        y_values = y_train.values.ravel()

        client = Client(PARAMETERS['DASK_SCHEDULER'])

        clf = tpot.TPOTClassifier(
            generations=10,
            population_size=10,
            cv=2,
            n_jobs=-1,
            random_state=0,
            verbosity=0,
            scoring='f1',
            # config_dict=tpot.config.classifier_config_dict_light,
            use_dask=True,
        )

        # fit
        clf.fit(X_values, y_values)

        # print score
        X_test = pd.read_parquet(f"{output_directory}/X_test.parquet", engine='pyarrow')
        y_test = pd.read_parquet(f"{output_directory}/y_test_classifier.parquet", engine='pyarrow')
        print(f"  -> score: {clf.score(X_test, y_test)}")

        model_file = training_params.get("MODEL")
        print(f"  -> model file: {model_file}")

        # pipeline file
        print(f"  -> pipeline file: {targets[0]}")
        clf.export(targets[0])

    # instruments
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f"{instrument}/training",
            'actions': [(training, [instrument])],
            'file_dep': [
                f'{output_directory}/training_params.toml',
                f"{output_directory}/X_train.parquet",
                f"{output_directory}/y_train_classifier.parquet",
                f"{output_directory}/y_train_linear.parquet"
            ],
            'targets': [
                f"{output_directory}/{instrument}_pipeline_tpot_auto.py",
            ],
            'verbosity': 2
        }


def task_pipeline():
    """
    Pipeline task

    """

    def pipeline(instrument, targets):
        # output directory
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        X = pd.read_parquet(f"{output_directory}/X_selected.parquet", engine='pyarrow')
        y = pd.read_parquet(f"{output_directory}/y_classifier.parquet", engine='pyarrow')

        df = pd.concat([X, y], axis=1)
        df = df.rename({'profit_range_resampled_backfill_600_range': 'target'})
        df['target'] = df['profit_range_resampled_backfill_600_range']
        del df['profit_range_resampled_backfill_600_range']
        # df = df.reset_index(drop=True)

        # dataset to CSV
        df.to_csv(f"{output_directory}/dataset.csv", index=False)

        # replace
        with open(f"{output_directory}/{instrument}_pipeline_tpot_auto.py", 'r') as fr:
            content = fr.read()
            content = content.replace('PATH/TO/DATA/FILE', f"{output_directory}/dataset.csv")
            content = content.replace('COLUMN_SEPARATOR', ",")

        with open(targets[2], 'w') as fw:
            content += "from sklearn.externals import joblib"
            content += "\n"
            content += f"joblib.dump(exported_pipeline, '{output_directory}/{instrument}_tpot.model')"
            fw.write(content)

        # execute
        output = subprocess.check_output(['python', targets[2]])

    # instruments
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f"{instrument}_pipeline",
            'actions': [(pipeline, [instrument])],
            'file_dep': [
                f"{output_directory}/X_selected.parquet",
                f"{output_directory}/y_classifier.parquet",
                f"{output_directory}/{instrument}_pipeline_tpot_auto.py",
            ],
            'targets': [
                f"{output_directory}/{instrument}_tpot.model",
                f"{output_directory}/dataset.csv",
                f"{output_directory}/{instrument}_pipeline_tpot.py",
            ],
            'verbosity': 2
        }


def task_testing():
    """
    Create model prediction from test data

    """

    def testing(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        testing_params = toml.load(f'{output_directory}/testing_params.toml')
        X_test = pd.read_parquet(f"{output_directory}/X_test.parquet", engine='pyarrow')

        y_test = pd.read_parquet(f"{output_directory}/y_test_classifier.parquet", engine='pyarrow')
        y_train = pd.read_parquet(f"{output_directory}/y_train_classifier.parquet", engine='pyarrow')

        model_file = testing_params.get("MODEL")
        print(f"  -> model file: {model_file}")
        model = joblib.load(model_file)

        X_test_values = X_test.astype(np.float32).values
        y_train_values = y_train.astype(np.float32).values
        y_test_values = y_test.astype(np.float32).values

        y_pred = model.predict(X_test_values)

        # created predicted
        df = pd.DataFrame()
        df['predicted'] = y_pred
        df.index = X_test.index
        df.to_parquet(targets[0], engine='pyarrow')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f'{instrument}_testing',
            'actions': [(testing, [instrument])],
            'verbosity': 2,
            'file_dep': [
                f'{output_directory}/testing_params.toml',
                f"{output_directory}/X_test.parquet",
                f"{output_directory}/{instrument}_tpot.model"
            ],
            'targets': [
                f"{output_directory}/y_pred.parquet"
            ]
        }


def task_performance():
    """
    Measure peformance with test data

    """

    def measure_performance(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        y_test = pd.read_parquet(f"{output_directory}/y_test_classifier.parquet")
        y_pred = pd.read_parquet(f"{output_directory}/y_pred.parquet")

        y_test_values = y_test.values.flatten()
        y_pred_values = y_pred.values.flatten()

        # confusion matrix
        confusion_matrix = pd.crosstab(
            y_test_values, y_pred_values, rownames=['Actual'], colnames=['Predicted'], margins=True
        )
        data = tabulate(confusion_matrix, headers=confusion_matrix.columns.values.tolist(), tablefmt="grid")
        print(data)

        # save confusion matrix
        confusion_matrix.to_csv(targets[1])

        report = classification_report(y_test_values, y_pred_values, output_dict=True)
        pd.DataFrame(report).to_csv(targets[2])

        # report
        report = classification_report_imbalanced(y_test_values, y_pred_values)
        with open(f"{output_directory}/tpot_report.txt", "w") as fw:
            print(report)
            fw.write(report)

    # estimators
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f'{instrument}_performance',
            'actions': [(measure_performance, [instrument])],
            'verbosity': 2,
            'file_dep': [
                f"{output_directory}/y_pred.parquet",
                f"{output_directory}/y_test_linear.parquet",
                f"{output_directory}/y_test_classifier.parquet"

            ],
            'targets': [
                f"{output_directory}/tpot_report.txt",
                f"{output_directory}/learning_crosstab.csv",
                f"{output_directory}/learning_perf.csv"

            ]
        }


def task_create_features_executor():
    """
    Create features executor

    """

    def create_features_executor(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        df = pd.read_parquet(f"{output_directory}/X_selected.parquet")
        features_names = list(df)
        print(f"     -> total number of features created: {len(features_names)}")
        feature_executor = FeatureExecutor.from_names(features_names, definitions_file=DEFINITIONS_FILE,
                                                      params={'feed': instrument})
        joblib.dump(feature_executor, targets[0])

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        yield {
            'name': f'{instrument}_features_executor',
            'actions': [(create_features_executor, [instrument])],
            'file_dep': [
                f"{output_directory}/X_selected.parquet"
            ],
            'targets': [
                f"{output_directory}/{instrument}_tpot.features"
            ],
        }


def task_upload_features_executor_to_s3():
    """
    Upload features executor to S3

    """

    def upload_features_executor_to_s3(instrument):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        storage_params = toml.load(f"{output_directory}/storage_params.toml")

        if PARAMETERS.get('STORE'):
            features_file = f"{output_directory}/{instrument}_tpot.features"
            print(f"  -> upload features: {features_file} to {storage_params.get('BUCKET')}")
            mlearn_storage = MLearnStorage.create()
            mlearn_storage.upload(features_file, bucket=storage_params.get('BUCKET'))

            # uploaded
            target = f"{output_directory}/{instrument}_tpot.features_uploaded"
            with open(target, "w") as fw:
                fw.write('UPLOADED')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f'{instrument}_features_executor_to_s3',
            'actions': [(upload_features_executor_to_s3, [instrument])],
            'file_dep': [
                f"{output_directory}/{instrument}_tpot.features",
                f"{output_directory}/storage_params.toml"
            ],
            'targets': [
                f"{output_directory}/{instrument}_tpot.features_uploaded",
            ],
            'verbosity': 2
        }


def task_upload_model_to_s3():
    """
    Upload model file to S3

    """

    def upload_model_to_s3(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        storage_params = toml.load(f"{output_directory}/storage_params.toml")

        if PARAMETERS.get('STORE'):
            model_file = f'{output_directory}/{instrument}_tpot.model'
            print(f"  -> upload model: {model_file}")
            mlearn_storage = MLearnStorage.create()
            mlearn_storage.upload(model_file, bucket=storage_params.get('BUCKET'))
            target = f"{output_directory}/{instrument}_tpot.model_uploaded"
            with open(target, "w") as fw:
                fw.write('UPLOADED')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f"{instrument}_upload_model_to_s3",
            'actions': [(upload_model_to_s3, [instrument])],
            'file_dep': [
                f"{output_directory}/{instrument}_tpot.model",
                f"{output_directory}/storage_params.toml"
            ],
            'targets': [
                f"{output_directory}/{instrument}_tpot.model_uploaded"
            ],
            'verbosity': 2
        }


def task_upload_pipeline():
    """
    Upload pipeline to S3

    """

    def upload_pipeline_to_s3(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        storage_params = toml.load(f"{output_directory}/storage_params.toml")

        if PARAMETERS.get('STORE'):
            pipeline_file = f"{output_directory}/{instrument}_pipeline_tpot.py"
            print(f"  -> upload pipeline: {pipeline_file}")
            mlearn_storage = MLearnStorage.create()
            mlearn_storage.upload(pipeline_file, bucket=storage_params.get('BUCKET'))
            with open(targets[0], "w") as fw:
                fw.write('UPLOADED')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f"{instrument}_upload_pipeline_to_s33",
            'actions': [(upload_pipeline_to_s3, [instrument])],
            'file_dep': [
                f"{output_directory}/{instrument}_tpot.model",
                f"{output_directory}/storage_params.toml"
            ],
            'targets': [
                f"{output_directory}/{instrument}_pipeline_tpot_auto.py_uploaed",
            ],
            'verbosity': 2
        }


def task_backtesting():
    """
    Run backtesting

    """

    def backtesting(instrument):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # load parameters
        backtesting_params = toml.load(f'{output_directory}/backtesting_params.toml')
        storage_params = toml.load(f'{output_directory}/storage_params.toml')

        feeds = backtesting_params.get("FEEDS")
        os.environ['SIGNAL__MODEL'] = backtesting_params.get("MODEL_FILE")
        os.environ['SIGNAL__FEATURES'] = backtesting_params.get("FEATURES_FILE")
        os.environ['SIGNAL__SCALER'] = backtesting_params.get("SCALER_FILE")
        os.environ['SIGNAL__LIFETIME'] = str(backtesting_params.get("LIFETIME"))
        os.environ['SIGNAL__STORAGE_BUCKET'] = storage_params.get('BUCKET')

        # backtester from name
        backtester = SignalBackTesting.create_from_name(
            "mlearn.scikit_classifier",
            [instrument],
            output_directory,
            simulation=None,
            ordering=None,
            signal_plot=False,
            monitoring=None
        )

        # run backtester
        with recursionlimit(150000):
            # backtester start date
            result = backtester.start(
                PARAMETERS.get('BTEST_START_DATE'),
                PARAMETERS.get('BTEST_END_DATE'),
                from_db=True,
            )

        # get performance
        backtesting_report = f'{output_directory}/backtesting_perf.csv'
        perf = backtester.get_perf()
        try:
            perf.to_df().to_csv(backtesting_report)
        except:
            with open(backtesting_report, 'w') as fw:
                fw.write('')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        backtesting_report = f'{output_directory}/backtesting_perf.csv'
        yield {
            'name': f"{instrument}_backtesting",
            'verbosity': 2,
            'actions': [(backtesting, [instrument])],
            'targets': [
                backtesting_report
            ],
            'file_dep': [
                f"{output_directory}/{instrument}_tpot.model_uploaded",
                f"{output_directory}/{instrument}_tpot.features_uploaded",
                f'{output_directory}/backtesting_params.toml',
                f"{output_directory}/storage_params.toml"
            ],
        }


def task_report():
    """
    Task report

    """
    columns = ['start_date', 'end_date', 'cum_profit', 'symbol.SELL', 'sharp_ratio', 'number_of_trades',
               'source_name.SELL']

    def report(report_data):
        """
        Reporting

        """
        data = list()
        for report in report_data:
            d = dict()
            try:
                instrument, learning_crosstab, learning_perf, backtesting_report = report
                d['instrument'] = instrument
                d['btest_start_date'] = PARAMETERS['BTEST_START_DATE']
                d['btest_end_date'] = PARAMETERS['BTEST_END_DATE']
                d['dset_start_date'] = PARAMETERS['DSET_START_DATE']
                d['dset_end_date'] = PARAMETERS['DSET_END_DATE']
                d['btest_lifetime'] = PARAMETERS['BTEST_LIFETIME']
                d['window'] = PARAMETERS['WINDOW']

                # process learning perf
                learning_perf = pd.read_csv(learning_perf)

                # feature 0
                try:
                    f1_score, precision, recall, support = learning_perf['0']
                    d['f1_score_0'] = f1_score
                    d['precision_0'] = precision
                    d['recall_0'] = recall
                    d['support_0'] = support
                except:
                    pass

                # feature 1
                try:
                    f1_score, precision, recall, support = learning_perf['1']
                    d['f1_score_1'] = f1_score
                    d['precision_1'] = precision
                    d['recall_1'] = recall
                    d['support_1'] = support
                except:
                    pass

                # process backtesting perf
                try:
                    backtesting_perf = pd.read_csv(backtesting_report)
                    number_of_trades = len(backtesting_perf.index)
                    records = backtesting_perf.to_dict('records')
                    record = records[-1]
                    d.update(record)
                    d['number_of_trades'] = number_of_trades
                except Exception as error:
                    print(f"Backtesting error: {error}!!")

            except Exception as error:
                print(f"Error: {error}!!")

            # dump
            finally:
                data.append(d)
                joblib.dump(d, f'{output_directory}/{instrument}_tpot.stats')

        dataset = pd.DataFrame(data)

        report_params = toml.load(f'{output_directory}/report_params.toml')
        dataset_name = report_params.get('REPORT_NAME')

        database_name = PARAMETERS['DATABASE_NAME']
        database_host = PARAMETERS['DATABASE_HOST']
        engine = create_engine(f'postgresql://:@{database_host}/{database_name.lower()}')
        dataset.to_sql(dataset_name, engine, if_exists='replace', index=True)

    # csv files
    report_data = list()
    file_dep = list()
    target = list()
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        learning_crosstab = f'{output_directory}/learning_crosstab.csv'
        learning_perf = f'{output_directory}/learning_perf.csv'
        backtesting_report = f'{output_directory}/backtesting_perf.csv'
        report_params = f'{output_directory}/report_params.toml'
        report_stats = f'{output_directory}/{instrument}_tpot.stats'
        report_data.append((instrument, learning_crosstab, learning_perf, backtesting_report))
        file_dep.append(learning_crosstab)
        file_dep.append(learning_perf)
        file_dep.append(backtesting_report)
        file_dep.append(report_params)
        target.append(report_stats)

    return {
        'verbosity': 2,
        'actions': [(report, [report_data])],
        'file_dep': file_dep,
        'targets': target
    }


def task_upload_stats():
    """
    Upload stats to S3

    """

    def upload_stats(instrument, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        storage_params = toml.load(f"{output_directory}/storage_params.toml")

        if PARAMETERS.get('STORE'):
            stats_file = f'{output_directory}/{instrument}_tpot.stats'
            print(f"  -> upload stats: {stats_file}")
            mlearn_storage = MLearnStorage.create()
            mlearn_storage.upload(stats_file, bucket=storage_params.get('BUCKET'))
            target = f"{output_directory}/{instrument}_tpot.stats_uploaded"
            with open(target, "w") as fw:
                fw.write('UPLOADED')

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        yield {
            'name': f"{instrument}_upload_stats_to_s3",
            'actions': [(upload_stats, [instrument])],
            'file_dep': [
                f"{output_directory}/{instrument}_tpot.stats",
                f"{output_directory}/storage_params.toml"
            ],
            'targets': [
                f"{output_directory}/{instrument}_tpot.stats_uploaded"
            ],
            'verbosity': 2
        }
