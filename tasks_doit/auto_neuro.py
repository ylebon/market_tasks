import sys

lib_dir = '/Users/madazone/Workspace/varatra_signaler'
sys.path.append(lib_dir)

import os
import pandas as pd

from varatra_mlearn.core.mlearn_storage import MLearnStorage
from varatra_backtesting.core.backtesting_runner import BacktestingRunner
from varatra_utils import time_util
from doit import get_var
import toml
from logbook import FileHandler
from varatra_tasks.core.tasks_factory import TasksFactory
from varatra_neuro.neuro import TimeSeriesPrediction
import re
import joblib

# DEFAULT PARAMETERS
DEFAULT_PARAMETERS = {
    "DSET_START_DATE": -9,
    "DSET_END_DATE": -3,
    "BTEST_START_DATE": -2,
    "BTEST_END_DATE": 0,
    "OUTPUT_DIRECTORY": "/Users/madazone/Workspace/varatra_signaler/varatra_doit/output/neuro",
    "NEUROS": {
        "TimeSeries": {

        }

    },
    "STORE": True,
    "INSTRUMENTS": [
        'OANDA_EUR_USD',
        'OANDA_GBP_USD',
        'OANDA_USD_CHF',
        'OANDA_AUD_USD',
    ]
}

PARAMETERS = dict()


def start_logging():
    """
    Start logging

    """
    if not os.path.exists(PARAMETERS['OUTPUT_DIRECTORY']):
        os.makedirs(PARAMETERS['OUTPUT_DIRECTORY'])
    log_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], "auto_neuro.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    FileHandler(log_file, level='INFO').push_application()


def load_parameters():
    """
    Setup parameters

    """
    # PARAMETERS
    PARAMETERS['INSTRUMENTS'] = get_var('instrument', DEFAULT_PARAMETERS["INSTRUMENTS"])
    PARAMETERS['PREDICTION'] = get_var('prediction', 600)
    PARAMETERS['DSET_START_DATE'] = time_util.get_date(
        int(get_var('dset_start', DEFAULT_PARAMETERS["DSET_START_DATE"])))
    PARAMETERS['DSET_END_DATE'] = time_util.get_date(int(get_var('dtest_end', DEFAULT_PARAMETERS["DSET_END_DATE"])))
    PARAMETERS['BTEST_START_DATE'] = time_util.get_date(
        int(get_var('btest_start', DEFAULT_PARAMETERS["BTEST_START_DATE"])))
    PARAMETERS['BTEST_END_DATE'] = time_util.get_date(int(get_var('btest_end', DEFAULT_PARAMETERS["BTEST_END_DATE"])))
    PARAMETERS['OUTPUT_DIRECTORY'] = get_var('output_directory', DEFAULT_PARAMETERS["OUTPUT_DIRECTORY"])
    PARAMETERS['NEUROS'] = DEFAULT_PARAMETERS['NEUROS']
    PARAMETERS['STORE'] = DEFAULT_PARAMETERS['STORE']

    # INSTRUMENTS
    tasks = TasksFactory.create()

    # filter feeds
    if get_var('instruments'):
        PARAMETERS["INSTRUMENTS"] = get_var('instruments').split(",")
    else:
        PARAMETERS["INSTRUMENTS"] = list()
        db_client = tasks.database.create_client.run()
        pattern = get_var('pattern', '.*')
        for feed in tasks.database.series_list.run(db_client):
            if re.match(pattern, feed):
                PARAMETERS["INSTRUMENTS"].append(feed)
            else:
                pass


# load parameters
load_parameters()

# start logging
start_logging()


def task_create_params():
    """
    Create output directory

    """

    def create_params(instrument, neuro_name, neuro_params, targets):
        """
        Create parameters

        """
        # output directory
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # neuro config
        model_file = os.path.join(output_directory, f'{instrument}_{neuro_name}.model')
        neuro_config = {
            'START_DATE': PARAMETERS['DSET_START_DATE'],
            'END_DATE': PARAMETERS['DSET_END_DATE'],
            'INSTRUMENT': instrument,
            'PREDICTION': PARAMETERS['PREDICTION'],
            'MODEL_FILE': model_file
        }
        with open(f'{output_directory}/neuro_{neuro_name}.toml', "w") as out_file:
            toml.dump(neuro_config, out_file)

        # backtesting params
        backtesting_params = {
            'START_DATE': PARAMETERS['BTEST_START_DATE'],
            'END_DATE': PARAMETERS['BTEST_END_DATE'],
            'ENV': {
                'SIGNAL__MODEL': model_file,
                'SIGNAL__LIFETIME': int(PARAMETERS['PREDICTION']) + 1
            },
            'OUTPUT_DIRECTORY': output_directory,
            'PLOTTING': False,
            'FEEDS': [instrument],
        }
        with open(f'{output_directory}/backtesting_{neuro_name}.toml', "w") as out_file:
            toml.dump(neuro_params, out_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for neuro_name, neuro_params in PARAMETERS['NEUROS'].items():
            yield {
                'name': f"{instrument}_{neuro_name}_params",
                'actions': [(create_params, [instrument, neuro_name, neuro_params])],
                'targets': [
                    f"{output_directory}/backtesting_{neuro_name}.toml",
                    f"{output_directory}/neuro_{neuro_name}.toml",
                ],
            }


def task_run_neuro():
    """
    Create output directory

    """

    def run_neuro(instrument, neuro_name, neuro_params, targets):
        """
        Create parameters

        """
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        neuro_config = toml.load(f'{output_directory}/neuro_{neuro_name}.toml')

        # tasks
        tasks = TasksFactory.create()

        # points
        points = tasks.histdata.load_influx.run(
            neuro_config.get('INSTRUMENT'),
            neuro_config.get('START_DATE'),
            neuro_config.get('END_DATE'),
            fields='bid_price')

        # create series
        dset = pd.DataFrame(list(points)).set_index('time')
        dset.index = pd.to_datetime(dset.index)
        dset = dset.resample('1s').bfill()
        dset = dset.rename(columns={'bid_price': 'x'})
        ts_prediction = TimeSeriesPrediction()
        df = dset.copy()
        ts_prediction.fit(df)
        rmse = ts_prediction.test(dset, int(neuro_config.get('PREDICTION')))

        # create model neuro file
        model_file = neuro_config.get('MODEL_FILE')
        joblib.dump(ts_prediction, model_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for neuro_name, neuro_params in PARAMETERS['NEUROS'].items():
            yield {
                'name': f"{instrument}_{neuro_name}_params",
                'actions': [(run_neuro, [instrument, neuro_name, neuro_params])],
                'file_dep': [
                    f"{output_directory}/neuro_{neuro_name}.toml",
                ],
            }


def task_upload_model_to_s3():
    """
    Upload model file to S3

    """

    def upload_model_to_s3(instrument, neuro_name, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        neuro_config = toml.load(f'{output_directory}/neuro_{neuro_name}.toml')

        if PARAMETERS.get('STORE'):
            model_file = neuro_config.get('model_file')
            print(f"     -> upload model: {model_file}")
            mlearn_storage = MLearnStorage.create()
            mlearn_storage.upload(model_file)

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for neuro_name, neuro_params in PARAMETERS['NEUROS'].items():
            yield {
                'name': f"{instrument}_{neuro_name}_upload_model_to_s3",
                'actions': [(upload_model_to_s3, [instrument, neuro_name])],
                'file_dep': [
                    f"{output_directory}/{instrument}_{neuro_name}.model"
                ],
                'verbosity': 2
            }


def task_backtesting():
    """
    Run backtesting

    """

    def backtesting(instrument, neuro_name, neuro_params, targets):
        # load parameters
        backtesting_params = toml.load(f'{instrument}/backtesting_{neuro_name}.toml')

        signalers = ["mlearn.scikit_linear"]
        feeds = backtesting_params.get("FEEDS")
        start_date = backtesting_params.get('START_DATE')
        end_date = backtesting_params.get('END_DATE')

        os.environ.update(backtesting_params.get('ENV'))
        backtester_runner = BacktestingRunner.create(feeds, signalers, ".")
        backtester_runner.run(
            backtesting_params.get('START_DATE'),
            backtesting_params.get('END_DATE'),
            from_db=True,
            plotting=False
        )

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for neuro_name, neuro_params in PARAMETERS['NEUROS'].items():
            yield {
                'name': f"{instrument}_{neuro_name}",
                'verbosity': 2,
                'actions': [(backtesting, [instrument, neuro_name, neuro_params])],
                'file_dep': [
                    f'{output_directory}/backtesting_{neuro_name}.toml'
                ],
            }
