import os
import sys

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))

sys.path.append(BASE_DIR)

import os
import pandas as pd

from varatra_backtesting.core.backesting import SignalBackTesting
from varatra_patterns.recursionlimit import recursionlimit

from varatra_utils import time_util
from doit import get_var
import toml
from collections import OrderedDict
from logbook import FileHandler
from varatra_tasks.core.tasks_factory import TasksFactory
import re
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import uuid

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))
OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "varatra_logs", "auto_backtesting")

# DEFAULT PARAMETERS
DEFAULT_PARAMETERS = {
    "BTEST_START_DATE": -1,
    "BTEST_END_DATE": 0,
    "OUTPUT_DIRECTORY": OUTPUT_DIRECTORY,
    "DATABASE_NAME": "TEST",
    "SIGNALERS": OrderedDict({
        'bbands.cross': {

        },
        'turtle.turtle': {

        },
        'bbands.double': {

        },
        'bbands.percentile': {

        },
        'bbands.turtle': {

        }
        ,
        'moving_average.simple': {

        },
        'opportunistic.crash': {

        },
        'rsi.tick_price': {

        },
        'percentile.book': {

        },
        'mlearn.scikit_classifier': {
            'SIGNAL__MODEL': '{instrument}_tpot.model',
            'SIGNAL__FEATURES': '{instrument}_tpot.features',
            'SIGNAL__SCALER': '{instrument}.scaler',
            'SIGNAL__LIFETIME': '601',
            'SIGNAL__STORAGE_BUCKET': 'varatra-models-prod'
        }
    }),
    "INSTRUMENTS": list()
}

PARAMETERS = dict()


def start_logging():
    """
    Start logging

    """
    # create output directory
    if not os.path.exists(PARAMETERS['OUTPUT_DIRECTORY']):
        os.makedirs(PARAMETERS['OUTPUT_DIRECTORY'])
    # create log file
    log_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], "auto_backtesting.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    # specify file handler
    FileHandler(log_file, level='INFO').push_application()


def load_parameters():
    """
    Setup parameters

    """
    # PARAMETERS
    PARAMETERS['INSTRUMENTS'] = get_var('instrument', DEFAULT_PARAMETERS["INSTRUMENTS"])

    PARAMETERS['DATABASE_NAME'] = get_var(
        'db_name',
        os.environ.get('DATABASE_NAME', DEFAULT_PARAMETERS["DATABASE_NAME"])
    )

    PARAMETERS['BTEST_START_DATE'] = time_util.get_date(
        int(get_var('btest_start', DEFAULT_PARAMETERS["BTEST_START_DATE"]))
    )

    PARAMETERS['BTEST_END_DATE'] = time_util.get_date(
        int(get_var('btest_end', DEFAULT_PARAMETERS["BTEST_END_DATE"]))
    )

    # CHECK BTEST LAST DURATION
    PARAMETERS['BTEST_LAST_DURATION'] = int(get_var(
        'btest_last_duration',
        os.environ.get('BTEST_LAST_DURATION', 0)
    ))

    # CHECK BTEST
    now = datetime.now()
    fmt = '%Y-%m-%d %H:%M:%S'

    if PARAMETERS['BTEST_LAST_DURATION']:
        PARAMETERS['BTEST_START_DATE'] = (now - timedelta(hours=PARAMETERS['BTEST_LAST_DURATION'])).strftime(fmt)
        PARAMETERS['BTEST_END_DATE'] = now.strftime(fmt)

    PARAMETERS['OUTPUT_DIRECTORY'] = get_var('output_directory', DEFAULT_PARAMETERS["OUTPUT_DIRECTORY"])


    # ESTIMATORS
    if get_var('signalers'):
        signaler_list = get_var('signalers').split(",")
    else:
        signaler_list = DEFAULT_PARAMETERS["SIGNALERS"].keys()

    # SIGNALERS
    PARAMETERS["SIGNALERS"] = list()
    for signaler_name in signaler_list:
        signaler_params = DEFAULT_PARAMETERS['SIGNALERS'][signaler_name]
        PARAMETERS["SIGNALERS"].append((signaler_name, signaler_params))

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

    def create_params(instrument, signaler_name, signaler_params, targets):
        """
        Create parameters

        """
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        ENV = dict()
        for k, v in signaler_params.items():
            ENV[k] = v.format(instrument=instrument)

        # backtesting params
        backtesting_params = {
            'START_DATE': PARAMETERS['BTEST_START_DATE'],
            'END_DATE': PARAMETERS['BTEST_END_DATE'],
            'ENV': ENV,
            'OUTPUT_DIRECTORY': output_directory,
            'PLOTTING': False,
            'FEEDS': [instrument],
        }
        with open(f'{output_directory}/backtesting_{signaler_name}.toml', "w") as out_file:
            toml.dump(backtesting_params, out_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for signaler_name, signaler_params in PARAMETERS['SIGNALERS']:
            yield {
                'name': f"{instrument}_{signaler_name}_params",
                'actions': [(create_params, [instrument, signaler_name, signaler_params])],
                'targets': [
                    f"{output_directory}/backtesting_{signaler_name}.toml",
                ],
            }


def task_backtesting():
    """
    Run backtesting

    """

    def backtesting(instrument, signaler_name, signaler_params, targets):
        # load parameters
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        backtesting_params = toml.load(f'{output_directory}/backtesting_{signaler_name}.toml')

        feeds = backtesting_params.get("FEEDS")
        start_date = backtesting_params.get("START_DATE")
        end_date = backtesting_params.get("END_DATE")
        plotting = backtesting_params.get("PLOTTING")
        env = backtesting_params.get("ENV")

        # update environment
        os.environ.update(env)

        # backtester from name
        backtester = SignalBackTesting.create_from_name(
            signaler_name,
            [instrument],
            output_directory,
            simulation=None,
            ordering=None,
            signal_plot=plotting,
            monitoring=False
        )

        # run backtester
        with recursionlimit(150000):
            # backtester start date
            result = backtester.start(
                start_date,
                end_date,
                from_db=True,
            )

        # get performance
        backtesting_report = f'{output_directory}/backtesting_{signaler_name}_{instrument}.csv'
        perf = backtester.get_perf()
        perf.to_df().to_csv(backtesting_report)

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for signaler_name, signaler_params in PARAMETERS['SIGNALERS']:
            backtesting_report = f'{output_directory}/backtesting_{signaler_name}_{instrument}.csv'
            yield {
                'name': f"{instrument}_{signaler_name}_backtesting",
                'verbosity': 2,
                'actions': [(backtesting, [instrument, signaler_name, signaler_params])],
                'targets': [
                    backtesting_report
                ],
                'file_dep': [
                    f'{output_directory}/backtesting_{signaler_name}.toml'
                ],
            }


def task_report():
    """
    Task report

    """
    columns = ['start_date', 'end_date', 'cum_profit', 'symbol.SELL', 'sharp_ratio', 'number_of_trades',
               'source_name.SELL']

    def report(csv_files):
        data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            number_of_trades = len(df.index)
            if len(df.index):
                records = df.to_dict('records')
                record = records[-1]
                record['start_date'] = PARAMETERS['BTEST_START_DATE']
                record['end_date'] = PARAMETERS['BTEST_END_DATE']
                record['number_of_trades'] = number_of_trades
                record['uuid'] = str(uuid.uuid4())
                data.append(record)

        dfs = pd.DataFrame(data)

        # create database
        db_name = PARAMETERS['DATABASE_NAME']
        POSTGRESL_URL = os.environ['POSTGRES_URL']
        engine = create_engine(POSTGRESL_URL)
        for signaler_name, df in dfs.groupby(by='source_name.SELL'):
            df.sort_values(by='sharp_ratio', ascending=False, inplace=True)
            df.to_sql(signaler_name.replace(".", "__"), engine, if_exists='replace', index=True)

    csv_files = list()
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for signaler_name, signaler_params in PARAMETERS['SIGNALERS']:
            backtesting_report = f'{output_directory}/backtesting_{signaler_name}_{instrument}.csv'
            csv_files.append(backtesting_report)

    return {
        'verbosity': 2,
        'actions': [(report, [csv_files])],
        'file_dep': csv_files
    }
