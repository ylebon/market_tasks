import os
import re
import sys
from os.path import expanduser

import toml
from doit import get_var
from logbook import StreamHandler

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(BASE_DIR)

from datetime import datetime, timedelta
from varatra_tasks.core.tasks.prometheus import get_metrics
from varatra_signaler.core.signaler_backtesting import SignalerBacktesting
from varatra_patterns.recursionlimit import recursionlimit
from varatra_tasks.core.tasks.backtesting import upload_report
from varatra_tasks.core.tasks.minio import upload_file

PARAMETERS = {
    'OUTPUT_DIRECTORY': os.path.join(expanduser("~"), 'varatra_output')
}


def start_logging():
    """
    Start logging

    """
    # create output directory
    if not os.path.exists(PARAMETERS['OUTPUT_DIRECTORY']):
        os.makedirs(PARAMETERS['OUTPUT_DIRECTORY'])

    # create log file
    log_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], "do_backtesting.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    # specify file handler
    StreamHandler(sys.stdout, level='INFO').push_application()


def load_config():
    """
    Setup parameters

    """
    # PARAMETERS
    PARAMETERS['INSTRUMENTS'] = get_var('instruments', PARAMETERS.get("INSTRUMENTS", None))
    PARAMETERS['START_DATE'] = get_var('start_date', PARAMETERS.get("START_DATE", None))
    PARAMETERS['END_DATE'] = get_var('end_date', PARAMETERS.get("END_DATE", None))
    PARAMETERS['LAST_DURATION'] = get_var('last_duration', PARAMETERS.get("LAST_DURATION", None))
    PARAMETERS['OUTPUT_DIRECTORY'] = get_var('output_directory', PARAMETERS.get("OUTPUT_DIRECTORY", None))
    PARAMETERS['SIGNALERS'] = get_var('signalers', PARAMETERS.get("SIGNALERS", []))
    PARAMETERS['INSTRUMENTS'] = get_var('instruments', PARAMETERS.get("INSTRUMENTS", []))

    # UPDATE LAST DURATION
    now = datetime.now()
    fmt = '%Y-%m-%d %H:%M:%S'
    if PARAMETERS['LAST_DURATION']:
        PARAMETERS['START_DATE'] = (now - timedelta(hours=PARAMETERS['LAST_DURATION'])).strftime(fmt)
        PARAMETERS['END_DATE'] = now.strftime(fmt)

    # UPDATE INSTRUMENTS
    if not PARAMETERS.get("INSTRUMENTS"):
        pattern = get_var('pattern', '.*')
        task = get_metrics.Task("get_metrics")
        [PARAMETERS["INSTRUMENTS"].append(feed) for feed in task.run(pattern) if re.match(pattern, feed)]


# load parameters
load_config()

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
        # output directory
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        ENV = dict()
        for k, v in signaler_params.items():
            ENV[k] = v.format(instrument=instrument)

        # backtesting params
        backtesting_params = {
            'START_DATE': PARAMETERS['START_DATE'],
            'END_DATE': PARAMETERS['END_DATE'],
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

    def backtesting(instrument, signaler_name):
        # load parameters
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        backtesting_params = toml.load(f'{output_directory}/backtesting_{signaler_name}.toml')

        start_date = backtesting_params.get("START_DATE")
        end_date = backtesting_params.get("END_DATE")
        plotting = backtesting_params.get("PLOTTING")
        env = backtesting_params.get("ENV")

        # update environment
        os.environ.update(env)

        # backtester from name
        backtester = SignalerBacktesting.create_from_name(
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
            backtester.start(
                start_date,
                end_date,
                from_db=True,
            )

        # get performance
        backtesting_report = f'{output_directory}/backtesting_{signaler_name}_{instrument}.csv'
        perf = backtester.get_perf()
        perf.to_df().to_csv(backtesting_report)

        # minio task upload file
        task = upload_file.Task("upload_file")
        task.run(PARAMETERS['BUCKET_NAME'], backtesting_report, f"{signaler_name}_{instrument}.csv")

    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
        for signaler_name, signaler_params in PARAMETERS['SIGNALERS']:
            backtesting_report = f'{output_directory}/backtesting_{signaler_name}_{instrument}.csv'
            yield {
                'name': f"{instrument}_{signaler_name}_backtesting",
                'verbosity': 2,
                'actions': [(backtesting, [instrument, signaler_name])],
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

    def report(csv_files):
        t_upload_report = upload_report.Task("upload_report")
        t_upload_report.run(csv_files)

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
