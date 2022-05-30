import os
import sys

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))

sys.path.append(BASE_DIR)

import os

import pandas as pd
from doit.tools import config_changed

from varatra_utils import time_util
from doit import get_var

import toml
import time
import re
from varatra_tasks.core.tasks_factory import TasksFactory
import dask.dataframe as dd
from distributed import Client
import itertools
import numpy as np

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))
DEFINITIONS_FILE = os.path.join(BASE_DIR, "varatra_features", "data")
OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "varatra_logs", "auto_portfolio")

DEFAULT_PARAMETERS = {
    "STORE": True,
    "INSTRUMENTS": [],
    "OUTPUT_DIRECTORY": OUTPUT_DIRECTORY,
    "DSET_START_DATE": -1,
    "DSET_END_DATE": -1,
    "BTEST_START_DATE": 0,
    "BTEST_END_DATE": 0,
    "WINDOW": 14400,
    "DASK_SCHEDULER": "51.15.102.144:8786",
    "REPORT_NAME": "auto_portfolio_oanda_correlation_portfolio"

}

PARAMETERS = dict()
CONTEXT = dict()


def start_logging():
    from logbook import FileHandler
    log_file = os.path.join(OUTPUT_DIRECTORY, "auto_portfolio.log")
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
    PARAMETERS['OUTPUT_DIRECTORY'] = get_var(
        'output_directory',
        DEFAULT_PARAMETERS["OUTPUT_DIRECTORY"]
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

    # DASK SCHEDULER
    PARAMETERS['DASK_SCHEDULER'] = get_var(
        'dask_scheduler',
        os.environ.get('DASK_SCHEDULER', DEFAULT_PARAMETERS["DASK_SCHEDULER"])
    )

    # WINDOW
    PARAMETERS['WINDOW'] = int(get_var(
        'window',
        os.environ.get('WINDOW', DEFAULT_PARAMETERS["WINDOW"])
    )
    )

    # DATASET
    PARAMETERS['DSET_START_DATE'] = time_util.get_date(int(get_var(
        'dset_start_date',
        os.environ.get('DSET_START_DATE', DEFAULT_PARAMETERS["DSET_START_DATE"])))
    )
    PARAMETERS['DSET_END_DATE'] = time_util.get_date(int(get_var(
        'dset_end_date', os.environ.get('DSET_END_DATE', DEFAULT_PARAMETERS["DSET_END_DATE"])))
    )

    # REPORT NAME
    PARAMETERS['REPORT_NAME'] = get_var(
        'report_name',
        os.environ.get(
            'REPORT_NAME'
            , DEFAULT_PARAMETERS["REPORT_NAME"]
        )
    )

    # filter feeds
    tasks = TasksFactory.create()
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


def task_create_report_params():
    """
    Create report parameters

    """

    def create_report_params(report_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'])

        with open(f'{output_directory}/report_params.toml', "w") as out_file:
            toml.dump(report_params, out_file)

    # storage parameters
    report_params = {
        'REPORT_NAME': PARAMETERS['REPORT_NAME']
    }

    output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'])
    return {
        'actions': [(create_report_params, [report_params])],
        'uptodate': [config_changed(report_params)],
        'verbosity': 2,
        'targets': [
            f"{output_directory}/report_params.toml",
        ],
    }


def task_create_dataset_params():
    """
    Create dataset parameters

    """

    def create_dataset_params(instrument, dataset_params, targets):
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        # create directory
        with open(f'{output_directory}/dataset_params.toml', "w") as out_file:
            toml.dump(dataset_params, out_file)

    # create output directory
    for instrument in PARAMETERS['INSTRUMENTS']:
        output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)

        dataset_params = {
            'START_DATE': PARAMETERS['DSET_START_DATE'],
            'END_DATE': PARAMETERS['DSET_END_DATE'],
            'FIELDS': 'ask_price,bid_price',
        }

        yield {
            'name': f"{instrument}_dataset_params",
            'actions': [(create_dataset_params, [instrument, dataset_params])],
            'file_dep': [
                f'{output_directory}/CREATED.txt'
            ],
            'uptodate': [config_changed(dataset_params)],
            'verbosity': 2,
            'targets': [
                f"{output_directory}/dataset_params.toml",
            ],
        }


def task_load_data():
    """
    Task load data

    """

    tasks = TasksFactory.create()

    def load_data(instruments, targets):
        """
        Load data method

        """
        for instrument in instruments:
            output_directory = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument)
            parquet_file = os.path.join(output_directory, 'data_resampled_1s.parquet')

            # load dataset parameters
            dataset_params = toml.load(f"{output_directory}/dataset_params.toml")

            # load data
            points = tasks.histdata.load_influx.run(
                instrument,
                dataset_params.get('START_DATE'),
                dataset_params.get('END_DATE'),
                fields='ask_price,bid_price'
            )

            # create series
            data = list()
            for p in points:
                p['time'] = pd.Timestamp(p['time'])
                data.append(p)

            # create dataframe
            df = pd.DataFrame(data)
            df = df.set_index("time")
            df = df.resample("1s").bfill().dropna()

            # dask dataframe
            df.to_parquet(parquet_file, engine='pyarrow')
            print(f' -> instrument: {instrument}')

    instruments = PARAMETERS['INSTRUMENTS']

    data_files = [os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], x, 'data_resampled_1s.parquet') for x in instruments]
    data_params = [os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], x, 'dataset_params.toml') for x in instruments]

    return {
        'actions': [
            (load_data, [instruments])
        ],
        'targets': data_files,
        'file_dep': data_params,
        'verbosity': 2
    }


def task_measure_return():
    """
    Task measure return

    """

    tasks = TasksFactory.create()

    def measure_return(instrument, targets):
        # read parquet file
        data_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument, 'data_resampled_1s.parquet')
        ddf = dd.read_parquet(data_file)

        # create return
        shift = PARAMETERS['WINDOW']
        fee_in = ddf['ask_price'].shift(shift) - ddf['bid_price'].shift(shift)
        fee_out = ddf['ask_price'] - ddf['bid_price']
        fee = fee_in + fee_out
        ddf['return'] = (ddf['bid_price'].shift(shift) - ddf['ask_price'] - fee) / ddf['ask_price']
        df = ddf.compute()

        df.to_parquet(targets[0], engine='pyarrow')

    for instrument in PARAMETERS['INSTRUMENTS']:
        parquet_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument, 'data_resampled_1s.parquet')
        target_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument, 'data_return.parquet')

        yield {
            'name': f'{instrument}_return',
            'actions': [
                (measure_return, [instrument])
            ],
            'targets': [
                target_file
            ],
            'file_dep': [
                parquet_file
            ],
            'verbosity': 2
        }


def task_merge_return():
    """
    Task merge return

    """

    def merge_return(instruments, targets):
        # read parquet file
        dfs = list()
        for instrument in instruments:
            rolling_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], instrument, 'data_return.parquet')
            df = pd.read_parquet(rolling_file)
            df.rename(columns={'return': instrument}, inplace=True)
            dfs.append(df[instrument])

        # concats
        df = pd.concat(dfs, axis=1)
        df.dropna(inplace=True)
        df.to_parquet(targets[0])

    instruments = PARAMETERS['INSTRUMENTS']
    data_files = [os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], x, 'data_return.parquet') for x in instruments]

    return {
        'actions': [
            (merge_return, [instruments])
        ],
        'targets': [
            os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'data_return_merged.parquet')
        ],
        'file_dep': data_files,
        'verbosity': 2
    }


def task_correlation():
    """
    Task correlation

    """

    def correlation(targets):
        # correlation
        df = pd.read_parquet(os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'data_return_merged.parquet'))
        df_corr = df.corr(method='pearson')
        df_corr.to_csv(targets[0])
        print(df_corr)

    return {
        'actions': [
            (correlation, [])
        ],
        'targets': [
            os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'correlation.csv')
        ],
        'file_dep': [
            os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'data_return_merged.parquet')
        ],
        'verbosity': 2
    }


def task_create_portfolio():
    """
    Create portfolio

    """

    def create_portfolio(targets):
        instruments = PARAMETERS['INSTRUMENTS']
        portfolio_list = itertools.permutations(instruments, r=3)

        data = list()
        for portfolio in portfolio_list:
            df = pd.read_parquet(os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'data_return_merged.parquet'))
            df = df[list(portfolio)]
            returns = df.sum(axis=1)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(returns.index))
            pct_win = returns.ge(0).sum() / len(returns.index) * 100.
            r = {'portfolio': "_".join(portfolio), 'sharpe_ratio': sharpe_ratio, 'total': returns.sum(), 'pct_win': pct_win}
            data.append(r)

        df = pd.DataFrame(data).sort_values(by='sharpe_ratio', ascending=False).set_index('portfolio')
        print(df)
        df.to_parquet(targets[0])

    target_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], "portfolio.parquet")
    return {
        'actions': [
            (create_portfolio, [])
        ],
        'targets': [
            target_file
        ],
        'file_dep': [
            os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'data_return_merged.parquet')
        ],
        'verbosity': 2
    }


def task_report():
    """
    Task report

    """

    def report(targets):
        """
        Reporting

        """
        dataset = pd.read_csv(os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'correlation.csv'))
        report_params = toml.load(os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'report_params.toml'))
        dataset_name = report_params.get('REPORT_NAME')

        with Client(PARAMETERS['DASK_SCHEDULER']) as client:
            datasets = client.list_datasets()
            if dataset_name in datasets:
                client.unpublish_dataset(dataset_name)
                time.sleep(5)
            client.publish_dataset(dataset, name=dataset_name)
            print(f'  -> dataset published: {dataset_name}')

    return {
        'verbosity': 2,
        'actions': [(report, [])],
        'file_dep': [
            os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'correlation.csv'),
            os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], 'report_params.toml')
        ],
    }
