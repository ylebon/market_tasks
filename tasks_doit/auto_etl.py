import sys
from pathlib import Path

lib_dir = '/Users/madazone/Workspace/signaler'
sys.path.append(lib_dir)

from tabulate import tabulate
import os
import pprint
import pandas as pd
import time

from mlearn.core.mlearn_storage import MLearnStorage
from backtesting.core.backtesting_runner import BacktestingRunner
from utils import time_util
from doit import get_var
import json
import toml
import time
from collections import OrderedDict
import sys
from logbook import Logger, FileHandler
from core.tasks_factory import TasksFactory
import re
from distributed import Client
from doit.tools import result_dep
from doit.tools import timeout
import datetime
import asyncio
from features.core.features_loader import FeaturesLoader
from features.core.feature import Feature

# DEFAULT PARAMETERS
DEFAULT_PARAMETERS = {
    "DSET_START_DATE": -30,
    "DSET_END_DATE": 0,
    "UPTODATE_CHECK": 30,
    "OUTPUT_DIRECTORY": "/Users/madazone/Workspace/signaler/doit/output/etl",
    "INSTRUMENTS": [
        'OANDA_EUR_USD',
        'OANDA_GBP_USD',
        'OANDA_USD_CHF',
        'OANDA_AUD_USD',
    ]
}

PARAMETERS = dict()
DEFINITIONS_FILE = "/Users/madazone/Workspace/signaler/features/data"
DASK_SCHEDULER = '51.15.110.238:8786'
SCHEDULER_CLIENT = Client(DASK_SCHEDULER)


def start_logging():
    """
    Start logging

    """
    if not os.path.exists(PARAMETERS['OUTPUT_DIRECTORY']):
        os.makedirs(PARAMETERS['OUTPUT_DIRECTORY'])
    log_file = os.path.join(PARAMETERS['OUTPUT_DIRECTORY'], "auto_etl.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    FileHandler(log_file, level='INFO').push_application()


def load_parameters():
    """
    Setup parameters

    """
    # PARAMETERS
    PARAMETERS['INSTRUMENTS'] = get_var('instrument', DEFAULT_PARAMETERS["INSTRUMENTS"])
    PARAMETERS['UPTODATE_CHECK'] = int(get_var('uptodate_check', DEFAULT_PARAMETERS["UPTODATE_CHECK"]))
    PARAMETERS['DSET_START_DATE'] =  time_util.get_date(DEFAULT_PARAMETERS["DSET_START_DATE"])
    PARAMETERS['DSET_END_DATE'] =  time_util.get_date(DEFAULT_PARAMETERS["DSET_END_DATE"])
    PARAMETERS['OUTPUT_DIRECTORY'] = get_var('output_directory', DEFAULT_PARAMETERS["OUTPUT_DIRECTORY"])


    # INSTRUMENTS
    tasks = TasksFactory.create()

    # filter feeds
    if get_var('instruments'):
        PARAMETERS["INSTRUMENTS"] = get_var('instruments').split(",")
    else:
        PARAMETERS["INSTRUMENTS"] = list()
        db_client = database.create_client.run()
        pattern = get_var('pattern', '.*')
        for feed in database.series_list.run(db_client):
            if re.match(pattern, feed):
                PARAMETERS["INSTRUMENTS"].append(feed)
            else:
                pass

# load parameters
load_parameters()

# start logging
start_logging()

tasks = TasksFactory.create()

def task_feed_dates():
    """
    Run feed

    """
    def feed_dates(feed):
        db_client = database.create_client.run()
        query_first = f"SELECT first(bid_price) from {feed}"
        query_last = f"SELECT last(bid_price) from {feed}"
        result_first = db_client.query(query_first)
        result_last = db_client.query(query_last)

        points_first = list(result_first.get_points())
        first_time, _ = points_first[0]

        points_last = list(result_last.get_points())
        last_time, _ = points_last[0]

        print(f'  -> start date: {first_time} to end date: {last_time}')

        return {'first': first_time, 'last': last_time}


    for feed in PARAMETERS['INSTRUMENTS']:
        yield {
            'name': f"{feed}_dates",
            'verbosity': 2,
            'actions': [(feed_dates, [feed])],
        }

def task_store_feed():
    """
    Run feed storage

    """

    def store_feed(instrument):
        points = histdata.load_influx.run(
            instrument,
            PARAMETERS['DSET_START_DATE'],
            PARAMETERS['DSET_END_DATE']
        )

        # create series
        dset = pd.DataFrame(list(points)).set_index('time')
        dset.index = pd.to_datetime(dset.index)

        # publish dataset
        dset_name = f'{instrument}_TICKS'
        datasets = SCHEDULER_CLIENT.list_datasets()
        if dset_name in datasets:
            SCHEDULER_CLIENT.unpublish_dataset(dset_name)
            time.sleep(10)
        SCHEDULER_CLIENT.publish_dataset(dset, name=dset_name)
        print(f'  -> dataset published: {dset_name}')


    
    for instrument in PARAMETERS['INSTRUMENTS']:
        yield {
            'name': f"{instrument}_store",
            'verbosity': 2,
            'actions': [(store_feed, [instrument])],
            'uptodate': [timeout(datetime.timedelta(minutes=PARAMETERS['UPTODATE_CHECK']))],
        }

def task_create_features():
    """
    Create features

    """

    def create_features(feed, targets):
        dset_name = f"{feed}_TICKS"
        dset = SCHEDULER_CLIENT.get_dataset(dset_name)
        print(f'  -> dataset downloaded')
        patterns = [".*diff.*", ".*stochastic.*", ".*profit.*", ".*std.*", ".*momentum.*"]

        # features loader
        features_loader = FeaturesLoader.from_file(definitions_file)
        featurs_names = features_loader.filter_patterns(patterns)
        params = {'feed': feed}
        
        def create_feature(feature_name):
            """
            Create feature
            
            """
            feature = Feature.from_name(feature_name, definitions=features_loader, params=params)
            feature.load_data()

            return 1
        
        futures = SCHEDULER_CLIENT.map(create_feature, features_names)
        results = SCHEDULER_CLIENT.gather(futures)
        print(results)
        


    for instrument in PARAMETERS['INSTRUMENTS']:
        yield {
            'name': f'{instrument}_features',
            'actions': [(create_features, [instrument])],
            'verbosity': 2
        }
