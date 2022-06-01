import os
import sys

import pandas as pd
from logbook import StreamHandler
from tabulate import tabulate

from features.core.feature_executor import FeatureExecutor
from features.core.feature_loader import FeatureLoader
from core.tasks.features import load_parquet
from core.tasks.features import select_importances
from core.tasks.features import sync_s3

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()
from logbook import Logger


def launch(instrument_id, features, target, limit=None, env_context=None):
    """
    Run machine learner

    """
    log = Logger("feature_importance")

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
    print(tabulate(df.count().to_frame(), headers=['count']))
    # drop nan
    df.dropna(inplace=True)
    log.info(f"msg='dataset length after dropping nan' length='{len(df)}' dropped='{len(df)- len_after_load}'")

    # create X, y
    features_valid = [c for c in df.columns if c in features]
    X = df[features_valid]
    y = df[target]

    # selectors
    selectors = os.getenv("SELECTORS", None)
    if selectors:
        selectors = selectors.split(",")

    log.info(f"msg='creating features importances' selectors='{selectors}'")

    t_select_importances = select_importances.Task("select_importances")
    result = t_select_importances.run(X, y, limit=limit, selectors=selectors)
    for key, value in result.items():
        print(f"------ {key} -------")
        df = pd.DataFrame(value)
        csv_file = os.path.join(f"/tmp/{key}.csv")
        df.to_csv(csv_file)
        log.info(f"msg='features importance generated' key='{key}' csv='{csv_file}'")


if __name__ == "__main__":
    launch("BINANCE_BTC_USDT", ["@variation_price"], 'profit__binary__30M')
