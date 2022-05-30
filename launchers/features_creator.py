import os
import sys

from logbook import StreamHandler

from varatra_features.core.feature_executor import FeatureExecutor
from varatra_utils import time_util
StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()


def launch(instrument_id, features, date, env_context=None):
    """
    Run backtesting

    """
    if env_context:
        os.environ.update(env_context)
    date = time_util.get_date(date, fmt="%Y%m%d")
    feature_executor = FeatureExecutor.create()
    feature_executor.create_dataset(instrument_id, features, date=date)


if __name__ == "__main__":
    launch("bbands.percentile", "BITSTAMP_XRP_USD", "2020-01-02", "2020-01-02", last_hours=24)
