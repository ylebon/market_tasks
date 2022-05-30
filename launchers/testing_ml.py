import os
import sys

from logbook import StreamHandler

from varatra_tasks.core.tasks.features import sync_s3 as sync_features
from varatra_tasks.core.tasks.feeds import load_parquet
from varatra_tasks.core.tasks.model import sync_s3 as sync_model
from varatra_tasks.core.tasks.model import testing

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()
from logbook import Logger


def launch(instrument_id, target, env_context=None, model_type='*', date="*"):
    """
    Run ML testing

    """
    logger = Logger("launcher_testing_ml")
    if env_context:
        os.environ.update(env_context)

    # sync parquet
    t_sync_s3 = sync_features.Task("sync_s3")
    exchange, symbol = instrument_id.split("_", 1)
    parquet_filter = {'exchange': exchange, 'symbol': symbol}
    t_sync_s3.run(parquet_filter=parquet_filter)

    # load parquet
    t_load_parquet = load_parquet.Task("load_parquet")
    df = t_load_parquet.run(instrument_id, date=date)
    df.dropna(inplace=True)

    # sync model
    t_sync_model = sync_model.Task("sync_model")
    files = t_sync_model.run(instrument_id, model_type=model_type, model_target=target)

    # testing model
    model_key_list = list(set([x for x in files if os.path.basename(x)]))
    model_testing = testing.Task("testing")
    for model_key in model_key_list:
        logger.info(f"msg='running test mode' model='{model_key}'")


if __name__ == "__main__":
    launch("OANDA_GBP_USD", 'profit__binary__10M', date="*")
