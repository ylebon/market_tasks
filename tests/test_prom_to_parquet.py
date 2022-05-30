from varatra_tasks.core.tasks.histdata import load_prom
from varatra_tasks.core.tasks.histdata import prom_to_dataframe
from varatra_tasks.core.tasks.histdata import dataframe_to_parquet
from varatra_services.services_config import ConfigServices
import os

def test_loading():
    """
    Test prometheus loading

    """
    t_1 = load_prom.Task("load_prom")
    metrics_values = t_1.run("BINANCE_ETH_BTC", "2019-12-23", "2019-12-23", start_hour="01:50:00.001", end_hour="01:59:59.999")

    t_2 = prom_to_dataframe.Task("prom_to_dataframe")
    dataframe = t_2.run(metrics_values)

    os.environ['PARQUET.LOCAL_DIRECTORY'] = "/Users/madazone/Desktop/parquet"

    t_3 = dataframe_to_parquet.Task("dataframe_to_parquet")
    t_3.run(dataframe)
