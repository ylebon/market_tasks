from core.tasks.histdata import load_prom
from core.tasks.histdata import prom_to_dataframe


def test_loading():
    """
    Test prometheus loading

    """
    t = load_prom.Task("load_prom")
    metrics_values = t.run("BINANCE_ETH_BTC", "2019-12-23", "2019-12-23", start_hour="01:50:00.001",
                           end_hour="01:59:59.999")

    t_2 = prom_to_dataframe.Task("prom_to_dataframe")
    t_2.run(metrics_values)
