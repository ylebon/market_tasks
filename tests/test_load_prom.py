from varatra_tasks.core.tasks.histdata import load_prom


def test_loading():
    """
    Test prometheus loading

    """
    t = load_prom.Task("load_prom")
    metrics_values = t.run("BINANCE_XRP_ETH", "2019-12-23", "2019-12-23", start_hour="01:50:00.001", end_hour="01:59:59.999")