import requests

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Load instrument prometheus

    """

    def run(self, instrument_id, start_date, end_date, start_hour="00:00:00.001", end_hour="23:59:59.999", fields="*"):
        """
        Download influxdb points

        """

        services_config = ConfigServices.create()
        url = services_config.get_value('PROMETHEUS.URL')

        # define metrics
        metrics = ["seq", "bid_price", "ask_price", "bid_qty", "ask_qty", "exchange_timestamp", "marketdata_timestamp",
                   "status"]

        # load metrics
        start_date = f"{start_date}T{start_hour}Z"
        end_date = f"{end_date}T{end_hour}Z"

        # instrument id
        if isinstance(instrument_id, tuple):
            instrument_id = "_".join(instrument_id)
        exchange, symbol = instrument_id.split("_", 1)

        # result
        metrics_values = dict()
        for metric in metrics:
            self.log.info(f"msg='loading metric' instrument='{instrument_id}' metric='{metric}' start_date='{start_date}' end_date='{end_date}'")
            query_url = f"{url}/api/v1/export?match={instrument_id}_{metric}&start={start_date}&end={end_date}"
            response = requests.get(query_url)
            if response.text:
                data = response.json()
                for timestamp, value in zip(data['timestamps'], data['values']):
                    try:
                        metrics_values[timestamp][metric] = value
                    except KeyError:
                        metrics_values[timestamp] = {
                            'time': timestamp, metric: value, 'exchange': exchange, 'symbol': symbol
                        }
        return metrics_values


if __name__ == "__main__":
    t = Task("load_prom")
    r = t.run("OANDA_EURs_USD", "2020-01-05", "2020-01-05")
    print(r)
