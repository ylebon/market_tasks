from datetime import datetime

from influxdb import InfluxDBClient
from tzlocal import get_localzone

from config.core.config_services import ConfigServices
from core.task_step import TaskStep
from utils import time_util


class Task(TaskStep):
    """
    Load instrument influx data

    """

    def run(self, instrument_id, start_date, end_date, start_hour="00:00:00.000", end_hour="23:59:59.999", fields="*"):
        """
        Download influxdb points

        """

        services_config = ConfigServices.create()
        db_client = InfluxDBClient(
            host=services_config.get_value('INFLUXDB.HOST'),
            port=services_config.get_value('INFLUXDB.PORT'),
            username=services_config.get_value('INFLUXDB.USERNAME'),
            password=services_config.get_value('INFLUXDB.PASSWORD'),
            database=services_config.get_value('INFLUXDB.DATABASE_PRICES')
        )

        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            datetime.strptime(start_date, fmt)
            start_date += '.000'
        except ValueError:
            start_date = time_util.string_to_date(" ".join((start_date, start_hour)))

        try:
            datetime.strptime(end_date, fmt)
            end_date += '.999'
        except ValueError:
            end_date = time_util.string_to_date(" ".join((end_date, end_hour)))

        if isinstance(instrument_id, tuple):
            instrument_str = "_".join(instrument_id)
        else:
            instrument_str = instrument_id
            instrument_id = instrument_id.split("_", 1)

        query = f"SELECT {fields} FROM {instrument_str} WHERE '{start_date}' < time and time < '{end_date}' TZ('{get_localzone().zone}')"
        self.log.info(
            "msg='downloading symbol prices' instrument_id={instrument_id}".format(instrument_id=instrument_id))
        self.log.info("msg='executing database query'  query={query}".format(query=query))
        result = db_client.query(query)
        points = list(result.get_points())
        if len(points) > 2:
            self.log.info(
                "msg='finish downloading new points' size={size} instrument_id={instrument_id} start_date={start_date} end_date={end_date}".format(
                    size=len(points), instrument_id=instrument_id, start_date=points[0]['time'],
                    end_date=points[-1]['time']))
        else:
            self.log.warn("msg='no data for specified date range' start_date={start_date} end_date={end_date}".format(
                start_date=start_date, end_date=end_date))
        return points
