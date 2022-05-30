from influxdb import InfluxDBClient

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Create database client

    """

    def run(self, db_host=None, db_port=None, db_name=None, db_username=None, db_password=None):
        services_config = ConfigServices.create()
        db_host = db_host or services_config.get_value('INFLUXDB.HOST')
        db_port = db_port or services_config.get_value('INFLUXDB.PORT')
        db_name = db_name or services_config.get_value('INFLUXDB.DATABASE_PRICES')
        db_username = db_username or services_config.get_value('INFLUXDB.USERNAME')
        db_password = db_password or services_config.get_value('INFLUXDB.PASSWORD')

        self.log.info(
            f"msg='connecting to influx database' host='{db_host}' port='{db_port}' db='{db_name}'"
        )
        db_client = InfluxDBClient(host=db_host, port=db_port, database=db_name, username=db_username, password=db_password)
        return db_client
