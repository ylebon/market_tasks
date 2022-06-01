import os

import boto3

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Load metadata

    """
    s3 = None
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, s3_file, model_name=None):
        self.log.info(f"msg='loading metadata from file'")
        services_config = ConfigServices.create()
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
            aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
        )

        bucket_name = services_config.get_value('MODEL.S3_BUCKET').replace("s3://", "")

        result = self.s3.get_object(Bucket=bucket_name, Key=s3_file)
        text = result["Body"].read().decode()

        return text


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    t = Task("script_to_s3")
    r = t.run(
        "exchange=BITSTAMP/symbol=BTC_USD/target=profit__binary__10M/type=AUTOML/run_id=20200128_000954/meta.json")
    print(r)
