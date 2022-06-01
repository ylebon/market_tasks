import boto3
import os

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Deploy lambda function

    """
    # services config
    services_config = ConfigServices.create()

    # boto client
    s3 = boto3.client(
        's3',
        aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
        aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
    )

    def run(self, local_file, bucket_name, remote_file=None):
        """Deploy lambda function"""
        remote_file = remote_file or os.path.basename(local_file)
        self.s3.upload_file(local_file, bucket_name, remote_file)
        return remote_file
