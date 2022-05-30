import boto3
import os

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """S3 Backup

    """
    # services config
    services_config = ConfigServices.create()

    # s3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
        aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
    )

    def run(self, directory):
        """
        Upload model to s3

        """
        for filename in os.listdir(directory):
            directory = os.path.normpath(directory)
            local_file = os.path.join(directory, filename)
            if os.path.isfile(local_file):
                remote_file = os.path.basename(directory) + "/" + filename
                self.log.info("msg='upload file' file={}".format(local_file))
                self.s3.upload_file(local_file, self.bucket_name, remote_file)
