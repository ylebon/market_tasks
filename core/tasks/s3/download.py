import boto3
import os

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Deploy lambda function
    """
    s3 = None

    def run(self, bucket_name, remote_src, local_dst):
        """Deploy lambda function"""

        if not self.__class__.s3:
            self.__class__.s3 = boto3.client(
                's3',
                aws_access_key_id=ConfigServices.create().get_value("AMAZON.ACCESS_KEY"),
                aws_secret_access_key=ConfigServices.create().get_value("AMAZON.SECRET_KEY"),
            )

        self.log.info("msg='download amazon file' src='{}' dst='{}'".format(remote_src, local_dst))
        if os.path.isdir(local_dst):
            local_dst = os.path.join(local_dst, os.path.basename(remote_src))
        self.s3.download_file(bucket_name, remote_src, local_dst)
        return local_dst
