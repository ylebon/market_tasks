import boto3
import os

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Download model from S3

    """

    def run(self, s3_path, local_directory, bucket_name="varatra-models"):
        """Download from s3"""
        # s3
        services_config = ConfigServices.create()

        s3 = boto3.client(
            's3',
            aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
            aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
        )

        # create directory
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        # list files
        resp = s3.list_objects_v2(**{"Bucket": bucket_name, "Prefix": s3_path})
        remote_files = []
        if resp['KeyCount']:
            for obj in resp['Contents']:
                remote_files.append(obj['Key'])
        # download
        for remote_file in remote_files:
            local_file = os.path.abspath(
                os.path.join(local_directory, s3_path, os.path.basename(remote_file))
            )
            self.log.info(f"msg='downloading file' file={remote_file} to {local_file}")
            if not os.path.exists(os.path.dirname(local_file)):
                os.makedirs(os.path.dirname(local_file))
            with open(local_file, 'wb') as data:
                s3.download_fileobj(bucket_name, remote_file, data)
        return os.path.abspath(os.path.join(local_directory, s3_path))
