import fnmatch
import os

import boto3

from config.core.config_services import ConfigServices
from core.task_step import TaskStep


class Task(TaskStep):
    """
    Task Sync S3

    Sync Parquet S3 data to local directory
    """

    s3 = boto3.client(
        's3',
        aws_access_key_id=ConfigServices.create().get_value("AMAZON.ACCESS_KEY"),
        aws_secret_access_key=ConfigServices.create().get_value("AMAZON.SECRET_KEY"),
        region_name="eu-west-1"
    )

    def run(self, output_directory=None, bucket_name=None, prefix=None, force=False,
            parquet_filter=None):
        """
        Sync amazon s3 data

        """
        services_config = ConfigServices.create()
        # output directory
        if not output_directory:
            output_directory = services_config.get_value('PARQUET.LOCAL_DIRECTORY')
        output_directory = os.path.abspath(os.path.expanduser(output_directory))

        # create output directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # bucket name
        bucket_name = bucket_name or services_config.get_value('PARQUET.S3_BUCKET')
        bucket_name = bucket_name.replace("s3://", "")

        self.log.info(f"msg='syncing amazon3 locally' bucket='{bucket_name}' directory={output_directory}")

        # list objects
        paginator = self.s3.get_paginator("list_objects")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        # create files to sync
        files_list = list()
        for page in page_iterator:
            if "Contents" in page:
                for key in page["Contents"]:
                    file_key = key["Key"]
                    files_list.append(file_key)
        self.log.info(f"msg='total number of files in S3' total='{len(files_list)}'")

        # Filter parquet
        parquet_filter = parquet_filter or dict()
        exchange = parquet_filter.get("exchange", "*")
        symbol = parquet_filter.get("symbol", "*")
        date = parquet_filter.get("date", "*")
        pattern = f"exchange={exchange}/symbol={symbol}/date={date}/*.parquet"

        matching_files = fnmatch.filter(files_list, pattern)
        self.log.info(f"msg='total maching files in S3' pattern='{pattern}' total='{len(matching_files)}'")

        # Download parquet file
        for parquet_file in matching_files:
            local_file = os.path.join(output_directory, parquet_file)
            if not os.path.exists(local_file) or force:
                self.log.info("msg='downloading bucket file' dst='{0}'".format(local_file))
                local_directory = os.path.dirname(local_file)
                if not os.path.exists(local_directory):
                    os.makedirs(local_directory)
                self.s3.download_file(bucket_name, parquet_file, local_file)


if __name__ == "__main__":
    from logbook import StreamHandler
    import sys

    StreamHandler(sys.stdout).push_application()
    task = Task("sync_s3")
    task.run()
