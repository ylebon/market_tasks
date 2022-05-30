import fnmatch
import os

import boto3

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """
    Sync S3

    """
    s3 = None
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    def run(self, instrument_id: str="*", model_target: str = "*", model_type="*", run_id="*", file_type="*", force=False):
        services_config = ConfigServices.create()
        output_directory = services_config.get_value('MODEL.LOCAL_DIRECTORY')
        output_directory = os.path.abspath(os.path.expanduser(output_directory))

        if not self.s3:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=services_config.get_value("AMAZON.ACCESS_KEY"),
                aws_secret_access_key=services_config.get_value("AMAZON.SECRET_KEY"),
            )

        # create output directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # bucket name
        bucket_name = services_config.get_value('MODEL.S3_BUCKET').replace("s3://", "")

        self.log.info(f"msg='syncing S3 models locally' bucket='{bucket_name}' directory={output_directory}")

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
        parquet_filter = dict(run_id=run_id, model_target=model_target, model_type=model_type)
        if instrument_id != "*":
            exchange, symbol = instrument_id.split("_", 1)
        else:
            exchange = "*"
            symbol = "*"
        model_target = parquet_filter.get("model_target", "*")
        model_type = parquet_filter.get("model_type", "*")
        run_id = parquet_filter.get("run_id", "*")

        pattern = f"exchange={exchange}/symbol={symbol}/target={model_target}/type={model_type.upper()}/run_id={run_id}/{file_type}"

        matching_files = fnmatch.filter(files_list, pattern)
        self.log.info(f"msg='total maching files in S3' pattern='{pattern}' total='{len(matching_files)}'")

        # Download parquet file
        result = list()
        for model_file in matching_files:
            local_file = os.path.join(output_directory, model_file)
            result.append(local_file)
            if not os.path.exists(local_file) or force:
                self.log.info("msg='downloading model file' dst='{0}'".format(local_file))
                local_directory = os.path.dirname(local_file)
                if not os.path.exists(local_directory):
                    os.makedirs(local_directory)
                self.s3.download_file(bucket_name, model_file, local_file)
        return result


if __name__ == "__main__":
    t = Task("script_to_s3")
    t.run("BITSTAMP_BTC_EUR")
